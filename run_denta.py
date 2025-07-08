import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import os
import traceback
import scanpy as sc
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist # For pairwise distances in Wasserstein and MMD
import sys # For checking installed modules
from denta.dataset import *
from denta.models import SpectralEncoder, ScoreNet, FeatureDecoder, ScoreSDE, LaplacianPerturb
from denta.models import *
from denta.seed import *        
from denta.utils import *
# Check for required libraries
try:
    import torch
    import torch_geometric
    import torch_scatter
    import scanpy
    import anndata
    import ot # Python Optimal Transport library
except ImportError as e:
    print(f"Missing required library: {e}. Please install it using pip.")
    print("Example: pip install torch torch-geometric torch-scatter scanpy anndata pandas numpy scikit-learn scipy POT")
    sys.exit("Required libraries not found.")

torch.backends.cudnn.benchmark = True
# attempt to set device, fall back to CPU if CUDA is not available or fails
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    print(f"Error setting up CUDA device: {e}. Falling back to CPU.")
    device = torch.device('cpu')

print(f"Using device: {device}")

class Trainer:
    def __init__(self, in_dim, hid_dim, lat_dim, num_cell_types, pe_dim, timesteps, lr, warmup_steps, total_steps, loss_weights=None, input_masking_fraction=0.0): # Added input_masking_fraction
        print("\nInitializing Trainer...")
        self.encoder = SpectralEncoder(in_dim, hid_dim, lat_dim, pe_dim=pe_dim).to(device)
        # pass hid_dim for ScoreNet's internal MLP, can be different from GNN hid_dim
        self.denoiser = ScoreNet(lat_dim, num_cell_types=num_cell_types, time_embed_dim=32, hid_dim_mlp=hid_dim).to(device)
        self.decoder = FeatureDecoder(lat_dim, hid_dim, in_dim).to(device) # Pass hid_dim for decoder's MLP
        self.diff = ScoreSDE(self.denoiser, T=1.0, N=timesteps).to(device)
        self.lap_pert = LaplacianPerturb()
        self.all_params = list(self.encoder.parameters()) + list(self.denoiser.parameters()) + list(self.decoder.parameters())
        self.optim = torch.optim.Adam(self.all_params, lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.current_step = 0
        self.input_masking_fraction = input_masking_fraction # store masking fraction

        if loss_weights is None: self.loss_weights = {'diff': 1.0, 'kl': 0.1, 'rec': 10.0}
        else: self.loss_weights = loss_weights
        print(f"Using loss weights: {self.loss_weights}")
        print(f"Input masking fraction: {self.input_masking_fraction}")


        num_warmup_steps_captured = warmup_steps
        num_training_steps_captured = total_steps
        def lr_lambda_fn(current_scheduler_step):
            if num_training_steps_captured == 0: return 1.0
            actual_warmup_steps = min(num_warmup_steps_captured, num_training_steps_captured)
            if current_scheduler_step < actual_warmup_steps:
                return float(current_scheduler_step + 1) / float(max(1, actual_warmup_steps))
            decay_phase_duration = num_training_steps_captured - actual_warmup_steps
            if decay_phase_duration <= 0: return 0.0
            current_step_in_decay = current_scheduler_step - actual_warmup_steps
            progress = float(current_step_in_decay) / float(max(1, decay_phase_duration))
            progress = min(1.0, progress)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda_fn)
        print("Trainer initialized.")

    def train_epoch(self, loader, current_epoch_num):
        """
        Trains the model for one epoch using batches from NeighborLoader.

        Args:
            loader (NeighborLoader): NeighborLoader providing subgraph batches.
            current_epoch_num (int): The current epoch number for logging.

        Returns:
            tuple: Average total loss, diffusion loss, KL loss, and reconstruction loss for the epoch.
        """
        self.encoder.train(); self.denoiser.train(); self.decoder.train()
        total_loss_val, total_loss_diff_val, total_loss_kl_val, total_loss_rec_val = 0.0, 0.0, 0.0, 0.0
        num_nodes_processed_in_epoch = 0
        num_batches_processed_in_epoch = 0
        self.optim.zero_grad(set_to_none=True)

        # iterate through subgraph batches provided by NeighborLoader
        for batch_data in loader:
            # batch_data is a Data object for the current subgraph batch
            batch_data = batch_data.to(device) 

            # neighborLoader batches contain different types of nodes:
            # batch_data.x[:batch_data.batch_size] are the seed nodes
            # the remaining nodes are neighbors sampled to compute representations for seed nodes
            # kosses are typically computed only for the seed nodes.

            num_seed_nodes_in_batch = batch_data.batch_size # number of seed nodes in this batch
            num_nodes_in_batch = batch_data.x.size(0) # total nodes in the subgraph batch (seeds + neighbors)

            if num_seed_nodes_in_batch == 0 or batch_data.x is None or batch_data.x.numel() == 0:
                 # print("Skipping empty batch (no seed nodes or features).")
                 continue

            # --- Input Gene Masking (applied to the full subgraph batch) ---
            # we apply masking to the features of all nodes in the subgraph batch
            original_x_batch = batch_data.x.clone() # Keep original for reconstruction target
            masked_x_batch = batch_data.x.clone()
            if self.encoder.training and self.input_masking_fraction > 0.0 and self.input_masking_fraction < 1.0:
                mask = torch.rand_like(masked_x_batch) < self.input_masking_fraction
                masked_x_batch[mask] = 0.0

            # get features and labels for the current subgraph batch
            lap_pe_batch = batch_data.lap_pe # laplacian PE for all nodes in the subgraph batch
            if lap_pe_batch is None or lap_pe_batch.size(0) != num_nodes_in_batch or lap_pe_batch.size(1) != self.encoder.pe_dim:
                # if PE is missing or wrong shape for the batch, create zero PEs for the batch
                lap_pe_batch = torch.zeros(num_nodes_in_batch, self.encoder.pe_dim, device=device, dtype=masked_x_batch.dtype)

            cell_type_labels_batch_full = batch_data.cell_type # cell types for all nodes in the subgraph batch
            if cell_type_labels_batch_full is None or cell_type_labels_batch_full.size(0) != num_nodes_in_batch:
                 cell_type_labels_batch_full = torch.zeros(num_nodes_in_batch, dtype=torch.long, device=device)

            # ensure cell type labels are within the valid range for the embedding layer
            if cell_type_labels_batch_full.max() >= self.denoiser.num_cell_types or cell_type_labels_batch_full.min() < 0:
                 cell_type_labels_batch_full = torch.clamp(cell_type_labels_batch_full, 0, self.denoiser.num_cell_types - 1)

            # --- Adversarial Perturbation (applied to edges within the subgraph batch) ---
            # apply perturbation to the edges of the current subgraph batch
            batch_edge_index = batch_data.edge_index
            batch_edge_weights = torch.ones(batch_edge_index.size(1), device=device, dtype=masked_x_batch.dtype) if batch_edge_index.numel() > 0 else None

            if batch_edge_weights is not None and batch_edge_weights.numel() > 0:
                 # use the masked features of the subgraph batch for adversarial calculation
                 initial_perturbed_weights = self.lap_pert.sample(batch_edge_index, num_nodes_in_batch)
                 adversarially_perturbed_weights = self.lap_pert.adversarial(
                     self.encoder, masked_x_batch, batch_edge_index, initial_perturbed_weights
                 )
                 if adversarially_perturbed_weights is not None:
                     adversarially_perturbed_weights = torch.nan_to_num(adversarially_perturbed_weights, nan=1.0, posinf=1.0, neginf=0.0)
                     adversarially_perturbed_weights = torch.clamp(adversarially_perturbed_weights, min=1e-4)
                 else:
                      adversarially_perturbed_weights = batch_edge_weights
            else:
                 adversarially_perturbed_weights = None

            # using autocast for mixed precision training
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # encoder gets masked features and the subgraph structure
                # it computes latent representations for ALL nodes in the subgraph batch (seeds + neighbors)
                mu_full_batch, logvar_full_batch = self.encoder(masked_x_batch, batch_edge_index, lap_pe_batch, adversarially_perturbed_weights) # [num_nodes_in_batch, lat_dim]

                if mu_full_batch.numel() == 0 or logvar_full_batch.numel() == 0:
                     print(f"Warning: Encoder output is empty for batch {num_batches_processed_in_epoch}. Skipping batch.")
                     continue

                # select outputs only for the SEED nodes (the first num_seed_nodes_in_batch nodes)
                mu = mu_full_batch[:num_seed_nodes_in_batch] # [num_seed_nodes_in_batch, lat_dim]
                logvar = logvar_full_batch[:num_seed_nodes_in_batch] # [num_seed_nodes_in_batch, lat_dim]
                cell_type_labels_batch = cell_type_labels_batch_full[:num_seed_nodes_in_batch] # [num_seed_nodes_in_batch]
                original_x_seed_batch = original_x_batch[:num_seed_nodes_in_batch] # [num_seed_nodes_in_batch, in_dim]

                # --- KL Divergence (computed for seed nodes) ---
                if torch.isnan(mu).any() or torch.isinf(mu).any() or \
                   torch.isnan(logvar).any() or torch.isinf(logvar).any():
                    print(f"Warning: NaN/Inf in encoder outputs (mu/logvar) for batch {num_batches_processed_in_epoch}. Skipping KL.")
                    kl_div = torch.tensor(0.0, device=device)
                else:
                    kl_div_terms = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
                    kl_div_terms_clamped = torch.relu(kl_div_terms)
                    if (kl_div_terms < -1e-5).any():
                        print(f"Warning: Negative KL term detected before clamping for batch {num_batches_processed_in_epoch}. Min term: {kl_div_terms.min().item()}")
                    kl_div = torch.sum(kl_div_terms_clamped, dim=-1).mean() # mean over seed nodes

                # --- Diffusion Loss (computed for seed nodes) ---
                # sample time steps per seed node
                t_indices = torch.randint(0, self.diff.N, (num_seed_nodes_in_batch,), device=device).long()
                time_values_for_loss = self.diff.timesteps[t_indices] # [num_seed_nodes_in_batch]

                sigma_t_batch = self.diff.marginal_std(time_values_for_loss) # [num_seed_nodes_in_batch]
                if sigma_t_batch.ndim == 1: sigma_t_batch = sigma_t_batch.unsqueeze(-1)

                noise_target = torch.randn_like(mu) # [num_seed_nodes_in_batch, lat_dim]
                # corrrupt mu (from the seed nodes)
                alpha_t = torch.exp(-time_values_for_loss).unsqueeze(-1)
                zt_corrupted = alpha_t * mu.detach() + sigma_t_batch * noise_target # [num_seed_nodes_in_batch, lat_dim]
                # denoiser predicts noise for the seed nodes, time, and cell type
                eps_predicted = self.denoiser(zt_corrupted, time_values_for_loss, cell_type_labels_batch) # [num_seed_nodes_in_batch, lat_dim]

                loss_diff = F.mse_loss(eps_predicted, noise_target)

                # --- Reconstruction Loss (computed for seed nodes) ---
                # decode the mean of the latent space (mu) for the seed nodes
                decoded_log_rates = self.decoder(mu) # [num_seed_nodes_in_batch, in_dim]
                # target is the original raw counts for the seed nodes
                target_counts_batch = original_x_seed_batch.float() 

                if decoded_log_rates.shape != target_counts_batch.shape:
                     print(f"Warning: Decoder output shape mismatch for batch {num_batches_processed_in_epoch}. Skipping reconstruction loss.")
                     loss_rec = torch.tensor(0.0, device=device)
                elif torch.isnan(decoded_log_rates).any() or torch.isinf(decoded_log_rates).any():
                     print(f"Warning: Decoder output contains NaN/Inf for batch {num_batches_processed_in_epoch}. Skipping reconstruction loss.")
                     loss_rec = torch.tensor(0.0, device=device)
                else:
                    loss_rec = F.poisson_nll_loss(decoded_log_rates, target_counts_batch, log_input=True, reduction='mean') # 'mean' averages over seed nodes and genes

                # apply weights and sum losses for the current batch
                # scale the batch loss by the number of SEED nodes in the batch for correct averaging
                batch_loss = (self.loss_weights.get('diff', 1.0) * loss_diff +
                              self.loss_weights.get('kl', 0.1) * kl_div +
                              self.loss_weights.get('rec', 1.0) * loss_rec)

                if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                    print(f"Warning: NaN/Inf batch loss detected at epoch {current_epoch_num}, batch {num_batches_processed_in_epoch}. Skipping backward pass for this batch.")
                    continue

                # Backward pass for the current batch loss
                # accumulate gradients across batches
                # scale the loss by the number of seed nodes to average gradients correctly
                self.scaler.scale(batch_loss / num_seed_nodes_in_batch).backward()

            # accumulate losses for reporting (weighted by number of seed nodes in batch)
            total_loss_val += batch_loss.item()
            total_loss_diff_val += loss_diff.item() * num_seed_nodes_in_batch # accumulate unscaled losses for correct average calculation later
            total_loss_kl_val += kl_div.item() * num_seed_nodes_in_batch
            total_loss_rec_val += loss_rec.item() * num_seed_nodes_in_batch
            num_nodes_processed_in_epoch += num_seed_nodes_in_batch # count seed nodes processed
            num_batches_processed_in_epoch += 1

            # optional: Print debug info
            # if num_batches_processed_in_epoch % 10 == 0 or num_batches_processed_in_epoch < 5 :
            #     lr_val = self.optim.param_groups[0]['lr']
            #     print(f"[DEBUG] Epoch {current_epoch_num} | Batch {num_batches_processed_in_epoch} | Seed Nodes: {num_seed_nodes_in_batch} | Total Nodes in Subgraph: {num_nodes_in_batch} | LR: {lr_val:.3e}")
            #     print(f"    Batch Losses -> Total: {batch_loss.item():.4f}, Diff: {loss_diff.item():.4f}, KL: {kl_div.item():.4f}, Rec: {loss_rec.item():.4f}")


        # --- End of Batch Loop ---

        # perform optimizer step and scheduler step AFTER processing all batches in the epoch
        if num_batches_processed_in_epoch > 0:
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.all_params, max_norm=1.0)
            self.scaler.step(self.optim)
            self.scaler.update()
            self.scheduler.step()
            # self.current_step += 1 # Current step is now handled by scheduler.step() implicitly or can be tracked separately if needed
            avg_total_loss = total_loss_val / num_seed_nodes_in_batch 
            if num_nodes_processed_in_epoch > 0:
                 avg_total_loss = total_loss_val / num_nodes_processed_in_epoch # total accumulated loss / total seed nodes
                 avg_diff_loss = total_loss_diff_val / num_nodes_processed_in_epoch
                 avg_kl_loss = total_loss_kl_val / num_nodes_processed_in_epoch
                 avg_rec_loss = total_loss_rec_val / num_nodes_processed_in_epoch
            else:
                 avg_total_loss, avg_diff_loss, avg_kl_loss, avg_rec_loss = 0.0, 0.0, 0.0, 0.0

            return avg_total_loss, avg_diff_loss, avg_kl_loss, avg_rec_loss 
        else:
             print(f"Warning: No batches processed in epoch {current_epoch_num}.")
             return 0.0, 0.0, 0.0, 0.0 # return 0 if no batches processed

    @torch.no_grad()
    def generate(self, num_samples, cell_type_condition=None):
        print(f"\nGenerating {num_samples} samples...")
        self.denoiser.eval(); self.decoder.eval() 
        if cell_type_condition is None:
            print("Generating unconditionally.")
            gen_cell_type_labels_tensor = torch.zeros(num_samples, dtype=torch.long, device=device)
        else:
            if isinstance(cell_type_condition, (list, np.ndarray)): gen_cell_type_labels_tensor = torch.tensor(cell_type_condition, dtype=torch.long, device=device)
            elif isinstance(cell_type_condition, torch.Tensor): gen_cell_type_labels_tensor = cell_type_condition.to(device).long()
            else: raise ValueError("cell_type_condition type error.")
            if gen_cell_type_labels_tensor.size(0) == 1 and num_samples > 1: gen_cell_type_labels_tensor = gen_cell_type_labels_tensor.repeat(num_samples)
            elif gen_cell_type_labels_tensor.size(0) != num_samples: raise ValueError(f"Cell type condition size mismatch.")
            if gen_cell_type_labels_tensor.max() >= self.denoiser.num_cell_types or gen_cell_type_labels_tensor.min() < 0:
                 print(f"Warning: Generated cell type label out of bounds. Clamping.")
                 gen_cell_type_labels_tensor = torch.clamp(gen_cell_type_labels_tensor, 0, self.denoiser.num_cell_types - 1)

        z_gen_shape = (num_samples, self.diff.score_model.lat_dim)
        z_generated = self.diff.sample(z_gen_shape, cell_type_labels=gen_cell_type_labels_tensor)
        decoded_log_rates = self.decoder(z_generated)
        try:
            rates = torch.exp(decoded_log_rates)
            rates = torch.clamp(rates, min=1e-6)
            rates = torch.nan_to_num(rates, nan=1e-6, posinf=1e6, neginf=1e-6)
            poisson_dist = torch.distributions.Poisson(rates)
            generated_counts_tensor = poisson_dist.sample().int().float()
        except Exception as e:
             print(f"Error during sampling from Poisson: {e}. Returning zero counts."); traceback.print_exc()
             generated_counts_tensor = torch.zeros_like(decoded_log_rates)
        generated_counts_np = generated_counts_tensor.cpu().numpy()
        generated_cell_types_np = gen_cell_type_labels_tensor.cpu().numpy()
        print("Generation complete.")
        return generated_counts_np, generated_cell_types_np

    @torch.no_grad()
    def evaluate_generation(self, real_adata, generated_counts, generated_cell_types, n_pcs=30, mmd_scales=[0.01, 0.1, 1, 10, 100]):
        print("\n--- Computing Evaluation Metrics ---")
        self.denoiser.eval(); self.decoder.eval() 
        if hasattr(self, 'encoder'): self.encoder.eval()

        if not hasattr(real_adata, 'X') or real_adata.X is None:
            print("Error: Real data AnnData missing .X."); return {"MMD": {}, "Wasserstein": {}, "Notes": "Missing real data counts."}
        real_counts = real_adata.X

        real_cell_types_present = False
        real_cell_type_labels = np.zeros(real_counts.shape[0], dtype=int) 
        real_cell_type_categories = ["Unknown"] 

        if hasattr(real_adata, 'obs') and 'cell_type' in real_adata.obs.columns:
            real_cell_types_present = True
            real_cell_type_series = real_adata.obs['cell_type']
            if not pd.api.types.is_categorical_dtype(real_cell_type_series):
                 try: real_cell_type_series = real_cell_type_series.astype('category')
                 except Exception: 
                     unique_types, real_cell_type_labels_temp = np.unique(real_cell_type_series.values, return_inverse=True)
                     real_cell_type_categories = unique_types.tolist()
                     real_cell_type_labels = real_cell_type_labels_temp 

            if pd.api.types.is_categorical_dtype(real_cell_type_series): 
                real_cell_type_labels = real_cell_type_series.cat.codes.values
                real_cell_type_categories = real_cell_type_series.cat.categories.tolist()

            print(f"Found {len(real_cell_type_categories) if real_cell_types_present else 'N/A'} cell types in real data.")
        else:
            print("Warning: 'cell_type' not found in real_adata.obs.")


        generated_counts_np = generated_counts.cpu().numpy() if isinstance(generated_counts, torch.Tensor) else generated_counts
        generated_cell_types_np = generated_cell_types.cpu().numpy() if isinstance(generated_cell_types, torch.Tensor) else generated_cell_types

        if real_counts.shape[1] != generated_counts_np.shape[1]:
             print(f"Error: Gene dimension mismatch. Real: {real_counts.shape[1]}, Gen: {generated_counts_np.shape[1]}"); return {"MMD": {}, "Wasserstein": {}, "Notes": "Gene dimension mismatch."}
        if real_counts.shape[0] == 0 or generated_counts_np.shape[0] == 0:
             print("Warning: Real or generated data empty."); return {"MMD": {}, "Wasserstein": {}, "Notes": "Real or generated data empty."}

        print("Applying normalization and log1p for evaluation.")
        def normalize_and_log(counts_arr):
            counts_dense = counts_arr.toarray() if sp.issparse(counts_arr) else counts_arr
            cell_totals = counts_dense.sum(axis=1, keepdims=True)
            cell_totals[cell_totals == 0] = 1.0
            normalized_counts = counts_dense / cell_totals * 1e4
            return np.log1p(normalized_counts)

        real_log1p = normalize_and_log(real_counts)
        generated_log1p = normalize_and_log(generated_counts_np)

        print(f"Performing PCA to {n_pcs} dimensions.")
        actual_n_pcs = min(n_pcs, real_log1p.shape[0] - 1 if real_log1p.shape[0] >1 else n_pcs, real_log1p.shape[1]) # handle single sample case for shape[0]-1
        if actual_n_pcs <= 0: print("Warning: PCA components <= 0."); return {"MMD": {}, "Wasserstein": {}, "Notes": "PCA components <= 0."}
        try:
            pca = PCA(n_components=actual_n_pcs, random_state=0)
            real_pca = pca.fit_transform(real_log1p)
            generated_pca = pca.transform(generated_log1p)
        except Exception as e: print(f"Error during PCA: {e}."); return {"MMD": {}, "Wasserstein": {}, "Notes": f"PCA failed: {e}"}
        if real_pca.shape[0] == 0 or generated_pca.shape[0] == 0: print("Error: PCA projected data empty."); return {"MMD": {}, "Wasserstein": {}, "Notes": "PCA projected data empty."}

        mmd_results, wasserstein_results = {}, {}
        def rbf_kernel_mmd(X, Y, scales_list):
            if X.shape[0] == 0 or Y.shape[0] == 0: return {scale: 0.0 for scale in scales_list}
            mmd_vals = {}
            try:
                dist_xx = cdist(X, X, 'sqeuclidean')
                dist_yy = cdist(Y, Y, 'sqeuclidean')
                dist_xy = cdist(X, Y, 'sqeuclidean')
                for scale in scales_list:
                    gamma = 1.0 / (2. * scale**2 + 1e-9)
                    K_xx, K_yy, K_xy = np.exp(-gamma * dist_xx), np.exp(-gamma * dist_yy), np.exp(-gamma * dist_xy)
                    mmd2 = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
                    mmd_vals[scale] = max(0, mmd2)
            except Exception as e_mmd: print(f"Error in MMD calc: {e_mmd}"); return {s: np.nan for s in scales_list}
            return mmd_vals

        if real_cell_types_present:
            unique_real_types = np.unique(real_cell_type_labels)
            unique_gen_types = np.unique(generated_cell_types_np)
            common_types = np.intersect1d(unique_real_types, unique_gen_types)
            if len(common_types) > 0:
                print(f"Computing conditional metrics for {len(common_types)} common cell types.")
                cond_mmd_agg = {scale: [] for scale in mmd_scales}
                cond_w_agg = []
                for ct_code in common_types:
                    ct_name = real_cell_type_categories[ct_code] if ct_code < len(real_cell_type_categories) else f"Code_{ct_code}"
                    real_pca_type = real_pca[real_cell_type_labels == ct_code]
                    gen_pca_type = generated_pca[generated_cell_types_np == ct_code]
                    if real_pca_type.shape[0] < 2 or gen_pca_type.shape[0] < 2: continue

                    mmd_type_s = rbf_kernel_mmd(real_pca_type, gen_pca_type, mmd_scales)
                    for s_val, val in mmd_type_s.items(): cond_mmd_agg[s_val].append(val)
                    try:
                        M = cdist(real_pca_type, gen_pca_type, 'sqeuclidean')
                        a = np.ones(real_pca_type.shape[0], dtype=np.float64) / real_pca_type.shape[0]
                        b = np.ones(gen_pca_type.shape[0], dtype=np.float64) / gen_pca_type.shape[0]
                        w2_type = ot.emd2(a, b, M)
                        cond_w_agg.append(np.sqrt(max(0, w2_type)))
                    except Exception as e_w: print(f"Error W-dist for type {ct_name}: {e_w}"); cond_w_agg.append(np.nan)

                for s_val_iter in mmd_scales: mmd_results[f'Conditional_Avg_Scale_{s_val_iter}'] = np.nanmean(cond_mmd_agg[s_val_iter]) if cond_mmd_agg[s_val_iter] else np.nan
                wasserstein_results['Conditional_Avg'] = np.nanmean(cond_w_agg) if cond_w_agg else np.nan
            else: print("No common cell types for conditional metrics.")
        else: print("Skipping conditional metrics: Real data missing cell types.")

        print("Computing unconditional metrics.")
        sample_size_uncond = 5000
        real_pca_s = real_pca[np.random.choice(real_pca.shape[0], min(sample_size_uncond, real_pca.shape[0]), replace=False)] if real_pca.shape[0] > 1 else real_pca
        gen_pca_s = generated_pca[np.random.choice(generated_pca.shape[0], min(sample_size_uncond, generated_pca.shape[0]), replace=False)] if generated_pca.shape[0] > 1 else generated_pca

        if real_pca_s.shape[0] >= 2 and gen_pca_s.shape[0] >= 2:
            mmd_uncond_s = rbf_kernel_mmd(real_pca_s, gen_pca_s, mmd_scales)
            for s_val_iter2, val2 in mmd_uncond_s.items(): mmd_results[f'Unconditional_Scale_{s_val_iter2}'] = val2
            try:
                M_uncond = cdist(real_pca_s, gen_pca_s, 'sqeuclidean')
                a_u = np.ones(real_pca_s.shape[0], dtype=np.float64) / real_pca_s.shape[0]
                b_u = np.ones(gen_pca_s.shape[0], dtype=np.float64) / gen_pca_s.shape[0]
                w2_uncond = ot.emd2(a_u, b_u, M_uncond)
                wasserstein_results['Unconditional'] = np.sqrt(max(0, w2_uncond))
            except Exception as e_w_u: print(f"Error W-dist uncond: {e_w_u}"); wasserstein_results['Unconditional'] = np.nan
        else: print("Not enough samples for unconditional metrics after sampling.")

        print("Evaluation metrics computation complete.")
        return {"MMD": mmd_results, "Wasserstein": wasserstein_results}


if __name__ == '__main__':
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-3
    EPOCHS = 2500
    HIDDEN_DIM = 512
    LATENT_DIM = 512
    PE_DIM = 20
    K_NEIGHBORS = 15
    PCA_NEIGHBORS = 50
    GENE_THRESHOLD = 20
    TIMESTEPS_DIFFUSION = 1000
    GLOBAL_SEED = 77
    set_seed(GLOBAL_SEED)

    loss_weights = {'diff': 10.0, 'kl': 0.05, 'rec': 10.0}
    INPUT_MASKING_FRACTION = 0.3

    TRAIN_H5AD = 'data/dentategyrus_train.h5ad' 
    TEST_H5AD = 'data/dentategyrus_test.h5ad' 
    DATA_ROOT = 'data/dentategyrus_processed' 
    os.makedirs(DATA_ROOT, exist_ok=True)
    # include masking fraction in train data root to avoid reusing processed data with different masking
    TRAIN_DATA_ROOT = os.path.join(DATA_ROOT, f'train_k{K_NEIGHBORS}_pe{PE_DIM}_gt{GENE_THRESHOLD}_pca{PCA_NEIGHBORS}_mask{INPUT_MASKING_FRACTION}')
    TEST_DATA_ROOT = os.path.join(DATA_ROOT, f'test_k{K_NEIGHBORS}_pe{PE_DIM}_gt{GENE_THRESHOLD}_pca{PCA_NEIGHBORS}')
    os.makedirs(os.path.join(TRAIN_DATA_ROOT, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_ROOT, 'processed'), exist_ok=True)

    train_dataset = None
    input_feature_dim, num_cell_types = 0, 1
    num_train_cells = 0 # initialize
    filtered_gene_names_from_train = []

    try:
        print(f"Loading/Processing training data from: {TRAIN_H5AD} into {TRAIN_DATA_ROOT}")
        train_dataset = PBMC3KDataset(h5ad_path=TRAIN_H5AD, k_neighbors=K_NEIGHBORS, pe_dim=PE_DIM, root=TRAIN_DATA_ROOT, train=True, gene_threshold=GENE_THRESHOLD, pca_neighbors=PCA_NEIGHBORS)
        if train_dataset and len(train_dataset) > 0 and train_dataset.get(0) and train_dataset.get(0).num_nodes > 0:
            num_train_cells = train_dataset.get(0).num_nodes
            input_feature_dim = train_dataset.get(0).x.size(1)
            num_cell_types = train_dataset.num_cell_types
            filtered_gene_names_from_train = train_dataset.filtered_gene_names
            print(f"Training data: {num_train_cells} cells, {input_feature_dim} genes, {num_cell_types} types. Filtered genes: {len(filtered_gene_names_from_train)}")
        else: raise ValueError("Training data is empty or invalid after processing.")
    except Exception as e:
        print(f"FATAL ERROR loading training data: {e}"); traceback.print_exc(); sys.exit(1)


    if num_train_cells > 0 and input_feature_dim > 0:
        # --- DataLoader with NeighborLoader has been used rather than DataLoader---
        # NeighborLoader samples a batch of nodes and their neighbors up to num_hops
        # num_neighbors=[-1] means include all neighbors at each hop
        # batch_size is the number of seed nodes per batch
        # num_hops should match the number of ChebConv layers (2 layers, K=4 means effectively 2 hops)
        # Note: ChebConv with K=4 uses information up to 4 hops away, but typically you'd set num_hops >= K/2
        # Let's set num_hops to 2 for a start, corresponding to two ChebConv layers.
        # If K=4 in ChebConv requires 4 hops of neighbors, set num_hops=4. Let's assume 2 hops for now.
        NUM_HOPS = 2 # Corresponds to the number of GNN layers
        train_loader = NeighborLoader(
             data=train_dataset.get(0), 
             num_neighbors=[-1] * NUM_HOPS, 
             batch_size=BATCH_SIZE, 
        )

        TOTAL_TRAINING_STEPS = len(train_loader) * EPOCHS
        WARMUP_STEPS = max(1, int(0.05 * TOTAL_TRAINING_STEPS))
        WARMUP_STEPS = min(WARMUP_STEPS, TOTAL_TRAINING_STEPS // 2)

        trainer = Trainer(in_dim=input_feature_dim, hid_dim=HIDDEN_DIM, lat_dim=LATENT_DIM, num_cell_types=num_cell_types,
                          pe_dim=PE_DIM, timesteps=TIMESTEPS_DIFFUSION, lr=LEARNING_RATE, warmup_steps=WARMUP_STEPS,
                          total_steps=TOTAL_TRAINING_STEPS, loss_weights=loss_weights, input_masking_fraction=INPUT_MASKING_FRACTION)

        if TOTAL_TRAINING_STEPS > 0:
             # trainer.scheduler.step()
             print("Scheduler will be stepped after the first batch.")

        if TOTAL_TRAINING_STEPS > 0:
            print(f"\nStarting training for {EPOCHS} epochs. Total steps: {TOTAL_TRAINING_STEPS}, Warmup: {WARMUP_STEPS}. Initial LR: {LEARNING_RATE:.2e}")
            for epoch in range(1, EPOCHS + 1):
                avg_total_loss, avg_diff_loss, avg_kl_loss, avg_rec_loss = trainer.train_epoch(train_loader, epoch)
                current_lr = trainer.optim.param_groups[0]["lr"]
                print(f"Epoch {epoch:03d}/{EPOCHS} Summary -> AvgTotal: {avg_total_loss:.4f}, AvgDiff: {avg_diff_loss:.4f}, AvgKL: {avg_kl_loss:.4f}, AvgRec: {avg_rec_loss:.4f}, LR: {current_lr:.3e}")
            print("\nTraining completed.")
        else: print("\nSkipping training: No training steps.")
    else: print("\nSkipping training: Training data empty/no features.")


    print("\n--- Starting Final Evaluation on Test Set ---")
    test_adata = None
    num_genes_eval = 0
    if not filtered_gene_names_from_train:
        print("ERROR: Filtered gene names from training not available. Skipping evaluation.")
    else:
        try:
            print(f"Loading test data from: {TEST_H5AD}")
            test_adata_raw = sc.read_h5ad(TEST_H5AD)
            if not hasattr(test_adata_raw, 'var_names') or test_adata_raw.var_names is None:
                 print("FATAL ERROR: Raw test adata missing .var_names. Skipping evaluation."); test_adata = None
            else:
                 genes_to_keep_in_test_mask = test_adata_raw.var_names.isin(filtered_gene_names_from_train)
                 if np.sum(genes_to_keep_in_test_mask) == 0:
                      print("FATAL ERROR: No genes in raw test data match filtered training genes. Skipping evaluation."); test_adata = None
                 else:
                      test_adata = test_adata_raw[:, genes_to_keep_in_test_mask].copy()
                      num_genes_eval = test_adata.shape[1]
                      print(f"Test data after consistent gene filtering: {test_adata.shape[0]} cells, {num_genes_eval} genes.")
                      if 'cell_type' not in test_adata.obs.columns: print("Warning: 'cell_type' missing in test_adata.")
        except FileNotFoundError: print(f"FATAL ERROR: Test H5AD not found: {TEST_H5AD}. Skipping evaluation."); test_adata = None
        except Exception as e: print(f"FATAL ERROR loading/filtering test AnnData: {e}. Skipping evaluation."); traceback.print_exc(); test_adata = None

    if test_adata is not None and test_adata.shape[0] > 0 and test_adata.shape[1] > 0 and filtered_gene_names_from_train:
        if 'trainer' in locals() and trainer is not None:
            if trainer.decoder.decoder_mlp[-1].out_features != num_genes_eval:
                 print(f"FATAL ERROR: Decoder output dim ({trainer.decoder.decoder_mlp[-1].out_features}) != test gene dim ({num_genes_eval}). Skipping eval.")
            else:
                num_test_cells = test_adata.shape[0]
                print(f"\nGenerating 3 datasets of size {num_test_cells} for evaluation.")
                generated_datasets_counts, generated_datasets_cell_types = [], []
                cell_type_condition_for_gen = None
                if 'cell_type' in test_adata.obs.columns and not test_adata.obs['cell_type'].empty:
                    if pd.api.types.is_categorical_dtype(test_adata.obs['cell_type']):
                        cell_type_condition_for_gen = test_adata.obs['cell_type'].cat.codes.values
                    else: # If not categorical, try to get codes
                        try:
                            _, cell_type_condition_for_gen = np.unique(test_adata.obs['cell_type'].values, return_inverse=True)
                        except Exception as e_ct_conv:
                            print(f"Warning: Could not convert test cell_types to codes for generation: {e_ct_conv}")
                            cell_type_condition_for_gen = None # Fallback to unconditional

                    if cell_type_condition_for_gen is not None: print("Generating conditionally based on real test set cell types.")

                if cell_type_condition_for_gen is None: print("Generating unconditionally for evaluation (or test cell types were problematic).")

                for i in range(3):
                    print(f"Generating dataset {i+1}/3...")
                    try:
                        gen_counts, gen_types = trainer.generate(num_samples=num_test_cells, cell_type_condition=cell_type_condition_for_gen)
                        generated_datasets_counts.append(gen_counts); generated_datasets_cell_types.append(gen_types)
                    except Exception as e_gen: print(f"Error generating dataset {i+1}: {e_gen}"); generated_datasets_counts.append(None); generated_datasets_cell_types.append(None)

                all_mmd_results_per_scale = {scale: [] for scale in [0.01, 0.1, 1, 10, 100]}
                all_wasserstein_results = []
                for i in range(len(generated_datasets_counts)):
                     if generated_datasets_counts[i] is not None and generated_datasets_cell_types[i] is not None:
                         print(f"\nEvaluating generated dataset {i+1}/{len(generated_datasets_counts)}...")
                         try:
                             metrics = trainer.evaluate_generation(real_adata=test_adata, generated_counts=generated_datasets_counts[i], generated_cell_types=generated_datasets_cell_types[i])
                             print(f"Metrics for dataset {i+1}: {metrics}")
                             if metrics and "MMD" in metrics and metrics["MMD"]:
                                  for scale_key, mmd_val in metrics["MMD"].items():
                                      try:
                                          # extract scale value from keys like 'Conditional_Avg_Scale_0.1' or 'Unconditional_Scale_1.0'
                                          parts = scale_key.split('_')
                                          if parts[-2] == 'Scale':
                                              scale_val_str = parts[-1]
                                              scale_val_float = float(scale_val_str)
                                              if scale_val_float in all_mmd_results_per_scale and not np.isnan(mmd_val):
                                                  all_mmd_results_per_scale[scale_val_float].append(mmd_val)
                                      except (ValueError, IndexError):
                                          print(f"Warning: Could not parse scale from MMD key: {scale_key}")
                                          pass
                             if metrics and "Wasserstein" in metrics and metrics["Wasserstein"]:
                                 for w_key, w_val in metrics["Wasserstein"].items():
                                     if not np.isnan(w_val): all_wasserstein_results.append(w_val)
                         except Exception as e_eval_loop: print(f"Error evaluating dataset {i+1}: {e_eval_loop}")

                print("\n--- Averaged Evaluation Metrics over Generated Datasets ---")
                averaged_metrics = {"MMD_Averages": {}, "Wasserstein_Average": np.nan}
                for scale_iter, results_list_iter in all_mmd_results_per_scale.items():
                    if results_list_iter: averaged_metrics["MMD_Averages"][f'Avg_Scale_{scale_iter}'] = np.mean(results_list_iter)
                    else: averaged_metrics["MMD_Averages"][f'Avg_Scale_{scale_iter}'] = np.nan
                if all_wasserstein_results: averaged_metrics["Wasserstein_Average"] = np.mean(all_wasserstein_results)
                print(averaged_metrics)
        else: print("\nSkipping final evaluation: Trainer not initialized.")
    else: print("\nSkipping final evaluation: Test data problematic or filtered genes unavailable.")
    print("\nScript execution finished.")
