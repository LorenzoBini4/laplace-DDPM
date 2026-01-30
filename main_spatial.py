import torch
from torch_scatter import scatter_add, scatter_mean
import torch.nn.functional as F
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
import torch.distributions as dist # Import for Negative Binomial distribution
from pbmc3k.seed import *
from pbmc3k.utils import * 
from pbmc3k.dataset import*
from pbmc3k.models import *
from pbmc3k.viz import *

# Check for required libraries
try:
    import torch
    import torch_geometric
    import torch_scatter
    import scanpy
    import anndata
    import pandas
    import numpy
    import sklearn
    import scipy
    import ot # Python Optimal Transport library
    import leidenalg # For fallback clustering
except ImportError as e:
    print(f"Missing required library: {e}. Please install it using pip.")
    # print("Example: pip install torch torch-geometric torch-scatter scanpy anndata pandas numpy scikit-learn scipy POT leidenalg")
    pass 

class Trainer:
    def __init__(self, in_dim, hid_dim, lat_dim, num_cell_types, pe_dim, timesteps, lr, warmup_steps, total_steps, loss_weights=None, input_masking_fraction=0.0): # Added input_masking_fraction
        print("\nInitializing Trainer...")
        self.encoder = SpectralEncoder(in_dim, hid_dim, lat_dim, pe_dim=pe_dim).to(device)
        # Pass hid_dim for ScoreNet's internal MLP, can be different from GNN hid_dim
        self.denoiser = ScoreNet(lat_dim, num_cell_types=num_cell_types, time_embed_dim=32, hid_dim_mlp=hid_dim).to(device)
        self.decoder = FeatureDecoder(lat_dim, hid_dim, in_dim).to(device) # Pass hid_dim for decoder's MLP
        self.diff = ScoreSDE(self.denoiser, T=1.0, N=timesteps).to(device)
        self.lap_pert = LaplacianPerturb()
        self.all_params = list(self.encoder.parameters()) + list(self.denoiser.parameters()) + list(self.decoder.parameters())
        self.optim = torch.optim.Adam(self.all_params, lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.current_step = 0
        self.input_masking_fraction = input_masking_fraction # Store masking fraction

        if loss_weights is None: self.loss_weights = {'diff': 1.0, 'kl': 0.01, 'rec': 1.0} # Adjusted KL and Rec weights for NB
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

    def train_epoch(self, loader, current_epoch_num): # Added current_epoch_num for logging
        self.encoder.train(); self.denoiser.train(); self.decoder.train()
        total_loss_val, total_loss_diff_val, total_loss_kl_val, total_loss_rec_val = 0.0, 0.0, 0.0, 0.0
        num_batches_processed = 0

        for data in loader:
            data = data.to(device)
            num_nodes_in_batch = data.x.size(0)
            if num_nodes_in_batch == 0 or data.x is None or data.x.numel() == 0: continue

            # --- Input Gene Masking ---
            original_x = data.x # Keep original for reconstruction target
            masked_x = data.x.clone()
            if self.encoder.training and self.input_masking_fraction > 0.0 and self.input_masking_fraction < 1.0:
                # Create a mask for each cell independently
                mask = torch.rand_like(masked_x) < self.input_masking_fraction
                masked_x[mask] = 0.0 # Set masked genes to zero
                print(f"[DEBUG] Applied input masking. Fraction: {self.input_masking_fraction}, {(mask.sum() / masked_x.numel()):.3f} actual masked elements.")

            lap_pe = data.lap_pe
            if lap_pe is None or lap_pe.size(0) != num_nodes_in_batch or lap_pe.size(1) != self.encoder.pe_dim:
                lap_pe = torch.zeros(num_nodes_in_batch, self.encoder.pe_dim, device=device, dtype=masked_x.dtype) # Use masked_x.dtype

            cell_type_labels = data.cell_type
            if cell_type_labels is None or cell_type_labels.size(0) != num_nodes_in_batch:
                cell_type_labels = torch.zeros(num_nodes_in_batch, dtype=torch.long, device=device)
            if cell_type_labels.max() >= self.denoiser.num_cell_types or cell_type_labels.min() < 0:
                cell_type_labels = torch.clamp(cell_type_labels, 0, self.denoiser.num_cell_types - 1)

            edge_weights = torch.ones(data.edge_index.size(1), device=device, dtype=masked_x.dtype) if data.edge_index.numel() > 0 else None
            if edge_weights is not None and edge_weights.numel() > 0:
                initial_perturbed_weights = self.lap_pert.sample(data.edge_index, num_nodes_in_batch)
                adversarially_perturbed_weights = self.lap_pert.adversarial(self.encoder, masked_x, data.edge_index, initial_perturbed_weights) # Use masked_x for adversarial
                adversarially_perturbed_weights = torch.nan_to_num(adversarially_perturbed_weights, nan=1.0, posinf=1.0, neginf=0.0)
                adversarially_perturbed_weights = torch.clamp(adversarially_perturbed_weights, min=1e-4)
            else: adversarially_perturbed_weights = None

            self.optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Encoder gets masked_x
                mu, logvar = self.encoder(masked_x, data.edge_index, lap_pe, adversarially_perturbed_weights)
                if mu.numel() == 0 or logvar.numel() == 0: continue
                if mu.size(0) != num_nodes_in_batch or logvar.size(0) != num_nodes_in_batch: continue

                # --- KL Divergence ---
                # Check for NaN/Inf in mu and logvar before KL calculation
                if torch.isnan(mu).any() or torch.isinf(mu).any() or \
                torch.isnan(logvar).any() or torch.isinf(logvar).any():
                    print(f"Warning: NaN/Inf detected in encoder outputs (mu/logvar) at epoch {current_epoch_num}, step {self.current_step}. Skipping KL for this batch.")
                    kl_div = torch.tensor(0.0, device=device) # Or a high penalty, or skip batch
                else:
                    # Correct KL formula: 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
                    # This should always be non-negative.
                    kl_div_terms = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
                    # Ensure terms are non-negative (they should be theoretically)
                    kl_div_terms_clamped = torch.relu(kl_div_terms) # Clamp at zero if any small negatives occur due to precision
                    if (kl_div_terms < -1e-5).any(): # Check if any term was significantly negative before clamping
                        print(f"Warning: Negative KL term detected before clamping at epoch {current_epoch_num}, step {self.current_step}. Min term: {kl_div_terms.min().item()}")

                    kl_div = torch.sum(kl_div_terms_clamped, dim=-1).mean() # Sum over lat_dim, then mean over batch

                std = torch.exp(0.5 * logvar)
                t_indices = torch.randint(0, self.diff.N, (num_nodes_in_batch,), device=device).long()
                time_values_for_loss = self.diff.timesteps[t_indices]
                sigma_t_batch = self.diff.marginal_std(time_values_for_loss)
                if sigma_t_batch.ndim == 1: sigma_t_batch = sigma_t_batch.unsqueeze(-1)
                noise_target = torch.randn_like(mu)
                alpha_t = torch.exp(-time_values_for_loss).unsqueeze(-1)
                zt_corrupted = alpha_t * mu.detach() + sigma_t_batch * noise_target
                eps_predicted = self.denoiser(zt_corrupted, time_values_for_loss, cell_type_labels)
                loss_diff = F.mse_loss(eps_predicted, noise_target)

                # --- Negative Binomial Reconstruction Loss (use original_x as target) ---
                log_mu_rec, log_theta_rec = self.decoder(mu) # Decoder now returns two outputs
                
                # Ensure mu and theta are positive and stable
                mu_rec = torch.exp(log_mu_rec)
                theta_rec = torch.exp(log_theta_rec)
                
                # Clamp theta to avoid numerical issues (e.g., very small or very large values)
                theta_rec = torch.clamp(theta_rec, min=1e-6, max=1e6) # Added clamping for theta

                target_counts = original_x.long() # Changed from .float() to .long()
                
                if mu_rec.shape != target_counts.shape or theta_rec.shape != target_counts.shape:
                    print(f"Warning: Decoder output shape mismatch for NB loss. Skipping reconstruction loss.")
                    loss_rec = torch.tensor(0.0, device=device)
                elif torch.isnan(mu_rec).any() or torch.isinf(mu_rec).any() or \
                    torch.isnan(theta_rec).any() or torch.isinf(theta_rec).any():
                    print("Warning: Decoder output (mu/theta) contains NaN/Inf for NB loss. Skipping reconstruction loss.")
                    loss_rec = torch.tensor(0.0, device=device)
                else:
                    # Negative Binomial distribution parameterized by total_count (dispersion) and probs
                    # total_count is often denoted as 'r' or 'k'
                    # probs = mean / (mean + total_count)
                    probs_rec = mu_rec / (mu_rec + theta_rec)
                    probs_rec = torch.clamp(probs_rec, min=1e-6, max=1 - 1e-6)  # Clamp to (0, 1)
                    nb_dist = dist.NegativeBinomial(total_count=theta_rec, probs=probs_rec)
                    loss_rec = -nb_dist.log_prob(target_counts).mean() # Negative log-likelihood

                final_loss = (self.loss_weights.get('diff', 1.0) * loss_diff +
                            self.loss_weights.get('kl', 0.1) * kl_div + # Optional KL annealing here
                            self.loss_weights.get('rec', 1.0) * loss_rec)

            if torch.isnan(final_loss) or torch.isinf(final_loss):
                print(f"Warning: NaN/Inf loss detected at epoch {current_epoch_num}, step {self.current_step}. Skipping batch.")
                continue

            self.scaler.scale(final_loss).backward()
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.all_params, max_norm=1.0)
            self.scaler.step(self.optim)
            self.scaler.update()
            self.scheduler.step()
            self.current_step += 1

            # Debug prints for loss values and learning rate
            if self.current_step % 10 == 0 or num_batches_processed < 5 : 
                lr_val = self.optim.param_groups[0]['lr']
                print(f"Epoch {current_epoch_num} | Batch Step {self.current_step} (Overall) | Optim Steps (approx): {self.optim._step_count} | Scheduler Steps: {self.scheduler._step_count} | LR: {lr_val:.3e}")
                print(f"    Losses -> Total: {final_loss.item():.4f}, Diff: {loss_diff.item():.4f}, KL: {kl_div.item():.4f}, Rec: {loss_rec.item():.4f}")


            total_loss_val += final_loss.item()
            total_loss_diff_val += loss_diff.item()
            total_loss_kl_val += kl_div.item()
            total_loss_rec_val += loss_rec.item()
            num_batches_processed +=1

        if num_batches_processed > 0:
            avg_total_loss = total_loss_val / num_batches_processed
            avg_diff_loss = total_loss_diff_val / num_batches_processed
            avg_kl_loss = total_loss_kl_val / num_batches_processed
            avg_rec_loss = total_loss_rec_val / num_batches_processed
            return avg_total_loss, avg_diff_loss, avg_kl_loss, avg_rec_loss # Return all for detailed logging
        else:
            print(f"Warning: No batches processed in epoch {current_epoch_num}.")
            return 0.0, 0.0, 0.0, 0.0

    def state_dict(self):
        return {
            'encoder': self.encoder.state_dict(),
            'denoiser': self.denoiser.state_dict(),
            'decoder': self.decoder.state_dict(),
            'diff': self.diff.state_dict() if hasattr(self.diff, 'state_dict') else {},
            'current_step': self.current_step,
            'input_masking_fraction': self.input_masking_fraction,
            'loss_weights': self.loss_weights,
        }
    
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
        # NOTE: sample_guided internally uses torch.enable_grad() if needed for guidance
        z_generated = self.diff.sample(z_gen_shape, cell_type_labels=gen_cell_type_labels_tensor)
        
        # decoder now returns log_mu and log_theta
        log_mu_gen, log_theta_gen = self.decoder(z_generated)
        
        # convert to mu and theta
        mu_gen = torch.exp(log_mu_gen)
        theta_gen = torch.exp(log_theta_gen)
        
        # clamp theta for stability during sampling
        theta_gen = torch.clamp(theta_gen, min=1e-6, max=1e6)

        try:
            # sample from Negative Binomial
            nb_dist = dist.NegativeBinomial(total_count=theta_gen, probs=mu_gen / (mu_gen + theta_gen))
            generated_counts_tensor = nb_dist.sample().int().float()
            generated_counts_tensor = torch.nan_to_num(generated_counts_tensor, nan=0.0, posinf=0.0, neginf=0.0) # handle potential NaNs from sampling
        except Exception as e:
            print(f"Error during sampling from Negative Binomial: {e}. Returning zero counts."); traceback.print_exc()
            generated_counts_tensor = torch.zeros_like(mu_gen) # use mu_gen shape for consistency

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
        if hasattr(real_adata, 'obs') and 'cell_type' in real_adata.obs.columns:
            real_cell_types_present = True
            real_cell_type_series = real_adata.obs['cell_type']
            if not pd.api.types.is_categorical_dtype(real_cell_type_series):
                try: real_cell_type_series = real_cell_type_series.astype('category')
                except Exception: unique_types, real_cell_type_labels = np.unique(real_cell_type_series.values, return_inverse=True); real_cell_type_categories = unique_types.tolist()
            if pd.api.types.is_categorical_dtype(real_cell_type_series): # Check again after potential conversion
                real_cell_type_labels = real_cell_type_series.cat.codes.values
                real_cell_type_categories = real_cell_type_series.cat.categories.tolist()
            print(f"Found {len(real_cell_type_categories) if real_cell_types_present and 'real_cell_type_categories' in locals() else 'N/A'} cell types in real data.")
        else:
            print("Warning: 'cell_type' not found in real_adata.obs.")
            real_cell_type_labels = np.zeros(real_counts.shape[0], dtype=int)
            real_cell_type_categories = ["Unknown"]

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
        actual_n_pcs = min(n_pcs, real_log1p.shape[0] - 1, real_log1p.shape[1])
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
                    gamma = 1.0 / (2. * scale**2 + 1e-9) # add epsilon for scale=0 case
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
                    for s, val in mmd_type_s.items(): cond_mmd_agg[s].append(val)
                    try:
                        M = cdist(real_pca_type, gen_pca_type, 'sqeuclidean')
                        a = np.ones(real_pca_type.shape[0]) / real_pca_type.shape[0]
                        b = np.ones(gen_pca_type.shape[0]) / gen_pca_type.shape[0]
                        w2_type = ot.emd2(a, b, M)
                        cond_w_agg.append(np.sqrt(max(0, w2_type))) # ensure non-negative before sqrt
                    except Exception as e_w: print(f"Error W-dist for type {ct_name}: {e_w}"); cond_w_agg.append(np.nan)
                
                for s in mmd_scales: mmd_results[f'Conditional_Avg_Scale_{s}'] = np.nanmean(cond_mmd_agg[s]) if cond_mmd_agg[s] else np.nan
                wasserstein_results['Conditional_Avg'] = np.nanmean(cond_w_agg) if cond_w_agg else np.nan
            else: print("No common cell types for conditional metrics.")
        else: print("Skipping conditional metrics: Real data missing cell types.")

        print("Computing unconditional metrics.")
        sample_size_uncond = 5000
        real_pca_s = real_pca[np.random.choice(real_pca.shape[0], min(sample_size_uncond, real_pca.shape[0]), replace=False)] if real_pca.shape[0] > 1 else real_pca
        gen_pca_s = generated_pca[np.random.choice(generated_pca.shape[0], min(sample_size_uncond, generated_pca.shape[0]), replace=False)] if generated_pca.shape[0] > 1 else generated_pca

        if real_pca_s.shape[0] >= 2 and gen_pca_s.shape[0] >= 2:
            mmd_uncond_s = rbf_kernel_mmd(real_pca_s, gen_pca_s, mmd_scales)
            for s, val in mmd_uncond_s.items(): mmd_results[f'Unconditional_Scale_{s}'] = val
            try:
                M_uncond = cdist(real_pca_s, gen_pca_s, 'sqeuclidean')
                a_u = np.ones(real_pca_s.shape[0]) / real_pca_s.shape[0]
                b_u = np.ones(gen_pca_s.shape[0]) / gen_pca_s.shape[0]
                w2_uncond = ot.emd2(a_u, b_u, M_uncond)
                wasserstein_results['Unconditional'] = np.sqrt(max(0, w2_uncond)) # Ensure non-negative
            except Exception as e_w_u: print(f"Error W-dist uncond: {e_w_u}"); wasserstein_results['Unconditional'] = np.nan
        else: print("Not enough samples for unconditional metrics after sampling.")

        print("Evaluation metrics computation complete.")
        return {"MMD": mmd_results, "Wasserstein": wasserstein_results}

    def evaluate_tstr(self, real_adata, generated_counts, generated_cell_types):
        print("\n--- Computing TSTR (Train on Synthetic, Test on Real) ---")
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.preprocessing import StandardScaler
            
            # Prepare Synthetic Data
            if generated_cell_types is None:
                 print("TSTR skipped: No generated cell types."); return {}
            
            # Use PCA features for speed and robustness
            def process_for_ml(counts_np):
                 # Log1p
                 sums = counts_np.sum(axis=1, keepdims=True)
                 sums[sums==0] = 1
                 norm = counts_np / sums * 1e4
                 return np.log1p(norm)
            
            X_syn = process_for_ml(generated_counts)
            y_syn = generated_cell_types.astype(int)
            
            # Prepare Real Data
            # Note: real_adata.obs should have 'cell_type' now if fallback worked
            if not hasattr(real_adata, 'X') or 'cell_type' not in real_adata.obs.columns:
                 print("TSTR skipped: Real data missing X or cell_type."); return {}
            
            X_real = real_adata.X
            if sp.issparse(X_real): X_real = X_real.toarray()
            X_real = process_for_ml(X_real)
            
            y_real = real_adata.obs['cell_type'].astype('category').cat.codes.values
            
            clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            clf.fit(X_syn, y_syn)
            
            y_pred_real = clf.predict(X_real)
            acc = accuracy_score(y_real, y_pred_real)
            f1 = f1_score(y_real, y_pred_real, average='weighted')
            
            print(f"TSTR Results -> Accuracy: {acc:.4f}, F1 (weighted): {f1:.4f}")
            return {"TSTR_Accuracy": acc, "TSTR_F1": f1}
            
        except Exception as e:
            print(f"Error computing TSTR: {e}")
            return {}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=77)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--loss_weight_diff', type=float, default=0.005)
    parser.add_argument('--loss_weight_kl', type=float, default=0.5)
    parser.add_argument('--loss_weight_rec', type=float, default=0.005)
    parser.add_argument('--input_masking_fraction', type=float, default=0.1)
    parser.add_argument('--knn', type=int, default=30)
    parser.add_argument('--pe_dim', type=int, default=50)
    parser.add_argument('--pca_dim', type=int, default=50)
    parser.add_argument('--gene_threshold', type=int, default=20)
    parser.add_argument('--timesteps_diffusion', type=int, default=1000)
    parser.add_argument('--viz', type=bool, default=True)
    parser.add_argument('--num_gnn_layers', type=int, default=3) # Unused in simple init but kept for args compatibility
    parser.add_argument('--num_mlp_layers', type=int, default=3)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    HIDDEN_DIM = args.hidden_dim
    LATENT_DIM = args.latent_dim
    PE_DIM = args.pe_dim
    K_NEIGHBORS = args.knn
    PCA_NEIGHBORS = args.pca_dim
    GENE_THRESHOLD = args.gene_threshold
    TIMESTEPS_DIFFUSION = args.timesteps_diffusion
    GLOBAL_SEED = args.seed
    VIZ = args.viz
    INPUT_MASKING_FRACTION = args.input_masking_fraction
    
    set_seed(GLOBAL_SEED)

    loss_weights = {'diff': args.loss_weight_diff, 'kl': args.loss_weight_kl, 'rec': args.loss_weight_rec}

    # PATHS FOR SPATIAL
    TRAIN_H5AD = 'data/visium_lymph_node.h5ad'
    TEST_H5AD = 'data/visium_lymph_node.h5ad' 
    DATA_ROOT = 'data/spatial_processed'
    os.makedirs(DATA_ROOT, exist_ok=True)
    TRAIN_DATA_ROOT = os.path.join(DATA_ROOT, f'train_k{K_NEIGHBORS}')
    TEST_DATA_ROOT = os.path.join(DATA_ROOT, f'test_k{K_NEIGHBORS}')
    os.makedirs(os.path.join(TRAIN_DATA_ROOT, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_ROOT, 'processed'), exist_ok=True)

    train_dataset = None
    input_feature_dim, num_cell_types = 0, 1
    filtered_gene_names_from_train = []

    try:
        print(f"Loading/Processing training data from: {TRAIN_H5AD}")
        train_dataset = PBMC3KDataset(h5ad_path=TRAIN_H5AD, k_neighbors=K_NEIGHBORS, pe_dim=PE_DIM, root=TRAIN_DATA_ROOT, train=True, gene_threshold=GENE_THRESHOLD, pca_neighbors=PCA_NEIGHBORS, seed=GLOBAL_SEED)
        
        if train_dataset and len(train_dataset) > 0:
            num_train_cells = train_dataset.get(0).num_nodes
            input_feature_dim = train_dataset.get(0).x.size(1)
            num_cell_types = train_dataset.num_cell_types
            filtered_gene_names_from_train = train_dataset.filtered_gene_names
            print(f"Training data: {num_train_cells} cells, {input_feature_dim} genes, {num_cell_types} types.")
        else: raise ValueError("Training data is empty.")
    except Exception as e:
        print(f"FATAL ERROR loading training data: {e}"); traceback.print_exc(); sys.exit(1)


    if num_train_cells > 0 and input_feature_dim > 0:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
        
        TOTAL_TRAINING_STEPS = len(train_loader) * EPOCHS
        WARMUP_STEPS = max(1, int(0.05 * TOTAL_TRAINING_STEPS))
        WARMUP_STEPS = min(WARMUP_STEPS, TOTAL_TRAINING_STEPS // 2)

        trainer = Trainer(in_dim=input_feature_dim, hid_dim=HIDDEN_DIM, lat_dim=LATENT_DIM, num_cell_types=num_cell_types,
                        pe_dim=PE_DIM, timesteps=TIMESTEPS_DIFFUSION, lr=LEARNING_RATE, warmup_steps=WARMUP_STEPS,
                        total_steps=TOTAL_TRAINING_STEPS, loss_weights=loss_weights, input_masking_fraction=INPUT_MASKING_FRACTION)

        print(f"\nStarting training for {EPOCHS} epochs...")
        for epoch in range(1, EPOCHS + 1):
            avg_total_loss, avg_diff_loss, avg_kl_loss, avg_rec_loss = trainer.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch:03d}/{EPOCHS} Summary -> AvgTotal: {avg_total_loss:.4f}, AvgDiff: {avg_diff_loss:.4f}, AvgKL: {avg_kl_loss:.4f}, AvgRec: {avg_rec_loss:.4f}, LR: {trainer.optim.param_groups[0]['lr']:.3e}")
            if epoch % 5 == 0 or epoch == EPOCHS:
                checkpoint_path = os.path.join(TRAIN_DATA_ROOT, f'trainer_checkpoint_epoch_{epoch}.pt')
                torch.save(trainer.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at: {checkpoint_path}")
        
        final_checkpoint_path = os.path.join(TRAIN_DATA_ROOT, 'trainer_final_state.pt')
        torch.save(trainer.state_dict(), final_checkpoint_path)
        print(f"Final trainer state saved at: {final_checkpoint_path}")
        print("\nTraining completed.")
    else: print("\nSkipping training: Training data empty.")

    print("\n--- Starting Final Evaluation on Test Set ---")
    test_adata = None
    num_genes_eval = 0
    if not filtered_gene_names_from_train:
        print("ERROR: Filtered gene names from training not available. Skipping evaluation.")
    else:
        try:
            print(f"Loading test data from: {TEST_H5AD}")
            test_adata_raw = sc.read_h5ad(TEST_H5AD)
            print(f"DEBUG: Loaded Test Data Shape: {test_adata_raw.shape}")
            
            # --- Auto-Annotation / Clustering ---
            possible_keys = ['cell_type', 'celltype', 'CellType', 'cluster', 'Cluster', 'clusters', 'leiden', 'annotation', 'subclass']
            found_key = None
            for key in possible_keys:
                if key in test_adata_raw.obs.columns:
                    found_key = key
                    break
            
            if found_key:
                print(f"DEBUG: Found cell type column: '{found_key}'. Renaming to 'cell_type'.")
                test_adata_raw.obs['cell_type'] = test_adata_raw.obs[found_key]
            else:
                print("Warning: 'cell_type' (or synonyms) NOT found in test columns. Performing basic clustering (Leiden) to generate labels for conditional evaluation.")
                try:
                    print("Running PCA/Neighbors/Leiden on Test Data...")
                    if test_adata_raw.is_view: test_adata_raw = test_adata_raw.copy()
                    
                    adata_for_clustering = test_adata_raw.copy()
                    sc.pp.normalize_total(adata_for_clustering, target_sum=1e4)
                    sc.pp.log1p(adata_for_clustering)
                    sc.pp.pca(adata_for_clustering, n_comps=50)
                    sc.pp.neighbors(adata_for_clustering, n_neighbors=10)
                    sc.tl.leiden(adata_for_clustering, key_added='cell_type', resolution=0.5)
                    
                    test_adata_raw.obs['cell_type'] = adata_for_clustering.obs['cell_type'].values
                    print(f"Generated {len(test_adata_raw.obs['cell_type'].unique())} Leiden clusters as proxy cell types.")
                    del adata_for_clustering
                except Exception as e_cluster:
                    print(f"Error during fallback clustering: {e_cluster}. Conditional generation will be skipped.")

            # --- Robust Alignment ---
            print("Aligning test data to training feature space (Robust Strategy)...")
            target_genes = filtered_gene_names_from_train
            n_target = len(target_genes)
            n_cells = test_adata_raw.shape[0]
            
            test_gene_to_idx = {gene: i for i, gene in enumerate(test_adata_raw.var_names)}
            src_indices = []
            dst_indices = []
            found_genes_count = 0
            
            for i, gene in enumerate(target_genes):
                if gene in test_gene_to_idx:
                    src_indices.append(test_gene_to_idx[gene])
                    dst_indices.append(i)
                    found_genes_count += 1
            
            print(f"DEBUG: Found {found_genes_count} / {n_target} training genes in test set.")
            
            if found_genes_count == 0:
                print("FATAL ERROR: Zero overlap between training and test genes."); test_adata = None
            else:
                X_test_source = test_adata_raw.X
                if not sp.issparse(X_test_source): X_test_source = sp.csr_matrix(X_test_source)
                X_test_source = X_test_source.tocsc()
                X_valid = X_test_source[:, src_indices]
                X_new = sp.lil_matrix((n_cells, n_target), dtype=np.float32)
                X_new[:, dst_indices] = X_valid
                
                test_adata = sc.AnnData(X=X_new.tocsr())
                test_adata.obs = test_adata_raw.obs.copy() 
                test_adata.var_names = target_genes
                test_adata.var_names_make_unique() 
                
                if 'spatial' in test_adata_raw.obsm:
                    test_adata.obsm['spatial'] = test_adata_raw.obsm['spatial']
                
                num_genes_eval = test_adata.shape[1]
                print(f"Test data aligned: {test_adata.shape[0]} cells, {num_genes_eval} genes.")
                if 'cell_type' not in test_adata.obs.columns: print("Warning: 'cell_type' missing in test_adata.")

        except Exception as e: print(f"FATAL ERROR loading/filtering test AnnData: {e}. Skipping evaluation."); traceback.print_exc(); test_adata = None

    if test_adata is not None and test_adata.shape[0] > 0:
        print(f"\nGenerating 3 datasets of size {test_adata.shape[0]} for evaluation.")
        print(f"DEBUG: test_adata.obs columns: {test_adata.obs.columns.tolist()}")
        if 'cell_type' in test_adata.obs.columns:
             print(f"DEBUG: cell_type unique values: {test_adata.obs['cell_type'].unique()}")

        cell_type_condition_for_gen = None
        if 'cell_type' in test_adata.obs.columns:
            if pd.api.types.is_categorical_dtype(test_adata.obs['cell_type']):
                cell_type_condition_for_gen = test_adata.obs['cell_type'].cat.codes.values
            else:
                _, cell_type_condition_for_gen = np.unique(test_adata.obs['cell_type'].values, return_inverse=True)
            print("Generating conditionally based on real test set cell types.")
        else:
            print("Generating unconditionally for evaluation.")

        generated_datasets_counts = []
        generated_datasets_cell_types = []

        for i in range(3): 
            print(f"Generating dataset {i+1}/3...")
            try:
                gen_counts, gen_types = trainer.generate(num_samples=test_adata.shape[0], cell_type_condition=cell_type_condition_for_gen)
                generated_datasets_counts.append(gen_counts); generated_datasets_cell_types.append(gen_types)
                
                # Evaluation
                print(f"Evaluating generated dataset {i+1}/3...")
                metrics = trainer.evaluate_generation(real_adata=test_adata, generated_counts=gen_counts, generated_cell_types=gen_types)
                print(f"Metrics for dataset {i+1}: {metrics}")
                
                # TSTR
                tstr = trainer.evaluate_tstr(real_adata=test_adata, generated_counts=gen_counts, generated_cell_types=gen_types)
                if tstr: print(f"TSTR: {tstr}")
                
            except Exception as e_gen: print(f"Error generating/evaluating dataset {i+1}: {e_gen}")

        # Qualitative Plots (last dataset)
        if generated_datasets_counts and VIZ:
             output_plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qualitative_evaluation_plots_v2")
             current_train_cell_type_categories = ["Unknown"]
             if train_dataset and hasattr(train_dataset, 'cell_type_categories'): current_train_cell_type_categories = train_dataset.cell_type_categories
             
             generate_qualitative_plots(
                 real_adata_filtered=test_adata,
                 generated_counts=generated_datasets_counts[-1],
                 generated_cell_types=generated_datasets_cell_types[-1],
                 train_cell_type_categories=current_train_cell_type_categories,
                 train_filtered_gene_names=filtered_gene_names_from_train,
                 output_dir=output_plot_dir,
                 umap_neighbors=K_NEIGHBORS,
                 model_name="LapDDPM"
             )
    
    print("\nScript execution finished.")
