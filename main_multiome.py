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
from scipy.spatial.distance import cdist 
import sys 
import torch.distributions as dist
from pbmc3k.seed import *
from pbmc3k.utils import * 
from pbmc3k.dataset import*
from pbmc3k.models import *
from pbmc3k.viz import *
import gc # For garbage collection

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
    import ot 
except ImportError as e:
    print(f"Missing required library: {e}.")
    pass

def sliced_wasserstein_gpu(X, Y, num_projections=50, seed=42):
    """
    Computes Sliced Wasserstein Distance (SWD) between two high-dimensional distributions X and Y on GPU.
    X, Y: (N, D) tensors
    """
    device = X.device
    dim = X.shape[1]
    
    torch.manual_seed(seed)
    
    projections = torch.randn(dim, num_projections, device=device)
    projections = F.normalize(projections, p=2, dim=0) # Normalize columns
    
    X_proj = X @ projections # (N, num_projections)
    Y_proj = Y @ projections
    
    X_sorted, _ = torch.sort(X_proj, dim=0)
    Y_sorted, _ = torch.sort(Y_proj, dim=0)
    
    wdists = torch.abs(X_sorted - Y_sorted).mean(dim=0) 
    return wdists.mean().item()

class Trainer:
    def __init__(self, in_dim, hid_dim, lat_dim, num_cell_types, pe_dim, timesteps, lr, warmup_steps, total_steps, loss_weights=None, input_masking_fraction=0.0):
        print("\nInitializing Trainer...")
        self.encoder = SpectralEncoder(in_dim, hid_dim, lat_dim, pe_dim=pe_dim).to(device)
        self.denoiser = ScoreNet(lat_dim, num_cell_types=num_cell_types, time_embed_dim=32, hid_dim_mlp=hid_dim).to(device)
        self.decoder = FeatureDecoder(lat_dim, hid_dim, in_dim).to(device)
        self.diff = ScoreSDE(self.denoiser, T=1.0, N=timesteps).to(device)
        self.lap_pert = LaplacianPerturb()
        self.all_params = list(self.encoder.parameters()) + list(self.denoiser.parameters()) + list(self.decoder.parameters())
        self.optim = torch.optim.Adam(self.all_params, lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.current_step = 0
        self.input_masking_fraction = input_masking_fraction

        if loss_weights is None: self.loss_weights = {'diff': 1.0, 'kl': 0.01, 'rec': 1.0}
        else: self.loss_weights = loss_weights # Should be passed from args
        print(f"Using loss weights: {self.loss_weights}")

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
        self.encoder.train(); self.denoiser.train(); self.decoder.train()
        total_loss_val, total_loss_diff_val, total_loss_kl_val, total_loss_rec_val = 0.0, 0.0, 0.0, 0.0
        num_batches_processed = 0

        for data in loader:
            data = data.to(device)
            num_nodes_in_batch = data.x.size(0)
            if num_nodes_in_batch == 0: continue

            original_x = data.x
            masked_x = data.x.clone()
            if self.encoder.training and self.input_masking_fraction > 0.0:
                mask = torch.rand_like(masked_x) < self.input_masking_fraction
                masked_x[mask] = 0.0

            lap_pe = data.lap_pe
            if lap_pe is None or lap_pe.size(0) != num_nodes_in_batch:
                lap_pe = torch.zeros(num_nodes_in_batch, self.encoder.pe_dim, device=device, dtype=masked_x.dtype)

            cell_type_labels = data.cell_type
            if cell_type_labels is None: cell_type_labels = torch.zeros(num_nodes_in_batch, dtype=torch.long, device=device)
            cell_type_labels = torch.clamp(cell_type_labels, 0, self.denoiser.num_cell_types - 1)

            edge_weights = torch.ones(data.edge_index.size(1), device=device) if data.edge_index.numel() > 0 else None
            adversarially_perturbed_weights = None
            if edge_weights is not None:
                initial_perturbed_weights = self.lap_pert.sample(data.edge_index, num_nodes_in_batch)
                adversarially_perturbed_weights = self.lap_pert.adversarial(self.encoder, masked_x, data.edge_index, initial_perturbed_weights)
                adversarially_perturbed_weights = torch.clamp(torch.nan_to_num(adversarially_perturbed_weights, nan=1.0), min=1e-4)

            self.optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                mu, logvar = self.encoder(masked_x, data.edge_index, lap_pe, adversarially_perturbed_weights)
                if mu.size(0) != num_nodes_in_batch: continue

                if torch.isnan(mu).any() or torch.isnan(logvar).any():
                    kl_div = torch.tensor(0.0, device=device)
                else:
                    kl_div_terms = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
                    kl_div = torch.sum(torch.relu(kl_div_terms), dim=-1).mean()

                std = torch.exp(0.5 * logvar)
                t_indices = torch.randint(0, self.diff.N, (num_nodes_in_batch,), device=device).long()
                time_values = self.diff.timesteps[t_indices]
                sigma_t = self.diff.marginal_std(time_values).unsqueeze(-1)
                alpha_t = torch.exp(-time_values).unsqueeze(-1)
                
                noise_target = torch.randn_like(mu)
                zt = alpha_t * mu.detach() + sigma_t * noise_target
                eps_pred = self.denoiser(zt, time_values, cell_type_labels)
                loss_diff = F.mse_loss(eps_pred, noise_target)

                # NB Loss
                log_mu_rec, log_theta_rec = self.decoder(mu)
                mu_rec = torch.exp(log_mu_rec)
                theta_rec = torch.clamp(torch.exp(log_theta_rec), min=1e-6, max=1e6)
                
                probs_rec = torch.clamp(mu_rec / (mu_rec + theta_rec), min=1e-6, max=1-1e-6)
                nb_dist = dist.NegativeBinomial(total_count=theta_rec, probs=probs_rec)
                loss_rec = -nb_dist.log_prob(original_x.long()).mean()

                final_loss = (self.loss_weights['diff'] * loss_diff + 
                              self.loss_weights['kl'] * kl_div + 
                              self.loss_weights['rec'] * loss_rec)

            if torch.isnan(final_loss): continue

            self.scaler.scale(final_loss).backward()
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
            self.scaler.step(self.optim)
            self.scaler.update()
            self.scheduler.step()
            self.current_step += 1

            if self.current_step % 10 == 0:
                 print(f"Epoch {current_epoch_num} | Step {self.current_step} | LR: {self.optim.param_groups[0]['lr']:.3e} | Total: {final_loss.item():.4f} Diff: {loss_diff.item():.4f} KL: {kl_div.item():.4f} Rec: {loss_rec.item():.4f}")

            total_loss_val += final_loss.item()
            total_loss_diff_val += loss_diff.item()
            total_loss_kl_val += kl_div.item()
            total_loss_rec_val += loss_rec.item()
            num_batches_processed += 1

        if num_batches_processed > 0:
            return total_loss_val/num_batches_processed, total_loss_diff_val/num_batches_processed, total_loss_kl_val/num_batches_processed, total_loss_rec_val/num_batches_processed
        return 0.0, 0.0, 0.0, 0.0

    def state_dict(self):
        return {
            'encoder': self.encoder.state_dict(),
            'denoiser': self.denoiser.state_dict(),
            'decoder': self.decoder.state_dict(),
            'current_step': self.current_step
        }

    @torch.no_grad()
    def generate(self, num_samples, cell_type_condition=None):
        self.denoiser.eval(); self.decoder.eval()
        if cell_type_condition is None:
             gen_labels = torch.zeros(num_samples, dtype=torch.long, device=device)
        else:
             gen_labels = torch.tensor(cell_type_condition, dtype=torch.long, device=device)
             gen_labels = torch.clamp(gen_labels, 0, self.denoiser.num_cell_types - 1)
        
        z_shape = (num_samples, self.diff.score_model.lat_dim)
        z_gen = self.diff.sample(z_shape, cell_type_labels=gen_labels)
        
        log_mu, log_theta = self.decoder(z_gen)
        mu = torch.exp(log_mu)
        theta = torch.clamp(torch.exp(log_theta), min=1e-6, max=1e6)
        
        probs = torch.clamp(mu / (mu + theta), min=1e-6, max=1-1e-6)
        nb = dist.NegativeBinomial(total_count=theta, probs=probs)
        # return TENSORS to keep on GPU
        counts = torch.nan_to_num(nb.sample().float(), nan=0.0)
        return counts, gen_labels

    @torch.no_grad()
    def evaluate_generation_gpu(self, real_adata_X_tensor, gen_counts_tensor, gen_cell_types_tensor, real_cell_types_tensor=None, n_pcs=30, mmd_scales=[0.01, 0.1, 1, 10, 100]):
        """
        Evaluate Generation completely on GPU to save System RAM.
        real_adata_X_tensor: sparse or dense tensor on GPU or CPU. Prefer GPU if fits.
        """
        print("\\n--- Computing Evaluation Metrics (GPU) ---")
        self.denoiser.eval(); self.decoder.eval()
        
        # Ensure inputs are tensors on device
        if not isinstance(gen_counts_tensor, torch.Tensor): gen_counts_tensor = torch.tensor(gen_counts_tensor, device=device)
        else: gen_counts_tensor = gen_counts_tensor.to(device)
        
        # Real Data handling
        if isinstance(real_adata_X_tensor, (sp.spmatrix, np.ndarray)):
            if sp.issparse(real_adata_X_tensor): real_adata_X_tensor = real_adata_X_tensor.toarray()
            real_counts = torch.tensor(real_adata_X_tensor, device=device).float()
        else: 
            real_counts = real_adata_X_tensor.to(device).float()
            
        if real_counts.size(1) != gen_counts_tensor.size(1):
             print(f"Error: Dimension mismatch. Real {real_counts.size(1)}, Gen {gen_counts_tensor.size(1)}")
             return {}

        # Log1p Normalization
        def normalize_and_log_gpu(counts):
             # counts: (N, G)
             sums = counts.sum(dim=1, keepdim=True)
             sums[sums==0] = 1.0
             norm = counts / sums * 1e4
             return torch.log1p(norm)
             
        real_log1p = normalize_and_log_gpu(real_counts)
        gen_log1p = normalize_and_log_gpu(gen_counts_tensor)
        
        # PCA on GPU
        actual_n_pcs = min(n_pcs, real_log1p.size(0), real_log1p.size(1))
        
        try:
             real_mean = real_log1p.mean(dim=0)
             real_centered = real_log1p - real_mean
             
             # SVD based PCA
             U, S, V = torch.pca_lowrank(real_centered, q=actual_n_pcs, center=False, niter=2) 
             
             real_pca = torch.matmul(real_centered, V)
             gen_pca = torch.matmul(gen_log1p - real_mean, V) 
             
        except Exception as e:
             print(f"GPU PCA Error: {e}"); return {}

        mmd_results, wasserstein_results = {}, {}
        
        def rbf_mmd_gpu(X, Y, scales):
             # X, Y: (N, PCs)
             xx = torch.cdist(X, X).pow(2)
             yy = torch.cdist(Y, Y).pow(2)
             xy = torch.cdist(X, Y).pow(2)
             
             res = {}
             for s in scales:
                  gamma = 1.0 / (2 * s**2 + 1e-9)
                  k_xx = torch.exp(-gamma * xx).mean()
                  k_yy = torch.exp(-gamma * yy).mean()
                  k_xy = torch.exp(-gamma * xy).mean()
                  mmd = k_xx + k_yy - 2*k_xy
                  res[s] = max(0.0, mmd.item())
             return res
        
        # Unconditional
        n_uncond = min(5000, real_pca.size(0), gen_pca.size(0))
        idx_r = torch.randperm(real_pca.size(0))[:n_uncond]
        idx_g = torch.randperm(gen_pca.size(0))[:n_uncond]
        
        real_pca_s = real_pca[idx_r]
        gen_pca_s = gen_pca[idx_g]
        
        mmd_uncond = rbf_mmd_gpu(real_pca_s, gen_pca_s, mmd_scales)
        for s, v in mmd_uncond.items(): mmd_results[f"Unconditional_Scale_{s}"] = v
        
        # SWD GPU
        swd_val = sliced_wasserstein_gpu(real_pca_s, gen_pca_s)
        wasserstein_results["Unconditional"] = swd_val
        
        # Conditional
        if real_cell_types_tensor is not None and gen_cell_types_tensor is not None:
             real_labels = real_cell_types_tensor.to(device).long()
             gen_labels = gen_cell_types_tensor.to(device).long()
             
             unique_real = torch.unique(real_labels)
             unique_gen = torch.unique(gen_labels)
             
             cond_mmd = {s: [] for s in mmd_scales}
             cond_w = []
             
             for ct in unique_real:
                  if ct in unique_gen:
                       real_sub = real_pca[real_labels == ct]
                       gen_sub = gen_pca[gen_labels == ct]
                       if real_sub.size(0) < 5 or gen_sub.size(0) < 5: continue
                       
                       mmd_res = rbf_mmd_gpu(real_sub, gen_sub, mmd_scales)
                       for s, v in mmd_res.items(): cond_mmd[s].append(v)
                       
                       swd_sub = sliced_wasserstein_gpu(real_sub, gen_sub)
                       cond_w.append(swd_sub)
                       
             for s in mmd_scales:
                  if cond_mmd[s]: mmd_results[f"Conditional_Avg_Scale_{s}"] = np.mean(cond_mmd[s])
                  else: mmd_results[f"Conditional_Avg_Scale_{s}"] = float('nan')
                  
             if cond_w: wasserstein_results["Conditional_Avg"] = np.mean(cond_w)
             else: wasserstein_results["Conditional_Avg"] = float('nan')
             
        return {"MMD": mmd_results, "Wasserstein": wasserstein_results}

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
    parser.add_argument('--loss_weight_diff', type=float, default=0.5)
    parser.add_argument('--loss_weight_kl', type=float, default=0.005)
    parser.add_argument('--loss_weight_rec', type=float, default=0.5)
    parser.add_argument('--input_masking_fraction', type=float, default=0.1)
    parser.add_argument('--knn', type=int, default=7)
    parser.add_argument('--pe_dim', type=int, default=50)
    parser.add_argument('--pca_dim', type=int, default=50)
    parser.add_argument('--gene_threshold', type=int, default=20)
    parser.add_argument('--timesteps_diffusion', type=int, default=1000)
    parser.add_argument('--viz', type=bool, default=True)
    parser.add_argument('--num_gnn_layers', type=int, default=3)
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

    # PATHS FOR MULTIOME
    TRAIN_H5AD = 'data/lymph_node_lymphoma_14k_raw_feature_bc_matrix.h5' 
    DATA_ROOT = 'data/multiome_processed'
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
        trainer = Trainer(in_dim=input_feature_dim, hid_dim=HIDDEN_DIM, lat_dim=LATENT_DIM, num_cell_types=num_cell_types,
                        pe_dim=PE_DIM, timesteps=TIMESTEPS_DIFFUSION, lr=LEARNING_RATE, warmup_steps=WARMUP_STEPS,
                        total_steps=TOTAL_TRAINING_STEPS, loss_weights=loss_weights, input_masking_fraction=INPUT_MASKING_FRACTION)

        print(f"\nStarting training for {EPOCHS} epochs...")
        for epoch in range(1, EPOCHS + 1):
            avg_total_loss, avg_diff_loss, avg_kl_loss, avg_rec_loss = trainer.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch:03d}/{EPOCHS} Summary -> AvgTotal: {avg_total_loss:.4f}, AvgDiff: {avg_diff_loss:.4f}, AvgKL: {avg_kl_loss:.4f}, AvgRec: {avg_rec_loss:.4f}")
            if epoch % 5 == 0 or epoch == EPOCHS:
                torch.save(trainer.state_dict(), os.path.join(TRAIN_DATA_ROOT, f'trainer_checkpoint_epoch_{epoch}.pt'))
        
        torch.save(trainer.state_dict(), os.path.join(TRAIN_DATA_ROOT, 'trainer_final_state.pt'))
        print("\nTraining completed.")
    else: print("\nSkipping training: Training data empty.")

    print("\n--- Starting Final Evaluation on Test Set ---")
    test_adata = None
    if not filtered_gene_names_from_train:
        print("ERROR: Filtered gene names from training not available. Skipping evaluation.")
    else:
        try:
            test_dataset = PBMC3KDataset(h5ad_path=TRAIN_H5AD, k_neighbors=K_NEIGHBORS, pe_dim=PE_DIM, root=TEST_DATA_ROOT, train=False, gene_threshold=GENE_THRESHOLD, pca_neighbors=PCA_NEIGHBORS, seed=GLOBAL_SEED)
            if test_dataset and len(test_dataset) > 0:
                 data_test = test_dataset.get(0)
                 X_test_tensor = data_test.x
                 cell_types_test = data_test.cell_type
                 
                 test_adata = sc.AnnData(X=X_test_tensor.cpu().numpy())
                 test_adata.obs['cell_type'] = cell_types_test.cpu().numpy()
                 test_adata.var_names = filtered_gene_names_from_train 
                 
                 print(f"Test data loaded: {test_adata.shape[0]} cells.")
            else: print("Test dataset empty.")

        except Exception as e: print(f"FATAL ERROR loading test data: {e}"); traceback.print_exc(); test_adata = None

    if test_adata is not None:
         num_test_cells = test_adata.shape[0]
         cell_type_condition_for_gen = test_adata.obs['cell_type'].values if 'cell_type' in test_adata.obs else None
         
         all_mmd = {s: [] for s in [0.01, 0.1, 1, 10, 100]}
         all_w = []
         
         for i in range(3):
             print(f"\n--- Processing Dataset {i+1}/3 (GPU Mode) ---")
             try:
                 gen_batch_size = 128
                 num_batches = int(np.ceil(num_test_cells / gen_batch_size))
                 gen_counts_list, gen_types_list = [], []
                 
                 print(f"Generating {num_test_cells} cells...")
                 for b in range(num_batches):
                     start = b * gen_batch_size
                     end = min((b + 1) * gen_batch_size, num_test_cells)
                     cond = cell_type_condition_for_gen[start:end] if cell_type_condition_for_gen is not None else None
                     
                     bc, bt = trainer.generate(num_samples=end-start, cell_type_condition=cond)
                     gen_counts_list.append(bc) # These are TENSORS now
                     gen_types_list.append(bt)
                 
                 gen_counts = torch.cat(gen_counts_list, dim=0)
                 gen_types = torch.cat(gen_types_list, dim=0)
                 del gen_counts_list, gen_types_list
                 
                 # Evaluate on GPU
                 real_adata_X_tensor = torch.tensor(test_adata.X).float() 
                 real_labels_tensor = torch.tensor(test_adata.obs['cell_type'].values).long()
                 
                 metrics = trainer.evaluate_generation_gpu(real_adata_X_tensor, gen_counts, gen_types, real_labels_tensor)
                 print(f"Metrics (GPU): {metrics}")
                 
                 if "MMD" in metrics:
                      for k, v in metrics["MMD"].items():
                          s = float(k.split('_')[-1])
                          if not np.isnan(v): all_mmd[s].append(v)
                 if "Wasserstein" in metrics:
                      for k, v in metrics["Wasserstein"].items():
                          if not np.isnan(v): all_w.append(v)
                          
                 del gen_counts, gen_types
                 torch.cuda.empty_cache()
                 
             except Exception as e: print(f"Error {e}"); traceback.print_exc()
             
    print("\nScript execution finished.")
