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
import json
import hashlib
from datetime import datetime
import traceback
import glob
import scanpy as sc
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist 
import sys 
import torch.distributions as dist
from laplace.seed import *
from laplace.utils import * 
from laplace.dataset import*
from laplace.models import *
from laplace.viz import *
import gc # For garbage collection
from tqdm import tqdm

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

def rbf_mmd_gpu(X, Y, scales=[0.1, 1.0, 10.0]):
    """
    Computes RBF MMD between two sets of samples X and Y on GPU to avoid OOM.
    X, Y: (N, D) tensors
    """
    if X.size(0) > 5000:
        idx_x = torch.randperm(X.size(0))[:5000]
        X = X[idx_x]
    if Y.size(0) > 5000:
        idx_y = torch.randperm(Y.size(0))[:5000]
        Y = Y[idx_y]
        
    device = X.device
    min_size = min(X.size(0), Y.size(0))
    X = X[:min_size]
    Y = Y[:min_size]
    
    # Efficient Distance Matrix computation: |x-y|^2 = x^2 + y^2 - 2xy
    xx = torch.mm(X, X.t())
    yy = torch.mm(Y, Y.t())
    xy = torch.mm(X, Y.t())
    
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    
    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*xy
    
    results = {}
    for scale in scales:
        gamma = 1.0 / (2 * scale)
        K_xx = torch.exp(-gamma * dxx)
        K_yy = torch.exp(-gamma * dyy)
        K_xy = torch.exp(-gamma * dxy)
        
        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        results[scale] = mmd.item()
        
    return results

def _experiment_plot_dir(args, label):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qualitative_evaluation_plots_v2")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = dict(vars(args))
    cfg["label"] = label
    encoder_tag = str(cfg.get("encoder_type", "encoder")).lower()
    cfg_json = json.dumps(cfg, sort_keys=True, default=str)
    digest = hashlib.sha1(cfg_json.encode("utf-8")).hexdigest()[:10]
    out_dir = os.path.join(base_dir, f"{label}_{encoder_tag}_{ts}_{digest}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True, default=str)
    return out_dir

class Trainer:
    def __init__(self, in_dim, hid_dim, lat_dim, num_cell_types, pe_dim, timesteps, max_time, lr, warmup_steps, total_steps,
                 loss_weights=None, input_masking_fraction=0.0, context_dim=0,
                 dual_graph=True, spectral_loss_weight=0.1, mask_rec_weight=1.0,
                 kl_anneal_steps=1000, kl_free_bits=0.1, encoder_type="gnn",
                 batch_adv_weight=0.0, num_batches=1, context_align_weight=0.0,
                 mask_strategy="random", mask_gene_probs=None,
                 gene_stats_weight=0.0, zero_rate_weight=0.0, cell_zero_rate_weight=0.0,
                 disable_kl=False,
                 cell_type_loss_weight=0.0, dispersion_weight=0.0,
                 pi_temperature=1.0, pi_bias=0.0, pi_blend=0.8, diffusion_temperature=1.0,
                 libsize_weight=0.1,
                 gen_calibrate_means=False, gen_calibrate_zero=False,
                 gen_match_zero_per_cell=False, gen_zero_per_cell_strength=1.0,
                 gen_mean_scale_clip_low=0.25, gen_mean_scale_clip_high=4.0,
                 diffusion_method="sde", simple_mode=False,
                 gene_stats_subsample=0, gene_stats_anneal_steps=1000, zero_rate_anneal_steps=1000,
                 transformer_layers=2, transformer_heads=8, transformer_dropout=0.1,
                 bio_positional=False, bio_pos_dim=32):
        print("\nInitializing Trainer...")
        self.encoder_type = encoder_type
        if encoder_type == "mlp":
            self.encoder = MLPEncoder(in_dim, hid_dim, lat_dim, pe_dim=pe_dim).to(device)
            self.dual_graph = False
        elif encoder_type == "transformer":
            self.encoder = GlobalTransformerEncoder(
                in_dim, hid_dim, lat_dim, pe_dim=pe_dim,
                num_layers=transformer_layers, num_heads=transformer_heads, dropout=transformer_dropout
            ).to(device)
            self.dual_graph = False
        else:
            self.encoder = SpectralEncoder(in_dim, hid_dim, lat_dim, pe_dim=pe_dim, dual_graph=dual_graph).to(device)
            self.dual_graph = dual_graph
        self.denoiser = ScoreNet(lat_dim, num_cell_types=num_cell_types, time_embed_dim=32, hid_dim_mlp=hid_dim, context_dim=context_dim).to(device)
        self.decoder = GeneDecoder(lat_dim, hid_dim, in_dim, zinb=True, use_libsize=True).to(device)
        self.diff = ScoreSDE(self.denoiser, T=max_time, N=timesteps).to(device)
        self.lap_pert = LaplacianPerturb()
        self.batch_adv_weight = batch_adv_weight
        self.context_align_weight = context_align_weight
        self.mask_strategy = mask_strategy
        self.mask_gene_probs = mask_gene_probs
        self.gene_stats_weight = gene_stats_weight
        self.zero_rate_weight = zero_rate_weight
        self.cell_zero_rate_weight = cell_zero_rate_weight
        self.gene_stats_subsample = gene_stats_subsample
        self.gene_stats_anneal_steps = max(1, int(gene_stats_anneal_steps))
        self.zero_rate_anneal_steps = max(1, int(zero_rate_anneal_steps))
        self.disable_kl = disable_kl
        self.cell_type_loss_weight = cell_type_loss_weight
        self.dispersion_weight = dispersion_weight
        self.pi_temperature = pi_temperature
        self.pi_bias = pi_bias
        self.pi_blend = pi_blend
        self.diffusion_temperature = diffusion_temperature
        self.libsize_weight = libsize_weight
        self.gen_calibrate_means = gen_calibrate_means
        self.gen_calibrate_zero = gen_calibrate_zero
        self.gen_match_zero_per_cell = gen_match_zero_per_cell
        self.gen_zero_per_cell_strength = gen_zero_per_cell_strength
        self.gen_mean_scale_clip_low = gen_mean_scale_clip_low
        self.gen_mean_scale_clip_high = gen_mean_scale_clip_high
        self.diffusion_method = diffusion_method
        self.simple_mode = simple_mode
        self.bio_positional = bio_positional
        self.bio_pos_dim = bio_pos_dim
        self.batch_discriminator = None
        self.ctx_proj = None
        self.cell_classifier = None

        if self.simple_mode:
            self.input_masking_fraction = 0.0
            self.mask_rec_weight = 0.0
            self.spectral_loss_weight = 0.0
            self.batch_adv_weight = 0.0
            self.context_align_weight = 0.0
            self.dispersion_weight = 0.0
        if num_batches > 1 and batch_adv_weight > 0.0:
            self.batch_discriminator = BatchDiscriminator(lat_dim, num_batches).to(device)
        if context_dim > 0 and context_align_weight > 0.0:
            self.ctx_proj = nn.Linear(context_dim, lat_dim).to(device)
        self.all_params = list(self.encoder.parameters()) + list(self.denoiser.parameters()) + list(self.decoder.parameters())
        if self.batch_discriminator is not None:
            self.all_params += list(self.batch_discriminator.parameters())
        if self.ctx_proj is not None:
            self.all_params += list(self.ctx_proj.parameters())
        if num_cell_types > 1 and self.cell_type_loss_weight > 0.0:
            self.cell_classifier = nn.Linear(lat_dim, num_cell_types).to(device)
            self.all_params += list(self.cell_classifier.parameters())
        self.optim = torch.optim.Adam(self.all_params, lr=lr)
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        self.current_step = 0
        self.input_masking_fraction = input_masking_fraction
        self.context_dim = context_dim
        self.spectral_loss_weight = spectral_loss_weight
        self.mask_rec_weight = mask_rec_weight
        self.kl_anneal_steps = max(1, int(kl_anneal_steps))
        self.kl_free_bits = kl_free_bits

        if loss_weights is None: self.loss_weights = {'diff': 1.0, 'kl': 0.01, 'rec': 1.0}
        else: self.loss_weights = loss_weights # Should be passed from args
        self.base_loss_weights = dict(self.loss_weights)
        print(f"Using loss weights: {self.loss_weights}")
        print(f"Input masking fraction: {self.input_masking_fraction}")
        print(f"Multimodal Context Dim: {self.context_dim}")
        print(f"Encoder type: {self.encoder_type}")
        print(f"Dual graph: {self.dual_graph}, Spectral loss weight: {self.spectral_loss_weight}, Mask-rec weight: {self.mask_rec_weight}")
        print(f"KL anneal steps: {self.kl_anneal_steps}, KL free bits: {self.kl_free_bits}")
        print(f"Disable KL: {self.disable_kl}")
        print(f"Batch adv weight: {self.batch_adv_weight}, Context align weight: {self.context_align_weight}")
        print(f"Mask strategy: {self.mask_strategy}")
        print(f"Gene stats weight: {self.gene_stats_weight}, Zero-rate weight: {self.zero_rate_weight}")
        print(f"Cell zero-rate weight: {self.cell_zero_rate_weight}")
        if self.gene_stats_subsample:
            print(f"Gene stats subsample: {self.gene_stats_subsample} genes")
        print(f"Gene stats anneal steps: {self.gene_stats_anneal_steps}, Zero-rate anneal steps: {self.zero_rate_anneal_steps}")
        print(f"Cell-type loss weight: {self.cell_type_loss_weight}, Dispersion weight: {self.dispersion_weight}")
        print(f"Libsize weight: {self.libsize_weight}")
        print(f"Pi temperature: {self.pi_temperature}, Pi bias: {self.pi_bias}, Diffusion temperature: {self.diffusion_temperature}, Pi blend: {self.pi_blend}")
        print(f"Gen calibrate means: {self.gen_calibrate_means}, Gen calibrate zero: {self.gen_calibrate_zero}")
        print(f"Gen match zero per cell: {self.gen_match_zero_per_cell}, Strength: {self.gen_zero_per_cell_strength}")
        print(f"Gen mean scale clip: [{self.gen_mean_scale_clip_low}, {self.gen_mean_scale_clip_high}]")
        print(f"Diffusion method: {self.diffusion_method}, Simple mode: {self.simple_mode}")
        print(f"Bio positional: {self.bio_positional}, Bio pos dim: {self.bio_pos_dim}")
        print(f"Transformer layers: {transformer_layers}, heads: {transformer_heads}, dropout: {transformer_dropout}")

        num_warmup_steps_captured = int(warmup_steps)
        num_training_steps_captured = int(total_steps)
        def lr_lambda_fn(current_scheduler_step):
            if num_training_steps_captured <= 0:
                return 1.0
            actual_warmup_steps = min(num_warmup_steps_captured, max(1, num_training_steps_captured))
            if current_scheduler_step < actual_warmup_steps:
                return float(current_scheduler_step + 1) / float(max(1, actual_warmup_steps))
            decay_phase_duration = num_training_steps_captured - actual_warmup_steps
            if decay_phase_duration <= 0:
                return 0.0
            current_step_in_decay = current_scheduler_step - actual_warmup_steps
            progress = float(current_step_in_decay) / float(max(1, decay_phase_duration))
            progress = min(1.0, progress)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda_fn)
        print("Trainer initialized.")

    def set_stage(self, stage, freeze_decoder=False):
        if stage == 1:
            self.loss_weights['diff'] = 0.0
            self.denoiser.requires_grad_(False)
            self.decoder.requires_grad_(True)
            print("Stage 1: diffusion loss off; denoiser frozen.")
        else:
            self.loss_weights['diff'] = self.base_loss_weights.get('diff', 1.0)
            self.denoiser.requires_grad_(True)
            self.decoder.requires_grad_(not freeze_decoder)
            msg = "Stage 2: diffusion loss on; denoiser unfrozen."
            if freeze_decoder:
                msg += " Decoder frozen."
            print(msg)

    def _bio_pos_enc(self, pos, dim):
        if pos is None or pos.numel() == 0:
            return None
        pos = pos.float()
        if pos.ndim != 2 or pos.size(1) == 0:
            return None
        pos = pos - pos.mean(dim=0, keepdim=True)
        pos = pos / (pos.std(dim=0, keepdim=True).clamp(min=1e-6))
        dim = int(dim)
        if dim <= 0:
            return None
        n_freq = max(1, dim // (2 * pos.size(1)))
        freqs = torch.logspace(0, 2, steps=n_freq, device=pos.device, dtype=pos.dtype)
        enc = []
        for i in range(pos.size(1)):
            for f in freqs:
                enc.append(torch.sin(pos[:, i] * f))
                enc.append(torch.cos(pos[:, i] * f))
        enc = torch.stack(enc, dim=1)
        if enc.size(1) < dim:
            pad = torch.zeros(pos.size(0), dim - enc.size(1), device=pos.device, dtype=pos.dtype)
            enc = torch.cat([enc, pad], dim=1)
        return enc[:, :dim]

    def train_epoch(self, loader, current_epoch_num):
        self.encoder.train(); self.denoiser.train(); self.decoder.train()
        total_loss_val, total_loss_diff_val, total_loss_kl_val, total_loss_rec_val = 0.0, 0.0, 0.0, 0.0
        total_loss_mask_val, total_loss_spec_val = 0.0, 0.0
        total_gene_stats_val, total_zero_rate_val, total_cell_zero_rate_val = 0.0, 0.0, 0.0
        num_batches_processed = 0

        pbar = tqdm(loader, desc=f"Epoch {current_epoch_num}", leave=False)
        for data in pbar:
            data = data.to(device)
            num_nodes_in_batch = data.x.size(0)
            if num_nodes_in_batch == 0: continue

            original_x = data.x
            masked_x = data.x.clone()
            mask = torch.zeros_like(masked_x, dtype=torch.bool)
            if self.encoder.training and self.input_masking_fraction > 0.0:
                if self.mask_strategy == "high_var" and self.mask_gene_probs is not None:
                    gene_probs = self.mask_gene_probs.to(masked_x.device)
                    mask = torch.rand_like(masked_x) < gene_probs
                else:
                    mask = torch.rand_like(masked_x) < self.input_masking_fraction
                masked_x[mask] = 0.0

            lap_pe = data.lap_pe
            if self.bio_positional and hasattr(data, 'pos') and data.pos is not None and data.pos.numel() > 0:
                bio_pe = self._bio_pos_enc(data.pos, self.bio_pos_dim)
                if bio_pe is not None:
                    lap_pe = bio_pe
            if lap_pe is None or lap_pe.size(0) != num_nodes_in_batch:
                lap_pe = torch.zeros(num_nodes_in_batch, self.encoder.pe_dim, device=device, dtype=masked_x.dtype)

            cell_type_labels = data.cell_type
            if cell_type_labels is None: cell_type_labels = torch.zeros(num_nodes_in_batch, dtype=torch.long, device=device)
            cell_type_labels = torch.clamp(cell_type_labels, 0, self.denoiser.num_cell_types - 1)

            edge_weights = None
            adversarially_perturbed_weights = None
            if self.encoder_type != "mlp":
                edge_weights = torch.ones(data.edge_index.size(1), device=device) if data.edge_index.numel() > 0 else None
                if edge_weights is not None:
                    initial_perturbed_weights = self.lap_pert.sample(data.edge_index, num_nodes_in_batch)
                    adversarially_perturbed_weights = self.lap_pert.adversarial(self.encoder, masked_x, data.edge_index, initial_perturbed_weights)
                    adversarially_perturbed_weights = torch.clamp(torch.nan_to_num(adversarially_perturbed_weights, nan=1.0), min=1e-4)

            # physical graph (spatial) if available
            edge_index_phys = getattr(data, 'edge_index_phys', None)
            phys_weights = None
            if self.encoder_type != "mlp" and edge_index_phys is not None and edge_index_phys.numel() > 0:
                phys_weights = self.lap_pert.sample(edge_index_phys, num_nodes_in_batch)
                phys_weights = self.lap_pert.adversarial(self.encoder, masked_x, edge_index_phys, phys_weights)
                phys_weights = torch.clamp(torch.nan_to_num(phys_weights, nan=1.0), min=1e-4)

            self.optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                mu, logvar = self.encoder(masked_x, data.edge_index, lap_pe, adversarially_perturbed_weights,
                                          edge_index_phys=edge_index_phys, edge_weight_phys=phys_weights)
                if mu.size(0) != num_nodes_in_batch: continue

                if torch.isnan(mu).any() or torch.isnan(logvar).any():
                    print(f"Warning: NaN/Inf detected in encoder outputs (mu/logvar). Skipping batch.")
                    continue
                
                if self.disable_kl:
                    logvar = torch.zeros_like(mu)
                    kl_div = torch.tensor(0.0, device=device)
                    z = mu
                else:
                    # Clamp logvar
                    logvar = torch.clamp(logvar, max=10)
                    kl_div_terms = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
                    if self.kl_free_bits is not None and self.kl_free_bits > 0:
                        kl_div_terms = torch.clamp(kl_div_terms, min=self.kl_free_bits)
                    # use per-dimension mean KL to avoid scale blow-up with large latent dims
                    kl_div = torch.mean(torch.relu(kl_div_terms), dim=-1).mean()
                    if torch.isnan(kl_div): kl_div = torch.tensor(0.0, device=device)

                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mu + std * eps
                loss_cls = torch.tensor(0.0, device=device)
                if self.cell_classifier is not None and cell_type_labels is not None:
                    logits = self.cell_classifier(z)
                    loss_cls = F.cross_entropy(logits, cell_type_labels)

                # Diffusion loss (skip entirely if weight is zero)
                loss_diff = torch.tensor(0.0, device=device)
                if self.loss_weights.get('diff', 0.0) > 0.0:
                    t_indices = torch.randint(0, self.diff.N, (num_nodes_in_batch,), device=device).long()
                    time_values = self.diff.timesteps[t_indices]
                    sigma_t = self.diff.marginal_std(time_values).unsqueeze(-1)
                    alpha_t = torch.exp(-time_values).unsqueeze(-1)
                    
                    noise_target = torch.randn_like(z)
                    zt = alpha_t * z.detach() + sigma_t * noise_target
                    
                    # Multimodal Conditioning
                    context_embedding = None
                    if hasattr(data, 'chromatin') and data.chromatin is not None and data.chromatin.size(1) > 0:
                         if data.chromatin.size(1) == self.denoiser.context_dim:
                             context_embedding = data.chromatin

                    eps_pred = self.denoiser(zt, time_values, cell_type_labels, context_embedding=context_embedding)
                    loss_diff = F.mse_loss(eps_pred, noise_target)
                    if torch.isnan(loss_diff): loss_diff = torch.tensor(0.0, device=device)

                # NB Loss
                log_mu_rec, log_theta_rec, logit_pi_rec, log_libsize_pred = self.decoder(z)
                
                mu_rec = torch.exp(torch.clamp(log_mu_rec, max=10)) 
                theta_rec = torch.exp(torch.clamp(log_theta_rec, min=-10, max=10))
                theta_rec = torch.clamp(theta_rec, min=1e-4, max=1e4)
                
                target_counts = original_x.long()
                libsize_true = target_counts.sum(dim=1, keepdim=True).float().clamp(min=1.0)
                mu_sum = mu_rec.sum(dim=1, keepdim=True).clamp(min=1.0)
                mu_rec = mu_rec * (libsize_true / mu_sum).detach()

                probs_rec = mu_rec / (mu_rec + theta_rec + 1e-6)
                probs_rec = torch.clamp(probs_rec, min=1e-4, max=1.0 - 1e-4)
                
                nb_dist = dist.NegativeBinomial(total_count=theta_rec, probs=probs_rec, validate_args=False)
                log_nb = nb_dist.log_prob(target_counts)
                if logit_pi_rec is None:
                    loss_rec = -log_nb.mean()
                else:
                    pi = torch.sigmoid(logit_pi_rec)
                    zero_mask = (target_counts == 0).float()
                    log_zero = torch.log(pi + (1.0 - pi) * torch.exp(log_nb) + 1e-8)
                    log_nonzero = torch.log(1.0 - pi + 1e-8) + log_nb
                    log_prob = zero_mask * log_zero + (1.0 - zero_mask) * log_nonzero
                    loss_rec = -log_prob.mean()
                if torch.isnan(loss_rec): loss_rec = torch.tensor(0.0, device=device)

                if log_libsize_pred is not None:
                    loss_libsize = F.mse_loss(log_libsize_pred, torch.log(libsize_true))
                else:
                    loss_libsize = torch.tensor(0.0, device=device)

                # masked reconstruction to emphasize inverse problem
                if mask.any():
                    masked_target = original_x[mask]
                    masked_mu = mu_rec[mask]
                    masked_theta = theta_rec[mask]
                    probs_mask = torch.clamp(masked_mu / (masked_mu + masked_theta + 1e-6), 1e-6, 1-1e-6)
                    nb_mask = dist.NegativeBinomial(total_count=masked_theta, probs=probs_mask, validate_args=False)
                    if logit_pi_rec is None:
                        loss_mask = -nb_mask.log_prob(masked_target).mean()
                    else:
                        pi_mask = torch.sigmoid(logit_pi_rec[mask])
                        log_nb_m = nb_mask.log_prob(masked_target)
                        zero_mask_m = (masked_target == 0).float()
                        log_zero_m = torch.log(pi_mask + (1.0 - pi_mask) * torch.exp(log_nb_m) + 1e-8)
                        log_nonzero_m = torch.log(1.0 - pi_mask + 1e-8) + log_nb_m
                        log_prob_m = zero_mask_m * log_zero_m + (1.0 - zero_mask_m) * log_nonzero_m
                        loss_mask = -log_prob_m.mean()
                else:
                    loss_mask = torch.tensor(0.0, device=device)

                # spectral alignment between feature and physical graphs
                spectral_loss = torch.tensor(0.0, device=device)
                if self.dual_graph and self.encoder_type != "mlp" and edge_index_phys is not None and edge_index_phys.numel() > 0 and mu.numel() > 0:
                    from laplace.utils import laplacian_smooth
                    mu_feat = laplacian_smooth(mu, data.edge_index, adversarially_perturbed_weights)
                    mu_phys = laplacian_smooth(mu, edge_index_phys, phys_weights)
                    spectral_loss = F.mse_loss(mu_feat, mu_phys)

                # gene-wise mean/variance + zero-rate calibration
                gene_stats_loss = torch.tensor(0.0, device=device)
                zero_rate_loss = torch.tensor(0.0, device=device)
                cell_zero_rate_loss = torch.tensor(0.0, device=device)
                dispersion_loss = torch.tensor(0.0, device=device)
                if (self.gene_stats_weight > 0.0 or self.zero_rate_weight > 0.0) and hasattr(self, 'gene_mean'):
                    target_mean = self.gene_mean.to(mu_rec.device)
                    target_var = self.gene_var.to(mu_rec.device) if hasattr(self, 'gene_var') and self.gene_var is not None else None
                    if self.gene_stats_subsample and target_mean.numel() > self.gene_stats_subsample:
                        idx = torch.randperm(target_mean.numel(), device=mu_rec.device)[:self.gene_stats_subsample]
                        mu_rec_s = mu_rec[:, idx]
                        theta_rec_s = theta_rec[:, idx]
                        log_theta_rec_s = log_theta_rec[:, idx]
                        logit_pi_s = logit_pi_rec[:, idx] if logit_pi_rec is not None else None
                        target_mean = target_mean[idx]
                        target_var = target_var[idx] if target_var is not None else None
                        target_zero = self.zero_rate.to(mu_rec.device)[idx] if hasattr(self, 'zero_rate') and self.zero_rate is not None else None
                    else:
                        mu_rec_s = mu_rec
                        theta_rec_s = theta_rec
                        log_theta_rec_s = log_theta_rec
                        logit_pi_s = logit_pi_rec
                        target_zero = self.zero_rate.to(mu_rec.device) if hasattr(self, 'zero_rate') and self.zero_rate is not None else None
                    if logit_pi_rec is None:
                        pi_eff = torch.zeros_like(mu_rec_s)
                    else:
                        pi_eff = torch.sigmoid(logit_pi_s)
                    exp_mean_cell = (1.0 - pi_eff) * mu_rec_s
                    exp_mean = exp_mean_cell.mean(dim=0)
                    if target_var is not None:
                        nb_var = mu_rec_s + (mu_rec_s ** 2) / (theta_rec_s + 1e-6)
                        exp_var_cell = (1.0 - pi_eff) * nb_var + pi_eff * (0.0 - exp_mean_cell) ** 2
                        exp_var = exp_var_cell.mean(dim=0) + exp_mean_cell.var(dim=0, unbiased=False)
                        mean_loss = F.mse_loss(torch.log1p(exp_mean), torch.log1p(target_mean))
                        var_loss = F.mse_loss(torch.log1p(exp_var), torch.log1p(target_var))
                        gene_stats_loss = mean_loss + var_loss
                    else:
                        gene_stats_loss = F.mse_loss(torch.log1p(exp_mean), torch.log1p(target_mean))
                    if target_zero is not None and self.zero_rate_weight > 0.0:
                        zero_nb = torch.exp(theta_rec_s * (torch.log(theta_rec_s + 1e-6) - torch.log(theta_rec_s + mu_rec_s + 1e-6)))
                        exp_zero = pi_eff + (1.0 - pi_eff) * zero_nb
                        zero_rate_loss = F.mse_loss(exp_zero.mean(dim=0), target_zero)
                    if self.dispersion_weight > 0.0 and target_var is not None:
                        denom = (target_var - target_mean).clamp(min=1e-6)
                        theta_target = (target_mean * target_mean / denom).clamp(min=1e-3, max=1e6)
                        log_theta_target = torch.log(theta_target)
                        log_theta_mean = log_theta_rec_s.mean(dim=0)
                        dispersion_loss = F.mse_loss(log_theta_mean, log_theta_target)
                if self.cell_zero_rate_weight > 0.0:
                    if logit_pi_rec is None:
                        pi_eff_full = torch.zeros_like(mu_rec)
                    else:
                        pi_eff_full = torch.sigmoid(logit_pi_rec)
                    zero_nb_full = torch.exp(theta_rec * (torch.log(theta_rec + 1e-6) - torch.log(theta_rec + mu_rec + 1e-6)))
                    exp_zero_full = pi_eff_full + (1.0 - pi_eff_full) * zero_nb_full
                    target_zero_cell = (target_counts == 0).float().mean(dim=1)
                    exp_zero_cell = exp_zero_full.mean(dim=1)
                    cell_zero_rate_loss = F.mse_loss(exp_zero_cell, target_zero_cell)

                # cross-modal alignment (context -> latent)
                context_align_loss = torch.tensor(0.0, device=device)
                if self.ctx_proj is not None and context_embedding is not None:
                    context_lat = self.ctx_proj(context_embedding)
                    context_align_loss = F.mse_loss(context_lat, mu)

                # batch adversarial invariance (DRO-style)
                batch_adv_loss = torch.tensor(0.0, device=device)
                if self.batch_discriminator is not None and hasattr(data, 'batch_id') and data.batch_id is not None:
                    batch_logits = self.batch_discriminator(grad_reverse(mu, lambda_=1.0))
                    batch_adv_loss = F.cross_entropy(batch_logits, data.batch_id)

                kl_weight = 0.0 if self.disable_kl else (min(1.0, self.current_step / self.kl_anneal_steps) * self.loss_weights['kl'])
                gene_stats_scale = min(1.0, float(self.current_step) / float(self.gene_stats_anneal_steps))
                zero_rate_scale = min(1.0, float(self.current_step) / float(self.zero_rate_anneal_steps))
                final_loss = (self.loss_weights['diff'] * loss_diff + 
                              kl_weight * kl_div + 
                              self.loss_weights['rec'] * loss_rec +
                              self.mask_rec_weight * loss_mask +
                              self.spectral_loss_weight * spectral_loss +
                              gene_stats_scale * self.gene_stats_weight * gene_stats_loss +
                              gene_stats_scale * self.dispersion_weight * dispersion_loss +
                              zero_rate_scale * self.zero_rate_weight * zero_rate_loss +
                              zero_rate_scale * self.cell_zero_rate_weight * cell_zero_rate_loss +
                              self.cell_type_loss_weight * loss_cls +
                              self.context_align_weight * context_align_loss +
                              self.batch_adv_weight * batch_adv_loss +
                              self.libsize_weight * loss_libsize)

            if torch.isnan(final_loss): 
                 print(f"Warning: NaN loss detected at epoch {current_epoch_num}, step {self.current_step}. Skipping batch.")
                 continue

            self.scaler.scale(final_loss).backward()
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
            self.scaler.step(self.optim)
            self.scaler.update()
            if getattr(self.optim, "_step_count", 0) > 0:
                self.scheduler.step()
            self.current_step += 1

            if self.current_step % 10 == 0:
                 pbar.set_postfix({'Total': final_loss.item(), 'Diff': loss_diff.item(), 'KL': kl_div.item(), 'Rec': loss_rec.item()})

            total_loss_val += final_loss.item()
            total_loss_diff_val += loss_diff.item()
            total_loss_kl_val += kl_div.item()
            total_loss_rec_val += loss_rec.item()
            total_loss_mask_val += loss_mask.item()
            total_loss_spec_val += spectral_loss.item()
            total_gene_stats_val += gene_stats_loss.item() if torch.is_tensor(gene_stats_loss) else float(gene_stats_loss)
            total_zero_rate_val += zero_rate_loss.item() if torch.is_tensor(zero_rate_loss) else float(zero_rate_loss)
            total_cell_zero_rate_val += cell_zero_rate_loss.item() if torch.is_tensor(cell_zero_rate_loss) else float(cell_zero_rate_loss)
            num_batches_processed += 1

        if num_batches_processed > 0:
            return (total_loss_val/num_batches_processed,
                    total_loss_diff_val/num_batches_processed,
                    total_loss_kl_val/num_batches_processed,
                    total_loss_rec_val/num_batches_processed,
                    total_loss_mask_val/num_batches_processed,
                    total_loss_spec_val/num_batches_processed,
                    total_gene_stats_val/num_batches_processed,
                    total_zero_rate_val/num_batches_processed,
                    total_cell_zero_rate_val/num_batches_processed)
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def state_dict(self):
        return {
            'encoder': self.encoder.state_dict(),
            'denoiser': self.denoiser.state_dict(),
            'decoder': self.decoder.state_dict(),
            'current_step': self.current_step
        }

    def evaluate_tstr(self, real_adata, generated_counts, generated_cell_types):
        print("\n--- Computing TSTR (Train on Synthetic, Test on Real) ---")
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, f1_score
            
            X_syn = generated_counts
            if isinstance(X_syn, torch.Tensor): X_syn = X_syn.cpu().numpy()
            if sp.issparse(X_syn): X_syn = X_syn.toarray()
            
            y_syn = generated_cell_types
            if isinstance(y_syn, torch.Tensor): y_syn = y_syn.cpu().numpy()
            y_syn = y_syn.astype(int)
            
            if not hasattr(real_adata, 'X') or 'cell_type' not in real_adata.obs.columns:
                 print("TSTR skipped: Real data missing X or cell_type."); return {}
            
            X_real = real_adata.X
            if sp.issparse(X_real): X_real = X_real.toarray()
            y_real = real_adata.obs['cell_type'].astype('category').cat.codes.values
            
            # Simple Log1p
            def process(X):
                s = X.sum(1, keepdims=True); s[s==0]=1
                return np.log1p(X/s*1e4)
            
            X_syn = process(X_syn)
            X_real = process(X_real)

            clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            clf.fit(X_syn, y_syn)
            y_pred = clf.predict(X_real)
            
            acc = accuracy_score(y_real, y_pred)
            f1 = f1_score(y_real, y_pred, average='weighted')
            print(f"TSTR Results -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
            return {"TSTR_Accuracy": acc, "TSTR_F1": f1}
        except Exception as e: print(f"Error TSTR: {e}"); return {}

    def evaluate_grn_recovery(self, real_adata, generated_counts):
        print("\n--- Computing GRN Recovery ---")
        try:
            X_syn = generated_counts
            if isinstance(X_syn, torch.Tensor): X_syn = X_syn.cpu().numpy()
            if sp.issparse(X_syn): X_syn = X_syn.toarray()
            
            X_real = real_adata.X
            if sp.issparse(X_real): X_real = X_real.toarray()
            
            n_genes = X_syn.shape[1]
            if n_genes > 2000:
                print("Selecting Top 2000 HVGs for GRN...")
                mean = X_real.mean(0); var = X_real.var(0); disp = var/(mean+1e-9)
                idx = np.argsort(disp)[-2000:]
                X_syn = X_syn[:, idx]
                X_real = X_real[:, idx]

            # remove zero-variance genes to avoid NaNs
            std_syn = X_syn.std(axis=0)
            std_real = X_real.std(axis=0)
            keep = (std_syn > 0) & (std_real > 0)
            X_syn = X_syn[:, keep]
            X_real = X_real[:, keep]
            std_syn = X_syn.std(axis=0)
            std_real = X_real.std(axis=0)
            keep = (std_syn > 0) & (std_real > 0)
            if np.sum(keep) < 2:
                print("Warning: Not enough variable genes for GRN correlation.")
                return {"GRN_Diff": float("nan"), "GRN_Spearman": float("nan")}
            X_syn = X_syn[:, keep]
            X_real = X_real[:, keep]
            corr_syn = np.corrcoef(X_syn, rowvar=False)
            corr_real = np.corrcoef(X_real, rowvar=False)
            corr_syn = np.nan_to_num(corr_syn)
            corr_real = np.nan_to_num(corr_real)
            
            diff_norm = np.linalg.norm(corr_real - corr_syn) / np.linalg.norm(corr_real)
            
            iu = np.triu_indices(corr_syn.shape[0], k=1)
            spearman = 0
            if len(iu[0]) > 0:
                from scipy.stats import spearmanr
                f_syn = corr_syn[iu]; f_real = corr_real[iu]
                if len(f_syn) > 100000:
                    idx = np.random.choice(len(f_syn), 100000, replace=False)
                    f_syn = f_syn[idx]; f_real = f_real[idx]
                spearman, _ = spearmanr(f_real, f_syn)
                
            print(f"GRN -> Diff: {diff_norm:.4f}, Spearman: {spearman:.4f}")
            return {"GRN_Diff": diff_norm, "GRN_Spearman": spearman}
        except Exception as e: print(f"Error GRN: {e}"); return {}

    @torch.no_grad()
    def evaluate_infill(self, real_adata, mask_fraction=0.2, num_samples=50):
        print(f"\n--- Computing Inpainting (Mask: {mask_fraction}) ---")
        try:
             X_real = real_adata.X
             if sp.issparse(X_real): X_real = X_real.toarray()
             idx = np.random.choice(X_real.shape[0], min(num_samples, X_real.shape[0]), replace=False)
             X_batch = torch.tensor(X_real[idx], device=device).float()
             
             mask = torch.rand_like(X_batch) < mask_fraction
             X_masked = X_batch.clone(); X_masked[mask] = 0.0
             
             print(f"Performing Diffusion-Based Inpainting for {num_samples} samples...")
             self.denoiser.eval(); self.decoder.eval()
             
             # diffusion-based inpainting using latent guidance
             def inpainting_guidance(z0_hat, t):
                 log_mu_rec, _, _, _ = self.decoder(z0_hat)
                 mu_rec = torch.exp(log_mu_rec)
                 observed_mask = ~mask
                 diff = (mu_rec - X_batch) * observed_mask.float()
                 loss = (diff ** 2).sum()
                 grad = torch.autograd.grad(loss, z0_hat)[0]
                 return -grad

             z_shape = (X_batch.size(0), self.diff.score_model.lat_dim)
             context_embedding = None
             if hasattr(real_adata, 'obsm') and 'chromatin' in real_adata.obsm:
                pass 

             z_inpainted = self.diff.sample_guided(z_shape, guidance_fn=inpainting_guidance, guidance_scale=100.0)
             
             log_mu_final, _, _, _ = self.decoder(z_inpainted)
             mu_rec = torch.exp(log_mu_final)
             
             imputed_vals = mu_rec[mask]
             true_vals = X_batch[mask]
             mse = F.mse_loss(imputed_vals, true_vals).item()
             
             try:
                 from scipy.stats import pearsonr
                 corr, _ = pearsonr(imputed_vals.cpu().numpy(), true_vals.cpu().numpy())
             except: corr = 0.0
             
             print(f"Inpainting MSE: {mse:.4f}, Correlation: {corr:.4f}")
             return {"Inpainting_MSE": mse, "Inpainting_Corr": corr}
        except Exception as e: print(f"Error Inpainting: {e}"); traceback.print_exc(); return {}

    @torch.no_grad()
    def evaluate_interpolation(self, real_adata, num_steps=10):
        """
        Performs latent space interpolation between different cell types.
        Simulates "differentiation trajectories" or transitions.
        """
        print(f"\n--- Computing Latent Interpolations (Geometric Surgery) ---")
        try:
            # get centroids of each cell type in real data
            if not hasattr(real_adata, 'X') or 'cell_type' not in real_adata.obs.columns:
                 print("Interpolation skipped: Missing data/labels."); return
            
            X_real = real_adata.X
            if sp.issparse(X_real): X_real = X_real.toarray()
            X_real = torch.tensor(X_real, device=device).float()
            
            labels = real_adata.obs['cell_type'].astype('category').cat.codes.values
            unique_labels = np.unique(labels)
            
            if len(unique_labels) < 2:
                 print("Interpolation skipped: < 2 cell types."); return

            print("Sampling Z for each cell type (Conditional Generation)...")
            z_centroids = {}
            for label in unique_labels:
                # Generate a batch of Z for this label
                z_shape = (50, self.diff.score_model.lat_dim)
                label_tensor = torch.full((50,), label, dtype=torch.long, device=device)
                z_gen = self.diff.sample(z_shape, cell_type_labels=label_tensor, method=self.diffusion_method) # this is the "Denoised Latent"
                z_centroids[label] = z_gen.mean(dim=0)
            
            # interpolate between pairs
            from itertools import combinations
            pairs = list(combinations(unique_labels, 2))
            if len(pairs) > 5: pairs = pairs[:5] 
            
            interpolated_trajectories = []
            
            for (l1, l2) in pairs:
                z1 = z_centroids[l1]
                z2 = z_centroids[l2]
                
                alphas = torch.linspace(0, 1, num_steps, device=device)
                trajectory_z = []
                for alpha in alphas:
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    trajectory_z.append(z_interp)
                
                trajectory_z = torch.stack(trajectory_z) # (Steps, LatDim)
                
                # Decode Trajectory
                log_mu, _, _, _ = self.decoder(trajectory_z)
                mu = torch.exp(log_mu).cpu().numpy()
                
                interpolated_trajectories.append({
                    'pair': (l1, l2),
                    'trajectory': mu
                })
                
            print(f"Computed interpolations for {len(interpolated_trajectories)} pairs.")
            return interpolated_trajectories
            print(f"Computed interpolations for {len(interpolated_trajectories)} pairs.")
            return interpolated_trajectories

        except Exception as e: print(f"Error Interpolation: {e}"); traceback.print_exc()

    def evaluate_marker_genes(self, real_adata, generated_counts, generated_cell_types, top_k=5):
        """
        Extracts marker genes for each cell type from Real Data and checks if they are 
        upregulated in the corresponding Generated Cell Type.
        Bio-Grounding Metric.
        """
        print("\n--- Computing Marker Gene Conservation ---")
        try:
            if not hasattr(real_adata, 'X') or 'cell_type' not in real_adata.obs.columns:
                 print("Marker Gene Eval skipped: Missing data/labels."); return {}

            # identify markers in real data (simple rank_genes_groups equivalent)
            print("Identifying Marker Genes in Real Data (Naive approach)...")
            # use scanpy's logic or a simple mean difference
            real_adata_raw = real_adata.copy()
            if not pd.api.types.is_categorical_dtype(real_adata_raw.obs['cell_type']):
                real_adata_raw.obs['cell_type'] = real_adata_raw.obs['cell_type'].astype('category')
            if sp.issparse(real_adata_raw.X): real_adata_raw.X = real_adata_raw.X.toarray()
            
            # normalize for comparison
            sc.pp.normalize_total(real_adata_raw, target_sum=1e4)
            sc.pp.log1p(real_adata_raw)
            
            sc.tl.rank_genes_groups(real_adata_raw, 'cell_type', method='t-test', use_raw=False)
            
            gen_counts_np = generated_counts.cpu().numpy() if isinstance(generated_counts, torch.Tensor) else generated_counts
            gen_types_np = generated_cell_types.cpu().numpy() if isinstance(generated_cell_types, torch.Tensor) else generated_cell_types
            
            # prepare generated AnnData for easy slicing
            gen_adata = sc.AnnData(X=gen_counts_np)
            gen_adata.obs['cell_type'] = gen_types_np
            # normalize gen data
            sc.pp.normalize_total(gen_adata, target_sum=1e4)
            sc.pp.log1p(gen_adata)
            
            marker_scores = []
            
            # check overlap/expression in generated
            groups = real_adata_raw.obs['cell_type'].unique()
            for group in groups:
                # top K markers for this group in real
                markers = real_adata_raw.uns['rank_genes_groups']['names'][group][:top_k]
                
                # check expression of these markers in gen clusters
                # get indices of genes
                gene_indices = [np.where(real_adata.var_names == m)[0][0] for m in markers if m in real_adata.var_names]
                
                if len(gene_indices) == 0: continue
                
                # Mean expression of markers in Gen Group
                gen_group_mask = gen_adata.obs['cell_type'] == (group if isinstance(group, int) else group) # handling codes vs names? 
                pass

            sc.tl.rank_genes_groups(gen_adata, 'cell_type', method='t-test', use_raw=False)
            overlaps = []
            for group in groups:
                 try:
                     real_markers = set(real_adata_raw.uns['rank_genes_groups']['names'][group][:50])
                     if group in gen_adata.obs['cell_type'].unique():
                         gen_markers = set(gen_adata.uns['rank_genes_groups']['names'][group][:50])
                         overlap = len(real_markers.intersection(gen_markers)) / 50.0
                         overlaps.append(overlap)
                 except: pass
            
            avg_overlap = np.mean(overlaps) if len(overlaps) > 0 else 0.0
            print(f"Marker Gene Overlap (Top 50): {avg_overlap:.4f}")
            return {"Marker_Overlap": avg_overlap}
            
        except Exception as e: print(f"Error Marker Genes: {e}"); return {}

    @torch.no_grad()
    def generate(self, num_samples, cell_type_condition=None, context_embedding=None):
        self.denoiser.eval(); self.decoder.eval()
        if cell_type_condition is None:
             if hasattr(self, 'cell_type_probs') and self.cell_type_probs is not None and self.denoiser.num_cell_types > 1:
                 probs = self.cell_type_probs.to(device)
                 gen_labels = torch.multinomial(probs, num_samples, replacement=True)
             else:
                 gen_labels = torch.zeros(num_samples, dtype=torch.long, device=device)
        else:
             gen_labels = torch.tensor(cell_type_condition, dtype=torch.long, device=device)
             gen_labels = torch.clamp(gen_labels, 0, self.denoiser.num_cell_types - 1)
        
        z_shape = (num_samples, self.diff.score_model.lat_dim)
        z_gen = self.diff.sample(
            z_shape,
            cell_type_labels=gen_labels,
            context_embedding=context_embedding,
            temperature=self.diffusion_temperature,
            method=self.diffusion_method,
        )
        
        log_mu, log_theta, logit_pi, log_libsize_pred = self.decoder(z_gen)
        mu = torch.exp(log_mu)
        theta = torch.clamp(torch.exp(log_theta), min=1e-6, max=1e6)
        libsize_target = None
        if hasattr(self, 'libsize_values') and self.libsize_values is not None and self.libsize_values.numel() > 0:
            lib_vals = self.libsize_values.to(mu.device)
            idx = torch.randint(0, lib_vals.numel(), (num_samples,), device=mu.device)
            libsize_target = lib_vals[idx].unsqueeze(1)
        elif log_libsize_pred is not None:
            libsize_target = torch.exp(log_libsize_pred).clamp(min=1.0)
        if libsize_target is not None:
            mu_sum = mu.sum(dim=1, keepdim=True).clamp(min=1.0)
            mu = mu * (libsize_target / mu_sum)
        
        probs = torch.clamp(mu / (mu + theta + 1e-6), min=1e-6, max=1-1e-6)
        nb = dist.NegativeBinomial(total_count=theta, probs=probs)
        if logit_pi is None:
            counts = nb.sample()
        else:
            pi_raw = torch.sigmoid((logit_pi + self.pi_bias) / max(self.pi_temperature, 1e-6))
            if self.gen_calibrate_means and hasattr(self, 'gene_mean') and self.gene_mean is not None:
                exp_mean = (1.0 - pi_raw) * mu
                target_mean = self.gene_mean.to(mu.device).unsqueeze(0)
                current_mean = exp_mean.mean(dim=0, keepdim=True).clamp(min=1e-6)
                scale = (target_mean / current_mean).clamp(min=self.gen_mean_scale_clip_low, max=self.gen_mean_scale_clip_high)
                mu = mu * scale
                probs = torch.clamp(mu / (mu + theta + 1e-6), min=1e-6, max=1-1e-6)
                nb = dist.NegativeBinomial(total_count=theta, probs=probs)
            if self.gen_calibrate_zero and hasattr(self, 'zero_rate') and self.zero_rate is not None:
                target_zero = self.zero_rate.to(pi_raw.device).unsqueeze(0)
                nb_zero = torch.exp(theta * (torch.log(theta + 1e-6) - torch.log(theta + mu + 1e-6)))
                total_zero = target_zero
                pi = (total_zero - nb_zero) / (1.0 - nb_zero + 1e-6)
                pi = torch.clamp(pi, min=1e-6, max=1.0 - 1e-6)
            elif hasattr(self, 'zero_rate') and self.zero_rate is not None and hasattr(self, 'pi_blend'):
                target_zero = self.zero_rate.to(pi_raw.device).unsqueeze(0)
                nb_zero = torch.exp(theta * (torch.log(theta + 1e-6) - torch.log(theta + mu + 1e-6)))
                total_zero = pi_raw + (1.0 - pi_raw) * nb_zero
                total_zero = torch.clamp(total_zero, min=1e-6, max=1.0 - 1e-6)
                total_zero = self.pi_blend * total_zero + (1.0 - self.pi_blend) * target_zero
                pi = (total_zero - nb_zero) / (1.0 - nb_zero + 1e-6)
                pi = torch.clamp(pi, min=1e-6, max=1.0 - 1e-6)
            else:
                pi = pi_raw
            nb_sample = nb.sample()
            zero_mask = (torch.rand_like(pi) < pi)
            nb_sample = nb_sample.masked_fill(zero_mask, 0)
            counts = nb_sample
        counts = torch.nan_to_num(counts.to(torch.int32).float(), nan=0.0)
        if self.gen_match_zero_per_cell and hasattr(self, 'zero_per_cell_values') and self.zero_per_cell_values is not None:
            zero_vals = self.zero_per_cell_values.to(counts.device)
            if zero_vals.numel() > 0:
                idx = torch.randint(0, zero_vals.numel(), (counts.size(0),), device=counts.device)
                target_zero_counts = zero_vals[idx]
                for i in range(counts.size(0)):
                    cur_zero = int((counts[i] == 0).sum().item())
                    tgt_zero = int(target_zero_counts[i].item())
                    diff = cur_zero - tgt_zero
                    if diff == 0:
                        continue
                    k = int(round(abs(diff) * float(self.gen_zero_per_cell_strength)))
                    if k <= 0:
                        continue
                    if diff > 0:
                        zero_idx = torch.nonzero(counts[i] == 0, as_tuple=False).squeeze(1)
                        if zero_idx.numel() == 0:
                            continue
                        pick = zero_idx[torch.randperm(zero_idx.numel(), device=counts.device)[:min(k, zero_idx.numel())]]
                        probs_i = mu[i, pick] / (mu[i, pick] + theta[i, pick] + 1e-6)
                        probs_i = torch.clamp(probs_i, min=1e-6, max=1-1e-6)
                        nb_i = dist.NegativeBinomial(total_count=theta[i, pick], probs=probs_i)
                        new_vals = nb_i.sample()
                        new_vals = torch.where(new_vals == 0, torch.ones_like(new_vals), new_vals)
                        counts[i, pick] = new_vals
                    else:
                        nz_idx = torch.nonzero(counts[i] > 0, as_tuple=False).squeeze(1)
                        if nz_idx.numel() == 0:
                            continue
                        vals = counts[i, nz_idx]
                        if vals.numel() > k:
                            _, order = torch.topk(vals, k, largest=False)
                            pick = nz_idx[order]
                        else:
                            pick = nz_idx
                        counts[i, pick] = 0
        return counts, gen_labels

    @torch.no_grad()
    def evaluate_generation_gpu(self, real_adata_X_tensor, gen_counts_tensor, gen_cell_types_tensor, real_cell_types_tensor=None, n_pcs=30, mmd_scales=[0.01, 0.1, 1, 10, 100]):
        """
        Evaluate Generation completely on GPU to save System RAM.
        real_adata_X_tensor: sparse or dense tensor on GPU or CPU. Prefer GPU if fits.
        """
        print("\\n--- Computing Evaluation Metrics ---")
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
             print(f"PCA Error: {e}"); return {}

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
    parser.add_argument('--cell_type_path', type=str, default="")
    parser.add_argument('--cell_type_col', type=str, default="cell_type")
    parser.add_argument('--barcode_col', type=str, default="barcode")
    parser.add_argument('--cell_type_unknown', type=str, default="Unknown")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--loss_weight_diff', type=float, default=0.5)
    parser.add_argument('--loss_weight_kl', type=float, default=0.005)
    parser.add_argument('--loss_weight_rec', type=float, default=0.5)
    parser.add_argument('--input_masking_fraction', type=float, default=0.2)
    parser.add_argument('--knn', type=int, default=7)
    parser.add_argument('--pe_dim', type=int, default=50)
    parser.add_argument('--pca_dim', type=int, default=50)
    parser.add_argument('--gene_threshold', type=int, default=20)
    parser.add_argument('--timesteps_diffusion', type=int, default=1000)
    parser.add_argument('--max_time', type=float, default=1.0)
    parser.add_argument('--viz', type=bool, default=True)
    parser.add_argument('--num_gnn_layers', type=int, default=3)
    parser.add_argument('--num_mlp_layers', type=int, default=3)
    parser.add_argument('--spectral_loss_weight', type=float, default=0.2)
    parser.add_argument('--mask_rec_weight', type=float, default=3.0)
    parser.add_argument('--batch_adv_weight', type=float, default=0.1)
    parser.add_argument('--context_align_weight', type=float, default=0.1)
    parser.add_argument('--mask_strategy', type=str, default="high_var", choices=["random", "high_var"])
    parser.add_argument('--gene_stats_weight', type=float, default=1.0)
    parser.add_argument('--zero_rate_weight', type=float, default=1.0)
    parser.add_argument('--cell_zero_rate_weight', type=float, default=0.0)
    parser.add_argument('--cell_type_loss_weight', type=float, default=0.0)
    parser.add_argument('--dispersion_weight', type=float, default=0.0)
    parser.add_argument('--libsize_weight', type=float, default=0.1)
    parser.add_argument('--pi_temperature', type=float, default=1.0)
    parser.add_argument('--pi_bias', type=float, default=0.0)
    parser.add_argument('--pi_blend', type=float, default=0.8)
    parser.add_argument('--diffusion_temperature', type=float, default=1.0)
    parser.add_argument('--gen_calibrate_means', action='store_true', default=False)
    parser.add_argument('--gen_calibrate_zero', action='store_true', default=False)
    parser.add_argument('--gen_match_zero_per_cell', action='store_true', default=False)
    parser.add_argument('--gen_zero_per_cell_strength', type=float, default=1.0)
    parser.add_argument('--gen_mean_scale_clip_low', type=float, default=0.25)
    parser.add_argument('--gen_mean_scale_clip_high', type=float, default=4.0)
    parser.add_argument('--diffusion_method', type=str, default="sde", choices=["sde", "ddim"])
    parser.add_argument('--simple_mode', action='store_true', default=False)
    parser.add_argument('--gene_stats_subsample', type=int, default=0)
    parser.add_argument('--gene_stats_anneal_steps', type=int, default=2000)
    parser.add_argument('--zero_rate_anneal_steps', type=int, default=2000)
    parser.add_argument('--early_stop_gene_stats', action='store_true', default=False)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4)
    parser.add_argument('--kl_anneal_steps', type=int, default=1000)
    parser.add_argument('--kl_free_bits', type=float, default=0.1)
    parser.add_argument('--disable_kl', action='store_true', default=False)
    parser.add_argument('--encoder_type', type=str, default="gnn", choices=["gnn", "mlp", "transformer"])
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_dropout', type=float, default=0.1)
    parser.add_argument('--bio_positional', action='store_true', default=False)
    parser.add_argument('--bio_pos_dim', type=int, default=32)
    parser.add_argument('--use_scgen_context', action='store_true', default=False)
    parser.add_argument('--scgen_hf_repo', type=str, default="")
    parser.add_argument('--scgen_local_dir', type=str, default="")
    parser.add_argument('--use_scvi_context', action='store_true', default=False)
    parser.add_argument('--scvi_hf_repo', type=str, default="")
    parser.add_argument('--scvi_local_dir', type=str, default="")
    parser.add_argument('--scvi_train', action='store_true', default=False)
    parser.add_argument('--scvi_train_dir', type=str, default="scvi_trained")
    parser.add_argument('--scvi_train_epochs', type=int, default=200)
    parser.add_argument('--scvi_train_batch_size', type=int, default=256)
    parser.add_argument('--dual_graph', action='store_true', default=True)
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--checkpoint_path', type=str, default="")
    parser.add_argument('--force_reprocess', action='store_true', default=False)
    parser.add_argument('--use_pseudo_labels', action='store_true', default=False)
    parser.add_argument('--stage2_start_epoch', type=int, default=0)
    parser.add_argument('--stage2_freeze_decoder', action='store_true', default=False)
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
    MAX_TIME = args.max_time
    GLOBAL_SEED = args.seed
    VIZ = args.viz
    INPUT_MASKING_FRACTION = args.input_masking_fraction
    SPECTRAL_LOSS_WEIGHT = args.spectral_loss_weight
    MASK_REC_WEIGHT = args.mask_rec_weight
    BATCH_ADV_WEIGHT = args.batch_adv_weight
    CONTEXT_ALIGN_WEIGHT = args.context_align_weight
    MASK_STRATEGY = args.mask_strategy
    GENE_STATS_WEIGHT = args.gene_stats_weight
    ZERO_RATE_WEIGHT = args.zero_rate_weight
    CELL_ZERO_RATE_WEIGHT = args.cell_zero_rate_weight
    CELL_TYPE_LOSS_WEIGHT = args.cell_type_loss_weight
    DISPERSION_WEIGHT = args.dispersion_weight
    LIBSIZE_WEIGHT = args.libsize_weight
    PI_TEMPERATURE = args.pi_temperature
    PI_BIAS = args.pi_bias
    PI_BLEND = args.pi_blend
    DIFFUSION_TEMPERATURE = args.diffusion_temperature
    GEN_CALIBRATE_MEANS = args.gen_calibrate_means
    GEN_CALIBRATE_ZERO = args.gen_calibrate_zero
    GEN_MATCH_ZERO_PER_CELL = args.gen_match_zero_per_cell
    GEN_ZERO_PER_CELL_STRENGTH = args.gen_zero_per_cell_strength
    GEN_MEAN_SCALE_CLIP_LOW = args.gen_mean_scale_clip_low
    GEN_MEAN_SCALE_CLIP_HIGH = args.gen_mean_scale_clip_high
    DIFFUSION_METHOD = args.diffusion_method
    SIMPLE_MODE = args.simple_mode
    GENE_STATS_SUBSAMPLE = args.gene_stats_subsample
    GENE_STATS_ANNEAL = args.gene_stats_anneal_steps
    ZERO_RATE_ANNEAL = args.zero_rate_anneal_steps
    EARLY_STOP_GENE_STATS = args.early_stop_gene_stats
    EARLY_STOP_PATIENCE = args.early_stop_patience
    EARLY_STOP_MIN_DELTA = args.early_stop_min_delta
    KL_ANNEAL_STEPS = args.kl_anneal_steps
    KL_FREE_BITS = args.kl_free_bits
    DISABLE_KL = args.disable_kl
    ENCODER_TYPE = args.encoder_type
    TRANSFORMER_LAYERS = args.transformer_layers
    TRANSFORMER_HEADS = args.transformer_heads
    TRANSFORMER_DROPOUT = args.transformer_dropout
    BIO_POSITIONAL = args.bio_positional
    BIO_POS_DIM = args.bio_pos_dim
    USE_SCGEN_CONTEXT = args.use_scgen_context
    SCGEN_HF_REPO = args.scgen_hf_repo
    SCGEN_LOCAL_DIR = args.scgen_local_dir
    USE_SCVI_CONTEXT = args.use_scvi_context
    SCVI_HF_REPO = args.scvi_hf_repo
    SCVI_LOCAL_DIR = args.scvi_local_dir
    SCVI_TRAIN = args.scvi_train
    SCVI_TRAIN_DIR = args.scvi_train_dir
    SCVI_TRAIN_EPOCHS = args.scvi_train_epochs
    SCVI_TRAIN_BATCH_SIZE = args.scvi_train_batch_size
    DUAL_GRAPH = args.dual_graph
    EVAL_ONLY = args.eval_only
    CHECKPOINT_PATH = args.checkpoint_path
    FORCE_REPROCESS = args.force_reprocess
    USE_PSEUDO_LABELS = args.use_pseudo_labels
    CELL_TYPE_PATH = args.cell_type_path.strip() if isinstance(args.cell_type_path, str) else ""
    CELL_TYPE_COL = args.cell_type_col
    BARCODE_COL = args.barcode_col
    CELL_TYPE_UNKNOWN = args.cell_type_unknown
    STAGE2_START_EPOCH = args.stage2_start_epoch
    STAGE2_FREEZE_DECODER = args.stage2_freeze_decoder
    
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
        train_dataset = LaplaceDataset(
            h5ad_path=TRAIN_H5AD, k_neighbors=K_NEIGHBORS, pe_dim=PE_DIM, root=TRAIN_DATA_ROOT, train=True,
            gene_threshold=GENE_THRESHOLD, pca_neighbors=PCA_NEIGHBORS, seed=GLOBAL_SEED, dataset_name='multiome',
            force_reprocess=FORCE_REPROCESS, use_pseudo_labels=USE_PSEUDO_LABELS,
            cell_type_path=CELL_TYPE_PATH or None, cell_type_col=CELL_TYPE_COL, barcode_col=BARCODE_COL,
            cell_type_unknown=CELL_TYPE_UNKNOWN
        )
        
        if train_dataset and len(train_dataset) > 0:
            num_train_cells = train_dataset.get(0).num_nodes
            input_feature_dim = train_dataset.get(0).x.size(1)
            num_cell_types = train_dataset.num_cell_types
            filtered_gene_names_from_train = train_dataset.filtered_gene_names
            print(f"Training data: {num_train_cells} cells, {input_feature_dim} genes, {num_cell_types} types.")
        else: raise ValueError("Training data is empty.")

        # Check for multimodal context
        context_dim = 0
        if hasattr(train_dataset.get(0), 'chromatin') and train_dataset.get(0).chromatin is not None:
            context_dim = train_dataset.get(0).chromatin.size(1)
            print(f"Detected Chromatin Context Dim: {context_dim}")

        num_batches = 1
        mask_gene_probs = None
        if train_dataset and len(train_dataset) > 0:
            data0 = train_dataset.get(0)
            if hasattr(data0, 'batch_id') and data0.batch_id is not None:
                num_batches = int(data0.batch_id.max().item()) + 1
            if MASK_STRATEGY == "high_var":
                x0 = data0.x
                if x0 is not None and x0.numel() > 0:
                    var = x0.var(dim=0, unbiased=False)
                    var = var / (var.mean() + 1e-8)
                    mask_gene_probs = torch.clamp(var * INPUT_MASKING_FRACTION, max=1.0).detach()

        scgen_context = None
        scgen_context_stats = None
        if USE_SCGEN_CONTEXT:
            try:
                import numpy as np
                from anndata import AnnData
                scgen_path = SCGEN_LOCAL_DIR
                if not scgen_path and SCGEN_HF_REPO:
                    try:
                        from huggingface_hub import snapshot_download
                        scgen_path = snapshot_download(repo_id=SCGEN_HF_REPO)
                    except Exception as e:
                        raise RuntimeError(f"HF download failed: {e}")
                if not scgen_path:
                    raise RuntimeError("No scGen path provided (use --scgen_local_dir or --scgen_hf_repo).")

                X_train = train_dataset.get(0).x.cpu().numpy()
                adata_scgen = AnnData(X=X_train)
                adata_scgen.var_names = train_dataset.filtered_gene_names if hasattr(train_dataset, 'filtered_gene_names') else None
                try:
                    from scvi.model import SCGEN as SCGENModel
                    scgen_model = SCGENModel.load(scgen_path, adata=adata_scgen)
                except Exception:
                    import scgen
                    scgen_model = scgen.SCGEN.load(scgen_path, adata=adata_scgen)

                scgen_latent = scgen_model.get_latent_representation(adata_scgen)
                scgen_context = torch.tensor(scgen_latent, device=device).float()
                train_dataset.get(0).chromatin = scgen_context
                context_dim = scgen_context.size(1)
                scgen_context_stats = (scgen_context.mean(dim=0), scgen_context.std(dim=0).clamp(min=1e-6))
                print(f"Using scGen latent context: {context_dim} dims.")
            except Exception as e:
                print(f"Warning: scGen context failed: {e}. Proceeding without scGen.")
                USE_SCGEN_CONTEXT = False

        scvi_context = None
        scvi_context_stats = None
        if USE_SCVI_CONTEXT:
            try:
                import numpy as np
                from anndata import AnnData
                from pathlib import Path
                if SCVI_TRAIN:
                    from scvi.model import SCVI
                    X_train = train_dataset.get(0).x.cpu().numpy()
                    X_train = np.clip(np.rint(X_train), 0, None)
                    train_genes = train_dataset.filtered_gene_names if hasattr(train_dataset, 'filtered_gene_names') else None
                    if train_genes is None:
                        train_genes = [f"gene_{i}" for i in range(X_train.shape[1])]
                    adata_scvi = AnnData(X=X_train)
                    adata_scvi.var_names = np.array(train_genes, dtype=str)
                    adata_scvi.obs["batch"] = "train"
                    adata_scvi.obs["batch"] = adata_scvi.obs["batch"].astype("category")
                    try:
                        import scvi
                        scvi.settings.verbosity = "error"
                    except Exception:
                        pass
                    SCVI.setup_anndata(adata_scvi, batch_key="batch")
                    scvi_model = SCVI(adata_scvi)
                    scvi_model.train(max_epochs=SCVI_TRAIN_EPOCHS, batch_size=SCVI_TRAIN_BATCH_SIZE)
                    if SCVI_TRAIN_DIR:
                        scvi_model.save(SCVI_TRAIN_DIR, overwrite=True)
                    scvi_latent = scvi_model.get_latent_representation(adata_scvi)
                    keep = np.ones(X_train.shape[0], dtype=bool)
                else:
                    scvi_path = Path(SCVI_LOCAL_DIR) if SCVI_LOCAL_DIR else None
                    if (scvi_path is None or not scvi_path.exists()) and SCVI_HF_REPO:
                        try:
                            from huggingface_hub import snapshot_download
                            scvi_path = Path(snapshot_download(repo_id=SCVI_HF_REPO, cache_dir="scvi_hub_cache"))
                        except Exception as e:
                            raise RuntimeError(f"HF download failed: {e}")
                    if scvi_path is None:
                        raise RuntimeError("No scVI path provided (use --scvi_local_dir or --scvi_hf_repo).")

                    import scanpy as sc
                    ref_adata = sc.read_h5ad(str(scvi_path / "adata.h5ad"))
                    ref_adata.var_names = ref_adata.var_names.astype(str)
                    ref_genes = ref_adata.var_names

                    X_train = train_dataset.get(0).x.cpu().numpy()
                    train_genes = train_dataset.filtered_gene_names if hasattr(train_dataset, 'filtered_gene_names') else None
                    if train_genes is None:
                        raise RuntimeError("Train gene names missing; cannot align to scVI reference.")

                    gene_to_idx = {g: i for i, g in enumerate(train_genes)}
                    X_aligned = np.zeros((X_train.shape[0], len(ref_genes)), dtype=X_train.dtype)
                    idx_target = []
                    idx_src = []
                    for i, g in enumerate(ref_genes):
                        if g in gene_to_idx:
                            idx_target.append(i)
                            idx_src.append(gene_to_idx[g])
                    if idx_target:
                        X_aligned[:, idx_target] = X_train[:, idx_src]

                    adata_scvi = AnnData(X=X_aligned)
                    adata_scvi.var_names = ref_genes.astype(str)

                    from scvi.model import SCVI
                    scvi_model = None
                    attr_path = scvi_path / "attr.pkl"
                    if attr_path.exists():
                        scvi_model = SCVI.load(scvi_path, adata=adata_scvi)
                        keep = np.ones(X_train.shape[0], dtype=bool)
                    else:
                        # HF scVI checkpoint direct load (scvi-tools 0.14.x)
                        ckpt = torch.load(str(scvi_path / "model.pt"), map_location="cpu", weights_only=False)
                        reg = ckpt.get("attr_dict", {}).get("registry_", {})
                        x_state = reg.get("field_registries", {}).get("X", {}).get("state_registry", {})
                        var_names = [str(v) for v in x_state.get("column_names", ref_genes)]
                        batch_state = reg.get("field_registries", {}).get("batch", {}).get("state_registry", {})
                        batch_categories = [str(v) for v in batch_state.get("categorical_mapping", ["0"])]

                        gene_to_idx = {g: i for i, g in enumerate(train_genes)}
                        X_aligned = np.zeros((X_train.shape[0], len(var_names)), dtype=X_train.dtype)
                        idx_target, idx_src = [], []
                        for i, g in enumerate(var_names):
                            if g in gene_to_idx:
                                idx_target.append(i)
                                idx_src.append(gene_to_idx[g])
                        if idx_target:
                            X_aligned[:, idx_target] = X_train[:, idx_src]
                        cell_sums = X_aligned.sum(axis=1)
                        keep = cell_sums > 0
                        X_aligned_filt = X_aligned[keep]
                        adata_scvi = AnnData(X=X_aligned_filt)
                        adata_scvi.var_names = np.array(var_names, dtype=str)
                        adata_scvi.obs["batch"] = batch_categories[0]
                        adata_scvi.obs["batch"] = adata_scvi.obs["batch"].astype("category")
                        adata_scvi.obs["batch"] = adata_scvi.obs["batch"].cat.set_categories(batch_categories)

                        try:
                            import scvi
                            scvi.settings.verbosity = "error"
                        except Exception:
                            pass
                        SCVI.setup_anndata(adata_scvi, batch_key="batch")
                        init_params = ckpt.get("attr_dict", {}).get("init_params_", {})
                        non_kwargs = init_params.get("non_kwargs", {})
                        kw_kwargs = init_params.get("kwargs", {}).get("kwargs", {})
                        state = ckpt.get("model_state_dict", {})
                        n_latent = state.get("z_encoder.mean_encoder.weight", torch.empty(0)).shape[0]
                        n_hidden = state.get("z_encoder.encoder.fc_layers.Layer 0.0.weight", torch.empty(0)).shape[0]
                        non_kwargs = dict(non_kwargs)
                        if n_latent > 0:
                            non_kwargs["n_latent"] = int(n_latent)
                        if n_hidden > 0:
                            non_kwargs["n_hidden"] = int(n_hidden)
                        scvi_model = SCVI(adata_scvi, **non_kwargs, **kw_kwargs)
                        scvi_model.module.load_state_dict(state, strict=True)
                        scvi_model.is_trained_ = True

                    scvi_latent = scvi_model.get_latent_representation(adata_scvi)
                # map back to full cell set to avoid batch mismatch
                scvi_context_full = np.zeros((X_train.shape[0], scvi_latent.shape[1]), dtype=scvi_latent.dtype)
                scvi_context_full[keep] = scvi_latent
                scvi_context = torch.tensor(scvi_context_full, device=device).float()
                train_dataset.get(0).chromatin = scvi_context
                if hasattr(train_dataset, "_data") and train_dataset._data is not None:
                    train_dataset._data.chromatin = scvi_context
                context_dim = scvi_context.size(1)
                scvi_context_stats = (scvi_context.mean(dim=0), scvi_context.std(dim=0).clamp(min=1e-6))
                print(f"Using scVI latent context: {context_dim} dims.")
            except Exception as e:
                print(f"Warning: scVI context failed: {e}. Proceeding without scVI.")
                USE_SCVI_CONTEXT = False

    except Exception as e:
        print(f"FATAL ERROR loading training data: {e}"); traceback.print_exc(); sys.exit(1)

    if num_train_cells > 0 and input_feature_dim > 0:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
        TOTAL_TRAINING_STEPS = len(train_loader) * EPOCHS
        WARMUP_STEPS = int(args.warmup)
        if WARMUP_STEPS <= 0:
            WARMUP_STEPS = max(1, int(0.05 * TOTAL_TRAINING_STEPS))
        WARMUP_STEPS = min(WARMUP_STEPS, max(1, TOTAL_TRAINING_STEPS // 2))
        trainer = Trainer(in_dim=input_feature_dim, hid_dim=HIDDEN_DIM, lat_dim=LATENT_DIM, num_cell_types=num_cell_types,
                        pe_dim=PE_DIM, timesteps=TIMESTEPS_DIFFUSION, max_time=MAX_TIME, lr=LEARNING_RATE, warmup_steps=WARMUP_STEPS,
                        total_steps=TOTAL_TRAINING_STEPS, loss_weights=loss_weights, input_masking_fraction=INPUT_MASKING_FRACTION,
                        context_dim=context_dim, dual_graph=DUAL_GRAPH, spectral_loss_weight=SPECTRAL_LOSS_WEIGHT,
                        mask_rec_weight=MASK_REC_WEIGHT, kl_anneal_steps=KL_ANNEAL_STEPS, kl_free_bits=KL_FREE_BITS,
                        encoder_type=ENCODER_TYPE, batch_adv_weight=BATCH_ADV_WEIGHT, num_batches=num_batches,
                        context_align_weight=CONTEXT_ALIGN_WEIGHT, mask_strategy=MASK_STRATEGY, mask_gene_probs=mask_gene_probs,
                        gene_stats_weight=GENE_STATS_WEIGHT, zero_rate_weight=ZERO_RATE_WEIGHT, cell_zero_rate_weight=CELL_ZERO_RATE_WEIGHT, disable_kl=DISABLE_KL,
                        cell_type_loss_weight=CELL_TYPE_LOSS_WEIGHT, dispersion_weight=DISPERSION_WEIGHT, libsize_weight=LIBSIZE_WEIGHT,
                        pi_temperature=PI_TEMPERATURE, pi_bias=PI_BIAS, pi_blend=PI_BLEND, diffusion_temperature=DIFFUSION_TEMPERATURE,
                        diffusion_method=DIFFUSION_METHOD, simple_mode=SIMPLE_MODE,
                        transformer_layers=TRANSFORMER_LAYERS, transformer_heads=TRANSFORMER_HEADS, transformer_dropout=TRANSFORMER_DROPOUT,
                        bio_positional=BIO_POSITIONAL, bio_pos_dim=BIO_POS_DIM,
                        gen_calibrate_means=GEN_CALIBRATE_MEANS, gen_calibrate_zero=GEN_CALIBRATE_ZERO,
                        gen_match_zero_per_cell=GEN_MATCH_ZERO_PER_CELL, gen_zero_per_cell_strength=GEN_ZERO_PER_CELL_STRENGTH,
                        gen_mean_scale_clip_low=GEN_MEAN_SCALE_CLIP_LOW, gen_mean_scale_clip_high=GEN_MEAN_SCALE_CLIP_HIGH,
                        gene_stats_subsample=GENE_STATS_SUBSAMPLE, gene_stats_anneal_steps=GENE_STATS_ANNEAL,
                        zero_rate_anneal_steps=ZERO_RATE_ANNEAL)
        if USE_SCGEN_CONTEXT and scgen_context_stats is not None:
            trainer.scgen_context_stats = scgen_context_stats
        if USE_SCVI_CONTEXT and scvi_context_stats is not None:
            trainer.scvi_context_stats = scvi_context_stats
        if train_dataset and len(train_dataset) > 0:
            data0 = train_dataset.get(0)
            if hasattr(train_dataset, 'log_libsize_mean') and train_dataset.log_libsize_mean is not None:
                trainer.log_libsize_mean = train_dataset.log_libsize_mean
                trainer.log_libsize_std = train_dataset.log_libsize_std
            if hasattr(data0, 'x') and data0.x is not None and data0.x.numel() > 0:
                x0f = data0.x.float()
                trainer.zero_rate = (x0f == 0).float().mean(dim=0).cpu()
                trainer.gene_mean = x0f.mean(dim=0).cpu()
                trainer.gene_var = x0f.var(dim=0, unbiased=False).cpu()
                trainer.libsize_values = x0f.sum(dim=1).cpu().clamp(min=1.0)
                trainer.zero_per_cell_values = (x0f == 0).sum(dim=1).cpu()
            if hasattr(data0, 'cell_type') and data0.cell_type is not None:
                counts = torch.bincount(data0.cell_type, minlength=num_cell_types).float()
                trainer.cell_type_probs = (counts / counts.sum()).cpu()

        def _resolve_checkpoint_path():
            if CHECKPOINT_PATH:
                return CHECKPOINT_PATH
            final_path = os.path.join(TRAIN_DATA_ROOT, 'trainer_final_state.pt')
            if os.path.exists(final_path):
                return final_path
            ckpts = sorted(glob.glob(os.path.join(TRAIN_DATA_ROOT, "trainer_checkpoint_epoch_*.pt")))
            return ckpts[-1] if ckpts else ""

        if EVAL_ONLY:
            ckpt_path = _resolve_checkpoint_path()
            if ckpt_path:
                state = torch.load(ckpt_path, map_location="cpu")
                def _load_matching(model, sd):
                    msd = model.state_dict()
                    filtered = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
                    model.load_state_dict(filtered, strict=False)
                _load_matching(trainer.encoder, state.get('encoder', {}))
                _load_matching(trainer.denoiser, state.get('denoiser', {}))
                _load_matching(trainer.decoder, state.get('decoder', {}))
                print(f"Loaded checkpoint for eval-only: {ckpt_path}")
            else:
                print("Warning: eval_only requested but no checkpoint found. Using current weights.")
        else:
            print(f"\nStarting training for {EPOCHS} epochs...")
            checkpoint_freq = max(1, int(EPOCHS * 0.1))
            print(f"Checkpoint frequency: Every {checkpoint_freq} epochs (approx 10% of total time).")

        stage2_start = STAGE2_START_EPOCH if STAGE2_START_EPOCH > 0 else max(1, EPOCHS // 2 + 1)
        stage2_start = min(stage2_start, EPOCHS + 1)
        epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training", leave=True)
        best_gene_stats = float("inf")
        no_improve = 0
        for epoch in epoch_bar:
            if epoch == 1:
                trainer.set_stage(1, freeze_decoder=False)
            if epoch == stage2_start:
                trainer.set_stage(2, freeze_decoder=STAGE2_FREEZE_DECODER)
            (avg_total_loss, avg_diff_loss, avg_kl_loss,
             avg_rec_loss, avg_mask_loss, avg_spec_loss,
             avg_gene_stats, avg_zero_rate, avg_cell_zero_rate) = trainer.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch:03d}/{EPOCHS} Summary -> "
                  f"Total: {avg_total_loss:.4f} | Diff: {avg_diff_loss:.4f} | KL: {avg_kl_loss:.4f} | "
                  f"Rec: {avg_rec_loss:.4f} | MaskRec: {avg_mask_loss:.4f} | SpecAlign: {avg_spec_loss:.4f} | "
                  f"GeneStats: {avg_gene_stats:.4f} | ZeroRate: {avg_zero_rate:.4f} | CellZeroRate: {avg_cell_zero_rate:.4f}")
            if EARLY_STOP_GENE_STATS:
                if avg_gene_stats + EARLY_STOP_MIN_DELTA < best_gene_stats:
                    best_gene_stats = avg_gene_stats
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping: gene_stats did not improve for {EARLY_STOP_PATIENCE} epochs.")
                    break
                if epoch % checkpoint_freq == 0 or epoch == EPOCHS:
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
            test_dataset = LaplaceDataset(
                h5ad_path=TRAIN_H5AD, k_neighbors=K_NEIGHBORS, pe_dim=PE_DIM, root=TEST_DATA_ROOT, train=False,
                gene_threshold=GENE_THRESHOLD, pca_neighbors=PCA_NEIGHBORS, seed=GLOBAL_SEED, dataset_name='multiome',
                force_reprocess=FORCE_REPROCESS, use_pseudo_labels=USE_PSEUDO_LABELS,
                cell_type_path=CELL_TYPE_PATH or None, cell_type_col=CELL_TYPE_COL, barcode_col=BARCODE_COL,
                cell_type_unknown=CELL_TYPE_UNKNOWN
            )
            if test_dataset and len(test_dataset) > 0:
                 data_test = test_dataset.get(0)
                 X_test_tensor = data_test.x
                 cell_types_test = data_test.cell_type
                 
                 test_adata = sc.AnnData(X=X_test_tensor.cpu().numpy())
                 test_adata.obs['cell_type'] = cell_types_test.cpu().numpy()
                 test_adata.var_names = test_dataset.filtered_gene_names if hasattr(test_dataset, 'filtered_gene_names') else filtered_gene_names_from_train # Use its OWN names first
                 
                 print(f"Test data loaded: {test_adata.shape[0]} cells.")
            else: print("Test dataset empty.")

        except Exception as e: print(f"FATAL ERROR loading test data: {e}"); traceback.print_exc(); test_adata = None

    if test_adata is not None:
         # --- ALIGNMENT ---
         target_genes = filtered_gene_names_from_train
         target_dim = len(target_genes)
         current_dim = test_adata.shape[1]
         
         if current_dim != target_dim:
             print(f"Aligning Test Data features ({current_dim}) to Train features ({target_dim})...")
             
             # Create mapping
             test_genes = test_adata.var_names.tolist()
             test_gene_to_idx = {g: i for i, g in enumerate(test_genes)}
             X_aligned = torch.zeros((test_adata.shape[0], target_dim), dtype=torch.float32)
             
             # Source X
             if isinstance(test_adata.X, torch.Tensor): src_X = test_adata.X.cpu()
             elif sp.issparse(test_adata.X): src_X = torch.from_numpy(test_adata.X.toarray())
             else: src_X = torch.from_numpy(test_adata.X)
             
             # Indices to copy
             indices_target = []
             indices_src = []
             
             for i, gene in enumerate(target_genes):
                 if gene in test_gene_to_idx:
                     indices_target.append(i)
                     indices_src.append(test_gene_to_idx[gene])
             
             if len(indices_target) > 0:
                 X_aligned[:, indices_target] = src_X[:, indices_src]
             
             print(f"Alignment complete. Feature overlap: {len(indices_target)}/{target_dim}.")
             
             # Replace test_adata
             test_adata = sc.AnnData(X=X_aligned.numpy())
             test_adata.obs['cell_type'] = cell_types_test.cpu().numpy()
             test_adata.var_names = target_genes # Now it matches!
             
         else:
             print("Test data features align with Training features.")

         num_test_cells = test_adata.shape[0]
         num_test_cells = test_adata.shape[0]
         
         if 'cell_type' not in test_adata.obs.columns:
             print("Warning: 'cell_type' not found in test_adata.obs.columns.")
             print("Performing Leiden clustering to generate substitute cell labels for conditional generation...")
             try:
                 sc.pp.log1p(test_adata) if np.max(test_adata.X) > 100 else None 
                 sc.pp.pca(test_adata, n_comps=min(50, test_adata.shape[1]-1))
                 sc.pp.neighbors(test_adata, n_neighbors=30)
                 sc.tl.leiden(test_adata, key_added='cell_type', resolution=0.5)
                 # Map string categories '0','1'... to integers
                 test_adata.obs['cell_type'] = test_adata.obs['cell_type'].astype(int)
                 print(f"Generated {len(test_adata.obs['cell_type'].unique())} Leiden clusters as proxy cell types.")
             except Exception as e:
                 print(f"Error during Leiden clustering: {e}. Conditional generation will be disabled.")
         
         cell_type_condition_for_gen = test_adata.obs['cell_type'].values if 'cell_type' in test_adata.obs else None
         all_mmd = {s: [] for s in [0.01, 0.1, 1, 10, 100]}
         all_w = []
         
         for i in range(3):
             print(f"\n--- Processing Dataset {i+1}/3 ---")
             try:
                 gen_batch_size = 128
                 num_batches = int(np.ceil(num_test_cells / gen_batch_size))
                 gen_counts_list, gen_types_list = [], []
                 
                 print(f"Generating {num_test_cells} cells...")
                 for b in range(num_batches):
                     start = b * gen_batch_size
                     end = min((b + 1) * gen_batch_size, num_test_cells)
                     cond = cell_type_condition_for_gen[start:end] if cell_type_condition_for_gen is not None else None

                     gen_context = None
                     if USE_SCVI_CONTEXT and hasattr(trainer, 'scvi_context_stats'):
                         mean_ctx, std_ctx = trainer.scvi_context_stats
                         gen_context = torch.randn((end-start, mean_ctx.numel()), device=device) * std_ctx + mean_ctx
                     elif USE_SCGEN_CONTEXT and hasattr(trainer, 'scgen_context_stats'):
                         mean_ctx, std_ctx = trainer.scgen_context_stats
                         gen_context = torch.randn((end-start, mean_ctx.numel()), device=device) * std_ctx + mean_ctx
                     bc, bt = trainer.generate(num_samples=end-start, cell_type_condition=cond, context_embedding=gen_context)
                     gen_counts_list.append(bc) # These are TENSORS now
                     gen_types_list.append(bt)
                 
                 gen_counts = torch.cat(gen_counts_list, dim=0)
                 gen_types = torch.cat(gen_types_list, dim=0)
                 del gen_counts_list, gen_types_list
                 
                 # Evaluate on GPU
                 real_adata_X_tensor = torch.tensor(test_adata.X).float() 
                 real_labels_tensor = torch.tensor(test_adata.obs['cell_type'].values).long()
                 
                 metrics = trainer.evaluate_generation_gpu(real_adata_X_tensor, gen_counts, gen_types, real_labels_tensor)
                 print(f"Metrics: {metrics}")
                 
                 # TSTR
                 tstr = trainer.evaluate_tstr(real_adata=test_adata, generated_counts=gen_counts, generated_cell_types=gen_types)
                 
                 # GRN
                 grn = trainer.evaluate_grn_recovery(real_adata=test_adata, generated_counts=gen_counts)

                 # Marker Gene Conservation
                 markers = trainer.evaluate_marker_genes(real_adata=test_adata, generated_counts=gen_counts, generated_cell_types=gen_types)

                 if VIZ:
                      output_plot_dir = _experiment_plot_dir(args, "multiome")
                      plot_zero_rate_calibration(
                          real_adata_filtered=test_adata,
                          generated_counts=gen_counts,
                          output_dir=output_plot_dir,
                          model_name="LapDDPM"
                      )

                 if "MMD" in metrics:
                      for k, v in metrics["MMD"].items():
                          s = float(k.split('_')[-1])
                          if not np.isnan(v): all_mmd[s].append(v)
                 if "Wasserstein" in metrics:
                      for k, v in metrics["Wasserstein"].items():
                          if not np.isnan(v): all_w.append(v)
             
             except Exception as e: print(f"Error {e}"); traceback.print_exc()
         try:
             if VIZ:
                 trainer.evaluate_infill(real_adata=test_adata, mask_fraction=0.2, num_samples=50)
                 
             if VIZ:
                 trainer.evaluate_interpolation(real_adata=test_adata, num_steps=10)
                 
         except Exception as e: print(f"Error Inpainting/Interp loop: {e}")
             
    print("\nScript execution finished.")
