import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import ChebConv
from torch_geometric.utils import add_self_loops
import numpy as np
import scipy.sparse as sp   
from .seed import *
from .utils import *

class LaplacianPerturb:
    def __init__(self, alpha_min=1e-3, alpha_max=1e-2):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def sample(self, edge_index, num_nodes):
        current_device = edge_index.device if edge_index.numel() > 0 else device
        if edge_index.numel() == 0: return torch.empty((0,), device=current_device)
        alpha = torch.rand(1, device=current_device, dtype=torch.float32) * (self.alpha_max - self.alpha_min) + self.alpha_min
        signs = torch.randint(0, 2, (edge_index.size(1),), device=current_device, dtype=torch.float32) * 2.0 - 1.0
        return 1.0 + alpha * signs

    def adversarial(self, model_not_used, x, edge_index, current_weights, xi=1e-6, epsilon=0.1, ip=3):
        num_nodes = x.size(0)
        current_device = x.device
        if num_nodes == 0 or edge_index.numel() == 0 or current_weights is None or current_weights.numel() == 0:
            return current_weights.clone() if current_weights is not None else None

        current_weights = torch.relu(current_weights) + 1e-8
        if edge_index.max() >= num_nodes or edge_index.min() < 0:
             print(f"Warning: Edge index out of bounds for adversarial perturbation. Skipping.")
             return current_weights.clone()

        adj = torch.sparse_coo_tensor(edge_index, current_weights.float(), (num_nodes, num_nodes), device=current_device).coalesce()
        perturbed_weights = current_weights.clone()
        with torch.no_grad():
            v = torch.randn(num_nodes, 1, device=current_device, dtype=x.dtype) * 0.01
            v = v / (v.norm() + 1e-8)
            for _ in range(ip):
                v_new = torch.sparse.mm(adj, v)
                v_norm = v_new.norm()
                if v_norm > 1e-9: v = v_new / v_norm
                else: v = torch.randn(num_nodes, 1, device=current_device, dtype=x.dtype) * 0.01; v = v / (v.norm() + 1e-8)

            if edge_index.size(1) == 0:
                edge_specific_perturbation_term = torch.empty(0, device=current_device, dtype=x.dtype)
            else:
                 v_i = v[edge_index[0]]
                 v_j = v[edge_index[1]]
                 edge_specific_perturbation_term = xi * v_i.squeeze(-1) * v_j.squeeze(-1)
                 if edge_specific_perturbation_term.shape != current_weights.shape:
                     print(f"Warning: Shape mismatch in adversarial perturbation. Skipping.")
                     return current_weights.clone()
            perturbed_weights = current_weights + epsilon * edge_specific_perturbation_term
        return perturbed_weights.clamp(min=1e-4, max=10.0)

class SpectralEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim, pe_dim):
        super().__init__()
        self.pe_dim = pe_dim
        self.cheb1 = ChebConv(in_dim + pe_dim, hid_dim, K=4)
        self.cheb2 = ChebConv(hid_dim, hid_dim, K=4)
        self.mu_net = nn.Linear(hid_dim, lat_dim)
        self.logvar_net = nn.Linear(hid_dim, lat_dim)

    def forward(self, x, edge_index, lap_pe, edge_weight=None):
        current_device = x.device
        num_nodes = x.size(0)
        if num_nodes == 0:
             return torch.empty(0, self.mu_net.out_features, device=current_device, dtype=x.dtype), \
                    torch.empty(0, self.logvar_net.out_features, device=current_device, dtype=x.dtype)

        if lap_pe is None or lap_pe.size(0) != num_nodes or lap_pe.size(1) != self.pe_dim:
            lap_pe = torch.zeros(num_nodes, self.pe_dim, device=current_device, dtype=x.dtype)
        x_combined = torch.cat([x, lap_pe], dim=1)

        edge_index_sl, edge_weight_sl = edge_index, edge_weight
        if edge_index.numel() > 0:
            edge_index_sl, edge_weight_sl = torch_geometric.utils.add_self_loops(edge_index, edge_weight, num_nodes=num_nodes, fill_value=1.0)
        elif num_nodes > 0:
             edge_index_sl = torch.arange(num_nodes, device=current_device).repeat(2, 1)
             edge_weight_sl = torch.ones(num_nodes, device=current_device, dtype=x.dtype)
        else:
             edge_index_sl = torch.empty((2,0), dtype=torch.long, device=current_device)
             edge_weight_sl = None

        if edge_weight_sl is not None:
             if edge_weight_sl.device != current_device: edge_weight_sl = edge_weight_sl.to(current_device)
             if edge_weight_sl.dtype != x_combined.dtype: edge_weight_sl = edge_weight_sl.to(x_combined.dtype)

        h = F.relu(self.cheb1(x_combined, edge_index_sl, edge_weight_sl))
        h = F.relu(self.cheb2(h, edge_index_sl, edge_weight_sl))
        mu = self.mu_net(h)
        logvar = self.logvar_net(h)
        return mu, logvar

class ScoreNet(nn.Module):
    def __init__(self, lat_dim, num_cell_types, time_embed_dim=32, hid_dim_mlp=512): # Added hid_dim_mlp
        super().__init__()
        self.lat_dim = lat_dim
        self.num_cell_types = num_cell_types
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.cell_type_embed = nn.Embedding(num_cell_types, time_embed_dim)

        # MLP with LayerNorm
        self.mlp = nn.Sequential(
            nn.Linear(lat_dim + time_embed_dim + time_embed_dim, hid_dim_mlp),
            nn.LayerNorm(hid_dim_mlp), # Added LayerNorm
            nn.ReLU(),
            nn.Linear(hid_dim_mlp, hid_dim_mlp),
            nn.LayerNorm(hid_dim_mlp), # Added LayerNorm
            nn.ReLU(),
            nn.Linear(hid_dim_mlp, lat_dim)
        )
        self.cond_drop_prob = 0.1

    def forward(self, zt, time_t, cell_type_labels=None):
        current_device = zt.device
        num_nodes = zt.size(0)
        if num_nodes == 0:
            return torch.empty(0, self.lat_dim, device=current_device, dtype=zt.dtype)

        if not isinstance(time_t, torch.Tensor):
            time_t_tensor = torch.full((num_nodes,), time_t, device=current_device, dtype=zt.dtype) # Ensure time_t is a tensor of size num_nodes
        else:
            if time_t.device != current_device: time_t = time_t.to(current_device)
            if time_t.dtype != zt.dtype: time_t = time_t.to(zt.dtype)
            if time_t.ndim == 0: time_t_tensor = time_t.unsqueeze(0).expand(num_nodes)
            elif time_t.ndim == 1 and time_t.size(0) != num_nodes: raise ValueError(f"Time tensor 1D shape mismatch")
            elif time_t.ndim == 2 and time_t.size(1) == 1 and time_t.size(0) == num_nodes: time_t_tensor = time_t.squeeze(1) # Remove last dim if size 1
            elif time_t.ndim == 2 and time_t.size(0) == num_nodes and time_t.size(1) != 1: raise ValueError(f"Time tensor 2D shape mismatch")
            elif time_t.ndim == 1 and time_t.size(0) == num_nodes: time_t_tensor = time_t # Already correct shape
            else: raise ValueError(f"Unexpected time_t tensor shape: {time_t.shape}")


        time_embedding = self.time_mlp(time_t_tensor.unsqueeze(-1)) # Add last dim for MLP

        if cell_type_labels is not None:
            if cell_type_labels.ndim == 0: cell_type_labels = cell_type_labels.unsqueeze(0)
            if cell_type_labels.size(0) != num_nodes and num_nodes > 0:
                 raise ValueError(f"Batch size mismatch for cell_type_labels and zt")
            is_dropout = self.training and torch.rand(1).item() < self.cond_drop_prob # self.training here refers to nn.Module.training
            if is_dropout:
                cell_type_embedding = torch.zeros(num_nodes, self.time_embed_dim, device=current_device, dtype=zt.dtype)
            else:
                 if cell_type_labels.max() >= self.num_cell_types or cell_type_labels.min() < 0:
                     cell_type_labels_clamped = torch.clamp(cell_type_labels, 0, self.num_cell_types - 1)
                     cell_type_embedding = self.cell_type_embed(cell_type_labels_clamped)
                 else:
                    cell_type_embedding = self.cell_type_embed(cell_type_labels)
        else:
            cell_type_embedding = torch.zeros(num_nodes, self.time_embed_dim, device=current_device, dtype=zt.dtype)

        combined_input = torch.cat([zt, time_embedding, cell_type_embedding], dim=1)
        score_val = self.mlp(combined_input)
        return score_val

class ScoreSDE(nn.Module):
    def __init__(self, score_model, T=1.0, N=1000):
        super().__init__()
        self.score_model = score_model
        self.T = T
        self.N = N
        self.timesteps = torch.linspace(T, T / N, N, device=device, dtype=torch.float32)

    def marginal_std(self, t):
        if not isinstance(t, torch.Tensor): t = torch.tensor(t, device=self.timesteps.device, dtype=torch.float32)
        elif t.device != self.timesteps.device: t = t.to(self.timesteps.device)
        if t.dtype != torch.float32 and t.dtype != torch.float64: t = t.float()
        if t.ndim == 0: t = t.unsqueeze(0)
        return torch.sqrt(1. - torch.exp(-2 * t) + 1e-8)

    @torch.no_grad()
    def sample(self, z_shape, cell_type_labels=None):
        current_device = self.timesteps.device
        num_samples, lat_dim = z_shape
        if num_samples == 0: return torch.empty(z_shape, device=current_device, dtype=torch.float32)

        z = torch.randn(z_shape, device=current_device, dtype=torch.float32)
        dt = self.T / self.N

        if cell_type_labels is not None:
            if cell_type_labels.device != current_device: cell_type_labels = cell_type_labels.to(current_device)
            if cell_type_labels.size(0) == 1 and num_samples > 1: cell_type_labels = cell_type_labels.repeat(num_samples)
            elif cell_type_labels.size(0) != num_samples: raise ValueError(f"Cell type labels size mismatch")

        for i in range(self.N):
            t_val_float = self.timesteps[i].item()
            t_tensor_for_model = torch.full((num_samples,), t_val_float, device=current_device, dtype=z.dtype) # Pass time as tensor of size num_samples
            sigma_t = self.marginal_std(t_val_float)
            predicted_epsilon = self.score_model(z, t_tensor_for_model, cell_type_labels)
            sigma_t_safe = sigma_t + 1e-8
            drift = 2 * predicted_epsilon / sigma_t_safe
            diffusion_coeff_input = torch.tensor(2 * dt + 1e-8, device=current_device, dtype=z.dtype)
            diffusion_coeff = torch.sqrt(diffusion_coeff_input)
            z = z + drift * dt + diffusion_coeff * torch.randn_like(z)
        return z

class FeatureDecoder(nn.Module):
    def __init__(self, lat_dim, hid_dim, out_dim): # hid_dim is used for MLP layers
        super().__init__()
        self.decoder_mlp = nn.Sequential(
            nn.Linear(lat_dim, hid_dim),
            nn.LayerNorm(hid_dim), # Added LayerNorm
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim), # Added LayerNorm
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, z):
        if z.size(0) == 0:
            return torch.empty(0, self.decoder_mlp[-1].out_features, device=z.device, dtype=z.dtype)
        log_rates = self.decoder_mlp(z)
        return log_rates