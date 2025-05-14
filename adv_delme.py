import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import ChebConv
import torch_geometric.utils

# Enable CuDNN benchmarking for optimal kernels
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MerfishCellGraphDataset(InMemoryDataset):
    def __init__(self, csv_path, k=7, root='data', train=True):
        self.train = train
        self.csv_path = csv_path
        self.k = k
        super().__init__(root)
        # Data loading is conditional on processed files existing.
        # If they don't exist, process() will be called, which then saves them.
        # __init__ should ensure that self.data and self.slices are loaded if processed files are present.
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(f"Successfully loaded processed data from {self.processed_paths[0]}")
        except FileNotFoundError:
            print(f"Processed file not found at {self.processed_paths[0]}. Will run process().")
            # process() will be called by InMemoryDataset logic if files not found
        except Exception as e:
            print(f"Error loading processed data: {e}. Will attempt to re-process.")


    @property
    def raw_file_names(self):
        return [] # No raw files to download, data is from CSV
    
    @property
    def processed_file_names(self):
        # Differentiate processed file names for train and test sets
        return [f'processed_{"train" if self.train else "test"}_k{self.k}.pt']


    def download(self):
        # This method is called if raw_file_names are not found in raw_dir.
        # Since we process from a local CSV, we don't need to download anything.
        pass

    def process(self):
        print(f"Processing data from CSV: {self.csv_path}")
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"ERROR in process(): CSV file not found at {self.csv_path}")
            raise
        
        # Example: Extract gene columns (adjust as per your actual CSV structure)
        # This assumes '1700022I11Rik' and 'Gad1' are valid column names.
        # If not, this will raise an error. Add error handling or ensure columns exist.
        try:
            gene_cols_start_idx = df.columns.get_loc('1700022I11Rik')
            gene_cols_end_idx = df.columns.get_loc('Gad1') + 1
            gene_cols = df.columns[gene_cols_start_idx:gene_cols_end_idx].tolist()
        except KeyError as e:
            print(f"ERROR: One or more gene column markers not found in CSV: {e}")
            print(f"Available columns: {df.columns.tolist()}")
            raise
        
        feats = df[gene_cols].values.astype(np.float32) # Ensure float32 for PyTorch
        coords = df[['coord_X', 'coord_Y']].values.astype(np.float32)

        if coords.shape[0] == 0:
            print("Warning: No coordinates found in CSV. Creating an empty graph.")
            data_list = [Data(x=torch.empty((0, feats.shape[1]), dtype=torch.float),
                               edge_index=torch.empty((2,0), dtype=torch.long),
                               pos=torch.empty((0,2), dtype=torch.float),
                               lap_pe=torch.empty((0,10), dtype=torch.float) # Assuming pe_dim=10
                               )]
            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        
        # Build k-NN graph
        if coords.shape[0] <= self.k:
            print(f"Warning: Number of samples ({coords.shape[0]}) is less than or equal to k ({self.k}). Adjusting k for k-NN.")
            actual_k = max(1, coords.shape[0] -1) # Need at least 1 neighbor if possible
            if actual_k == 0 : # Only one node
                 edge_index = torch.empty((2,0), dtype=torch.long)
            else:
                nbrs = NearestNeighbors(n_neighbors=actual_k + 1, algorithm='auto').fit(coords)
                _, idx = nbrs.kneighbors(coords)
        else:
            nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='auto').fit(coords)
            _, idx = nbrs.kneighbors(coords)

        edges = []
        if coords.shape[0] > 1 and actual_k > 0 if coords.shape[0] <= self.k else True : # Check if idx exists and is usable
            for i, nbr_indices in enumerate(idx):
                for j_idx in nbr_indices[1:]: # Exclude self (0-th neighbor)
                    # Ensure j_idx is a valid index for coords
                    if i != j_idx and 0 <= j_idx < coords.shape[0]:
                        edges.append([i, j_idx])
        
        if not edges:
            edge_index = torch.empty((2,0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        data = Data(
            x=torch.tensor(feats, dtype=torch.float),
            edge_index=edge_index,
            pos=torch.tensor(coords, dtype=torch.float)
            # gene_cols can be stored if needed for inspection: gene_cols=gene_cols
        )
        
        # Compute and store Laplacian PE
        pe_dim = 10 # Should match SpectralEncoder's pe_dim
        max_nodes_for_pe = 500000 
        if data.num_nodes > max_nodes_for_pe:
            print(f"Warning: Graph too large for PE computation ({data.num_nodes} nodes). "
                  f"Max supported: {max_nodes_for_pe}. Assigning zero PEs.")
            data.lap_pe = torch.zeros((data.num_nodes, pe_dim), dtype=torch.float)
        elif data.num_nodes == 0:
             print(f"Warning: Graph has 0 nodes. Assigning empty PE tensor.")
             data.lap_pe = torch.zeros((0, pe_dim), dtype=torch.float)
        elif data.edge_index.numel() == 0:
            print(f"Warning: Graph has {data.num_nodes} nodes but no edges. Cannot compute Laplacian PE. Assigning zero PEs.")
            data.lap_pe = torch.zeros((data.num_nodes, pe_dim), dtype=torch.float)
        else:
            data.lap_pe = compute_lap_pe(data.edge_index, data.num_nodes, k=pe_dim)
        
        # Collate and save the processed data.
        # self.collate expects a list of Data objects.
        data_list = [data]
        torch.save(self.collate(data_list), self.processed_paths[0])
        print(f"Successfully processed and saved data to {self.processed_paths[0]}")


def compute_lap_pe(edge_index, num_nodes, k=10):
    """
    Computes Laplacian Positional Encodings (PE).
    Args:
        edge_index (Tensor): Edge indices of the graph.
        num_nodes (int): Number of nodes in the graph.
        k (int): Number of smallest non-trivial eigenvectors to use.
    Returns:
        Tensor: Laplacian PE of shape [num_nodes, k].
    """
    target_device = edge_index.device if edge_index.numel() > 0 else torch.device('cpu')

    if num_nodes == 0:
        return torch.zeros((0, k), device=target_device)
    if edge_index.numel() == 0: # No edges, PE is typically zero or undefined
        print("Warning: No edges found for PE computation. Returning zero PEs.")
        return torch.zeros((num_nodes, k), device=target_device)

    edge_index_np = edge_index.cpu().numpy()
    
    if edge_index_np.size == 0:
        return torch.zeros((num_nodes, k), device=target_device)

    # Ensure indices are within bounds
    if edge_index_np.max() >= num_nodes:
        print(f"Warning: Max edge index {edge_index_np.max()} >= num_nodes {num_nodes}. Clamping indices.")
        edge_index_np = np.clip(edge_index_np, 0, num_nodes - 1)


    data_np = np.ones(edge_index_np.shape[1])

    try:
        adj = sp.coo_matrix((data_np, (edge_index_np[0], edge_index_np[1])), 
                           shape=(num_nodes, num_nodes))
    except Exception as e:
        print(f"Error creating sparse adj matrix: {e}")
        return torch.zeros((num_nodes, k), device=target_device)

    adj = adj + adj.T 
    adj.data = np.clip(adj.data, 0, 1)  

    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12)) 
    deg_inv_sqrt_mat = sp.diags(deg_inv_sqrt)
    
    L = sp.eye(num_nodes) - deg_inv_sqrt_mat @ adj @ deg_inv_sqrt_mat

    num_eigenvectors_to_compute = min(k + 1, num_nodes -1) 
    if num_eigenvectors_to_compute <= 0:
         return torch.zeros((num_nodes, k), device=target_device)

    try:
        eigvals, eigvecs = eigsh(L, k=num_eigenvectors_to_compute, which='SM', tol=1e-4, 
                                 ncv=min(num_nodes-1, max(2 * num_eigenvectors_to_compute + 1, 20)))
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}. Returning zero PEs.")
        return torch.zeros((num_nodes, k), device=target_device)
    
    actual_k_to_use = min(k, eigvecs.shape[1] - 1)
    if actual_k_to_use <=0: 
        pe = torch.zeros((num_nodes, k), dtype=torch.float)
    else:
        # Take eigenvectors corresponding to smallest k non-zero eigenvalues
        # eigvecs are sorted by eigvals. Skip the first (trivial for connected graph).
        pe = torch.from_numpy(eigvecs[:, 1:1+actual_k_to_use]).float()
        if pe.shape[1] < k:
            padding = torch.zeros((num_nodes, k - pe.shape[1]), dtype=torch.float)
            pe = torch.cat((pe, padding), dim=1)
            
    return pe.to(target_device)


class LaplacianPerturb:
    def __init__(self, alpha_min=1e-4, alpha_max=1e-3):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def sample(self, edge_index, num_nodes):
        """Applies random perturbation to edge weights, returning weights of 1 + noise."""
        current_device = edge_index.device if edge_index.numel() > 0 else device # Fallback to global device
        if edge_index.numel() == 0: 
            return torch.empty((0,), device=current_device)

        alpha = torch.rand(1, device=current_device) * (self.alpha_max - self.alpha_min) + self.alpha_min
        signs = torch.randint(0, 2, (edge_index.size(1),), device=current_device, dtype=torch.float) * 2.0 - 1.0
        return 1.0 + alpha * signs

    def adversarial(self, model_not_used, x, edge_index, current_weights, xi=1e-6, epsilon=0.1, ip=3):
        """Memory-efficient adversarial perturbation on existing edge weights."""
        num_nodes = x.size(0)
        current_device = x.device
        
        if num_nodes == 0 or edge_index.numel() == 0 or current_weights.numel() == 0:
            return current_weights.clone() # Return a clone to avoid modifying input if it's empty

        adj = torch.sparse_coo_tensor(
            edge_index,
            current_weights.float(), 
            (num_nodes, num_nodes),
            device=current_device
        ).coalesce()
        
        perturbed_weights = current_weights.clone() 

        with torch.no_grad():
            v = torch.randn(num_nodes, 1, device=current_device)
            for _ in range(ip):
                v_new = torch.sparse.mm(adj, v)
                v_norm = v_new.norm()
                if v_norm > 1e-9: 
                    v = v_new / v_norm
                else:
                    v = torch.randn(num_nodes, 1, device=current_device) 
            
            v_i = v[edge_index[0]] 
            v_j = v[edge_index[1]] 
            
            edge_specific_perturbation_term = xi * v_i.squeeze(-1) * v_j.squeeze(-1) # Squeeze last dim
            
            perturbed_weights = current_weights + epsilon * edge_specific_perturbation_term
        
        return perturbed_weights.clamp(min=0.01, max=2.0)

class SpectralEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim, pe_dim):
        super().__init__()
        self.pe_dim = pe_dim # Store pe_dim
        self.cheb1 = ChebConv(in_dim + pe_dim, hid_dim, K=4) 
        self.cheb2 = ChebConv(hid_dim, hid_dim, K=4)
        self.mu_net = nn.Linear(hid_dim, lat_dim) # Renamed to avoid conflict with mu variable
        self.logvar_net = nn.Linear(hid_dim, lat_dim) # Renamed

    def forward(self, x, edge_index, lap_pe, edge_weight=None):
        current_device = x.device
        num_nodes = x.size(0)

        # Validate and prepare lap_pe
        if lap_pe is None or lap_pe.size(0) != num_nodes or lap_pe.size(1) != self.pe_dim:
            # print(f"Warning: lap_pe is missing, has incorrect node count, or incorrect dimension. Recreating zeros.")
            # print(f"x.shape: {x.shape}, lap_pe.shape: {lap_pe.shape if lap_pe is not None else 'None'}, expected pe_dim: {self.pe_dim}")
            lap_pe = torch.zeros(num_nodes, self.pe_dim, device=current_device, dtype=x.dtype)
        
        x_combined = torch.cat([x, lap_pe], dim=1)

        edge_index_sl, edge_weight_sl = torch_geometric.utils.add_self_loops(
            edge_index, edge_weight, num_nodes=num_nodes, fill_value=1.0 # fill_value for new self-loop weights
        )
        
        # Handle case of no edges after adding self-loops (e.g. single node graph)
        if edge_index_sl.numel() == 0 and num_nodes > 0:
            # ChebConv might not handle empty edge_index well.
            # For a graph with no edges (even after self-loops, though add_self_loops should add them for num_nodes > 0),
            # the convolution result is often just a transformation of features.
            # Here, we might pass x_combined through a linear layer or return zero embeddings.
            # For simplicity, if ChebConv handles it, let it. Otherwise, this needs specific logic.
            # A robust ChebConv should ideally handle this (e.g., by acting as an MLP).
            # If K=0, ChebConv is an MLP. With K>0 and no edges, behavior might be undefined or error.
            # PyG ChebConv typically expects edges.
            # Fallback: if no edges, pass through linear layers mimicking the structure.
            # This is a simplification; real behavior depends on ChebConv's implementation details for no edges.
             if num_nodes > 0: # If there are nodes but no edges
                # Pass through layers that don't rely on edges
                # This is a placeholder for how one might handle edgeless graphs in GNNs
                h = F.relu(self.cheb1.lin(x_combined)) # Apply the initial linear transformation of ChebConv
                h = F.relu(self.cheb2.lin(h))          # Apply the second
                graph_embed = scatter_mean(h, torch.zeros(num_nodes, dtype=torch.long, device=current_device), dim=0)
                return self.mu_net(graph_embed), self.logvar_net(graph_embed)
             else: # 0 nodes
                # Should ideally return empty tensors of correct latent dim, or handle upstream
                return torch.empty(0, self.mu_net.out_features, device=current_device), \
                       torch.empty(0, self.logvar_net.out_features, device=current_device)


        h = F.relu(self.cheb1(x_combined, edge_index_sl, edge_weight_sl))
        h = F.relu(self.cheb2(h, edge_index_sl, edge_weight_sl))
        
        # Determine batch index for scatter_mean. Assumes single graph if data.batch is not present.
        # If using PyG DataLoader, data.batch would be available.
        batch_idx = torch.zeros(num_nodes, dtype=torch.long, device=current_device)
        graph_embed = scatter_mean(h, batch_idx, dim=0) # Output shape [1, hid_dim] for single graph
        
        return self.mu_net(graph_embed), self.logvar_net(graph_embed)

class ScoreNet(nn.Module):
    def __init__(self, lat_dim, cond_dim, time_embed_dim=32): # Added time_embed_dim
        super().__init__()
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding layer
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(lat_dim + cond_dim + time_embed_dim, 512), nn.ReLU(), # Added time_embed_dim
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, lat_dim)
        )
        self.cond_drop_prob = 0.1

    def forward(self, zt, time_t, cond=None): # Added time_t argument
        # Embed time
        # time_t is expected to be a scalar or [batch_size, 1]
        if not isinstance(time_t, torch.Tensor):
            time_t = torch.tensor([time_t], device=zt.device, dtype=zt.dtype)
        if time_t.ndim == 0: # scalar
            time_t = time_t.unsqueeze(0)
        if time_t.ndim == 1: # [B]
            time_t = time_t.unsqueeze(1) # -> [B,1]

        time_embedding = self.time_mlp(time_t) # [B, time_embed_dim]

        # Classifier-free guidance for condition
        if cond is not None and self.training and torch.rand(1).item() < self.cond_drop_prob:
            cond_input = torch.zeros(zt.size(0), self.cond_dim, device=zt.device, dtype=zt.dtype)
        elif cond is not None:
            if cond.size(0) == 1 and zt.size(0) > 1: 
                cond_input = cond.expand(zt.size(0), -1)
            elif cond.size(0) != zt.size(0):
                # This can happen if zt is [N, D] (node-level) and cond is [1, D] (graph-level)
                # We need to expand cond to match zt's first dimension (N or B)
                if cond.size(0) == 1:
                    cond_input = cond.expand(zt.size(0), -1)
                else:
                    raise ValueError(f"Batch size mismatch for cond ({cond.shape}) and zt ({zt.shape}) that cannot be broadcasted.")
            else: 
                cond_input = cond
        else: 
            cond_input = torch.zeros(zt.size(0), self.cond_dim, device=zt.device, dtype=zt.dtype)

        # Expand time_embedding if zt is [N, D] and time_embedding is [1, T_D] (from graph-level time)
        if time_embedding.size(0) == 1 and zt.size(0) > 1:
            time_embedding = time_embedding.expand(zt.size(0), -1)
        elif time_embedding.size(0) != zt.size(0):
             raise ValueError(f"Batch size mismatch for time_embedding ({time_embedding.shape}) and zt ({zt.shape})")


        combined_input = torch.cat([zt, cond_input, time_embedding], dim=1)
        return self.mlp(combined_input)

class ScoreSDE(nn.Module):
    def __init__(self, score_model, T=1.0, N=1000):
        super().__init__()
        self.score_model = score_model
        self.T = T 
        self.N = N 
        # Ensure timesteps are on the default device (CPU or GPU)
        self.timesteps = torch.linspace(T, T/N, N, device=device) 

    def marginal_std(self, t):
        # Ensure t is a tensor and on the correct device
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device)
        elif t.device != device:
            t = t.to(device)
        return torch.sqrt(1. - torch.exp(-2 * t)) 

    def sample(self, z_shape, cond=None): # z_shape is (num_entities, lat_dim)
        """Ancestral sampling from the SDE."""
        current_device = self.timesteps.device # Should be same as global device
        with torch.no_grad():
            z = torch.randn(z_shape, device=current_device) 
            
            dt = self.T / self.N # float

            for t_val_float in self.timesteps.cpu().numpy(): # Iterate with float values for t
                # t_val_float is a Python float here.
                # Score model expects t_tensor
                t_tensor_for_model = torch.tensor([t_val_float], device=current_device, dtype=z.dtype) # Pass as [1] tensor
                
                sigma_t = self.marginal_std(t_tensor_for_model) # Pass tensor
                
                # score_model(zt, time_t, cond)
                # z is [N, D] or [B, D]. t_tensor_for_model is [1]. cond is [1, D_c] or [B, D_c]
                # ScoreNet's forward will handle broadcasting/expanding t and cond.
                score_val = self.score_model(z, t_tensor_for_model, cond) 
                
                if sigma_t > 1e-6: # Avoid division by zero if sigma_t is very small
                    score_val = score_val / sigma_t
                else:
                    score_val = torch.zeros_like(z) # Or handle as per SDE if sigma_t is zero

                # Convert step_size (dt, which is float) to a tensor for torch.sqrt
                step_size_tensor = torch.tensor(dt, device=current_device, dtype=z.dtype)
                
                # Euler-Maruyama step
                # z = z_prev_t + f(z,t)dt + g(t)dW  (for forward SDE)
                # z_prev = z_curr - [f(z,t) - g(t)^2 score] dt + g(t) dW_bar (for reverse SDE)
                # Simplified for VP-SDE (f approx 0 after score transform, g approx 1 or sqrt(beta))
                # z = z + score_val * effective_step + noise * sqrt(effective_step_noise_variance)
                # Using the previous step form:
                z = z + score_val * step_size_tensor + torch.randn_like(z) * torch.sqrt(step_size_tensor)
            return z

class FeatureDecoder(nn.Module):
    def __init__(self, lat_dim, hid_dim, out_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(lat_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, z):
        return self.decoder(z)

class Trainer:
    def __init__(self, in_dim, hid_dim, lat_dim, cond_dim, pe_dim, timesteps, lr, warmup_steps=1000, total_steps=10000):
        self.encoder = SpectralEncoder(in_dim, hid_dim, lat_dim, pe_dim=pe_dim).to(device)
        self.denoiser = ScoreNet(lat_dim, cond_dim, time_embed_dim=32).to(device) # ScoreNet now takes time_embed_dim
        self.decoder = FeatureDecoder(lat_dim, hid_dim, in_dim).to(device)
        self.diff = ScoreSDE(self.denoiser, T=1.0, N=timesteps).to(device)
        self.lap_pert = LaplacianPerturb()
        
        all_params = list(self.encoder.parameters()) + \
                     list(self.denoiser.parameters()) + \
                     list(self.decoder.parameters())
        self.optim = torch.optim.Adam(all_params, lr=lr)

        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=lambda step: 
                min((step + 1) / warmup_steps, 
                    0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))) 
                if step >= warmup_steps and total_steps > warmup_steps 
                else (step + 1) / warmup_steps if warmup_steps > 0 else 1.0 # Handle total_steps <= warmup_steps
        )
        self.current_step = 0

    def train_epoch(self, loader):
        self.encoder.train()
        self.denoiser.train() 
        self.decoder.train()
        
        total_loss = 0
        for data in loader:
            data = data.to(device)
            
            # Skip if graph is problematic (e.g. no nodes, or features/PE mismatch)
            if data.x is None or data.x.size(0) == 0:
                print("Skipping a graph with 0 nodes or no features in training.")
                continue
            if data.lap_pe is None or data.lap_pe.size(0) != data.x.size(0):
                print(f"Skipping graph due to lap_pe mismatch (x nodes: {data.x.size(0)}, pe nodes: {data.lap_pe.size(0) if data.lap_pe is not None else 'None'}). Re-check dataset processing.")
                # Fallback: create zero PEs if this happens unexpectedly
                # data.lap_pe = torch.zeros(data.x.size(0), self.encoder.pe_dim, device=device, dtype=data.x.dtype)
                continue # Or handle by creating zero PEs if that's acceptable


            lap_pe = data.lap_pe # Already on device from DataLoader or .to(device)

            initial_perturbed_weights = self.lap_pert.sample(data.edge_index, data.num_nodes)
            adversarially_perturbed_weights = self.lap_pert.adversarial(
                self.encoder, data.x, data.edge_index, initial_perturbed_weights
            )
            
            self.optim.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                mu, logvar = self.encoder(data.x, data.edge_index, lap_pe, adversarially_perturbed_weights)
                
                std = torch.exp(0.5 * logvar)
                eps_noise_for_z = torch.randn_like(std)
                z = mu + eps_noise_for_z * std # z is [batch_graph_size, lat_dim], typically [1, lat_dim] per graph from loader

                # --- Diffusion Model Training Loss ---
                # 1. Sample random timesteps t for each item in the batch (each graph's latent z)
                # z is typically [1, lat_dim] if batch_size=1 for graphs, or [B_graphs, lat_dim] if graphs are batched by encoder
                # Assuming z is [num_graphs_in_batch, lat_dim]
                t_indices = torch.randint(0, self.diff.N, (z.size(0),), device=device).long()
                time_values_for_loss = self.diff.timesteps[t_indices] # [num_graphs_in_batch]

                # 2. Corrupt z to z_t (noisy latent)
                # Using a simple Gaussian noise model for corruption: z_t = z + sigma_t * noise
                # A more standard DDPM/SDE corruption: z_t = sqrt(alpha_bar_t) * z + sqrt(1 - alpha_bar_t) * noise
                # Here, sigma_t is self.diff.marginal_std(time_values_for_loss)
                
                # marginal_std expects [B] or [B,1], ensure time_values_for_loss is passed correctly
                sigma_t_batch = self.diff.marginal_std(time_values_for_loss) # [num_graphs_in_batch]
                if sigma_t_batch.ndim == 1:
                    sigma_t_batch = sigma_t_batch.unsqueeze(-1) # -> [num_graphs_in_batch, 1] for broadcasting with z

                noise_target = torch.randn_like(z) # The noise we want the model to predict
                xt_corrupted = z + sigma_t_batch * noise_target # Noisy input z_t

                # 3. Predict noise using the denoiser (ScoreNet)
                # ScoreNet now takes (zt, time_t, cond)
                # time_values_for_loss is [B], ScoreNet expects [B,1] or scalar for time
                eps_predicted = self.denoiser(xt_corrupted, time_values_for_loss, cond=mu)
                
                loss_diff = F.mse_loss(eps_predicted, noise_target)
                
                kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                x_reconstructed = self.decoder(mu) # mu is [num_graphs_in_batch, lat_dim]
                # data.x is [total_nodes_in_batch, feat_dim]. Need to aggregate data.x per graph for comparison.
                # This requires data.batch vector if graphs are batched by PyG.
                # If loader gives one graph at a time, data.x.mean(0, keepdim=True) is fine.
                # For batched graphs:
                if hasattr(data, 'batch') and data.batch is not None:
                    # scatter_mean data.x per graph
                    target_x_mean = scatter_mean(data.x, data.batch, dim=0) # [num_graphs_in_batch, feat_dim]
                else: # Assuming single graph per iteration from loader
                    target_x_mean = data.x.mean(dim=0, keepdim=True) # [1, feat_dim]

                loss_rec = F.mse_loss(x_reconstructed, target_x_mean)
                
                loss = loss_diff * 0.1 + kl_div * 0.05 + loss_rec * 1.0

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
            self.scaler.step(self.optim)
            self.scaler.update() 
            
            # Correct order: optimizer.step() then scheduler.step()
            self.scheduler.step() # Call scheduler AFTER optimizer step
            self.current_step +=1
            
            total_loss += loss.item()

        return total_loss / len(loader) if len(loader) > 0 else 0.0

    @torch.no_grad() # Decorator for evaluation methods
    def evaluate(self, real_list, gen_list):
        def graph_structure_similarity(ei_r, ei_g):
            def to_edge_set(ei):
                if ei is None or ei.numel() == 0: return set()
                # Ensure edge_index is on CPU and is 2xNumEdges
                if ei.dim() != 2 or ei.size(0) != 2:
                    # print(f"Warning: Invalid edge_index shape {ei.shape} for GSS. Returning 0.")
                    return set() # Invalid format
                rows, cols = ei.cpu().numpy()
                return set(tuple(sorted((int(u), int(v)))) for u, v in zip(rows, cols))
            
            S_r = to_edge_set(ei_r)
            S_g = to_edge_set(ei_g)
            
            if not S_r and not S_g: return 1.0 
            union_len = len(S_r | S_g)
            if union_len == 0: return 1.0 

            intersection_len = len(S_r & S_g)
            return intersection_len / union_len

        GSS, FMS, ARI = [], [], []
        for real_graph, gen_graph in zip(real_list, gen_list):
            if real_graph is None or gen_graph is None: continue
            if real_graph.x is None or gen_graph.x is None or real_graph.x.size(0)==0 or gen_graph.x.size(0)==0:
                # print("Warning: Skipping graph pair in evaluation due to missing/empty features.")
                continue
            
            # GSS
            if real_graph.edge_index is not None and gen_graph.edge_index is not None:
                 GSS.append(graph_structure_similarity(real_graph.edge_index, gen_graph.edge_index))
            # else:
                 # print("Warning: Skipping GSS for a graph pair due to missing edge_index.")

            # Feature Mean Squared Error
            # Ensure features are on the same device for comparison
            real_x_mean = real_graph.x.mean(0).to(device)
            gen_x_mean = gen_graph.x.mean(0).to(device)
            FMS.append(F.mse_loss(real_x_mean, gen_x_mean).item())
            
            n_clusters_eval = 5 
            # Ensure k for KMeans is not > number of samples
            k_real = min(n_clusters_eval, real_graph.x.size(0))
            k_gen = min(n_clusters_eval, gen_graph.x.size(0))

            if k_real > 1 and k_gen > 1 : # KMeans needs at least 2 clusters (and samples)
                try:
                    r_lbl = KMeans(n_clusters=k_real, random_state=0, n_init='auto').fit_predict(real_graph.x.cpu().detach().numpy())
                    g_lbl = KMeans(n_clusters=k_gen, random_state=0, n_init='auto').fit_predict(gen_graph.x.cpu().detach().numpy())
                    # ARI needs labels of same length if comparing directly, or handle appropriately.
                    # If k_real != k_gen, direct ARI might not be meaningful unless one expects different cluster counts.
                    # For now, assume we want to compare clustering quality even if k differs slightly due to node count.
                    # A more common setup is to force same k if comparing clusterings.
                    # If lengths of r_lbl and g_lbl differ, adjusted_rand_score will error.
                    # This implies we are evaluating how well the *intrinsic clustering* matches,
                    # not necessarily a fixed number of clusters.
                    # For this to work, we'd typically cluster both to the same K, or K derived from data.
                    # Let's assume we use min(k_real, k_gen) if they differ, or fixed K if both have enough samples.
                    # For simplicity, if K=5 is target, use it if possible, else skip.
                    if real_graph.x.size(0) >= n_clusters_eval and gen_graph.x.size(0) >= n_clusters_eval:
                         r_lbl_fixedk = KMeans(n_clusters=n_clusters_eval, random_state=0, n_init='auto').fit_predict(real_graph.x.cpu().detach().numpy())
                         g_lbl_fixedk = KMeans(n_clusters=n_clusters_eval, random_state=0, n_init='auto').fit_predict(gen_graph.x.cpu().detach().numpy())
                         ARI.append(adjusted_rand_score(r_lbl_fixedk, g_lbl_fixedk))
                    # else:
                        # print(f"Skipping ARI for pair: not enough samples in one/both graphs for {n_clusters_eval} clusters.")

                except Exception as e:
                    # print(f"KMeans/ARI calculation failed: {e}. Skipping ARI for this pair.")
                    pass # Silently skip if clustering fails
            # else:
                # print(f"Skipping ARI: Not enough nodes for clustering (real: {real_graph.x.size(0)}, gen: {gen_graph.x.size(0)}).")


        return {
            'GSS': np.mean(GSS) if GSS else 0.0,
            'FeatMSE': np.mean(FMS) if FMS else 0.0,
            'ARI': np.mean(ARI) if ARI else 0.0,
        }

if __name__ == '__main__':
    # --- Configuration ---
    BATCH_SIZE = 16 # Reduced batch size further for memory safety during debugging
    LEARNING_RATE = 1e-4 # Slightly lower LR
    EPOCHS = 100 
    HIDDEN_DIM = 512 
    LATENT_DIM = 512 
    PE_DIM = 10      
    K_NEIGHBORS = 7  
    TIMESTEPS_DIFFUSION = 500 # Reduced for faster iteration
    WARMUP_STEPS = 1000 # Reduced for faster iteration
    TOTAL_TRAINING_STEPS = 1000 # Reduced for faster iteration
    
    # --- Dynamic Path for Processed Data ---
    # Ensures that if K_NEIGHBORS changes, a new processed file is used/created.
    TRAIN_DATA_ROOT = f'data_train_k{K_NEIGHBORS}'
    TEST_DATA_ROOT = f'data_test_k{K_NEIGHBORS}'


    # --- Load Training Data ---
    try:
        print(f"Attempting to load training dataset from root: {TRAIN_DATA_ROOT}")
        train_dataset = MerfishCellGraphDataset(csv_path='data/merfish_train.csv', k=K_NEIGHBORS, root=TRAIN_DATA_ROOT, train=True)
        if len(train_dataset) == 0:
            print("ERROR: Training dataset is empty after initialization. Check CSV processing or file content.")
            exit()
        num_train_samples = len(train_dataset)
        print(f"Training dataset loaded: {num_train_samples} graphs.")
    except FileNotFoundError:
        print("ERROR: Training data CSV 'data/merfish_train.csv' not found.")
        exit()
    except Exception as e:
        print(f"ERROR: Could not load or process training dataset: {e}")
        import traceback
        traceback.print_exc()
        exit()
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True) # drop_last if batches are small

    # Estimate total steps for scheduler
    TOTAL_TRAINING_STEPS = (num_train_samples // BATCH_SIZE) * EPOCHS if num_train_samples > 0 and BATCH_SIZE > 0 else EPOCHS
    WARMUP_STEPS = TOTAL_TRAINING_STEPS // 10 if TOTAL_TRAINING_STEPS > 10 else max(1, TOTAL_TRAINING_STEPS //2) # e.g., 10%
    if WARMUP_STEPS == 0 and TOTAL_TRAINING_STEPS > 0 : WARMUP_STEPS = 1 # Ensure at least 1 warmup step if training occurs

    # --- Instantiate Trainer ---
    first_data_sample = train_dataset[0]
    if first_data_sample is None or first_data_sample.x is None:
        print("ERROR: Could not get feature dimension from the first training sample.")
        exit()
    input_feature_dim = first_data_sample.x.size(1)

    trainer = Trainer(
        in_dim=input_feature_dim,
        hid_dim=HIDDEN_DIM,
        lat_dim=LATENT_DIM,
        cond_dim=LATENT_DIM, 
        pe_dim=PE_DIM,
        timesteps=TIMESTEPS_DIFFUSION,
        lr=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        total_steps=TOTAL_TRAINING_STEPS 
    )

    # --- Training Loop ---
    print(f"Starting training for {EPOCHS} epochs with {TOTAL_TRAINING_STEPS} total steps (warmup: {WARMUP_STEPS})...")
    for epoch in range(1, EPOCHS + 1):
        avg_loss = trainer.train_epoch(train_loader)
        current_lr = trainer.optim.param_groups[0]["lr"]
        print(f'Epoch {epoch:03d}/{EPOCHS}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}')
        
        if epoch % 10 == 0: # Periodic generation and evaluation (on a small subset)
            print(f"\n--- Intermediate Evaluation at Epoch {epoch} ---")
            trainer.encoder.eval()
            trainer.decoder.eval()
            trainer.diff.score_model.eval() # Set score_model (denoiser) to eval

            eval_real_graphs = [train_dataset[i].to(device) for i in range(min(len(train_dataset), 5))] # Small subset
            eval_generated_graphs = []
            for real_g_template in eval_real_graphs:
                if real_g_template.x is None or real_g_template.x.size(0) == 0: continue
                with torch.no_grad():
                    lap_pe_template = real_g_template.lap_pe.to(device)
                    mu_c, _ = trainer.encoder(real_g_template.x, real_g_template.edge_index, lap_pe_template)
                    z_gen_shape_eval = (real_g_template.num_nodes, LATENT_DIM)
                    z_gen_eval = trainer.diff.sample(z_gen_shape_eval, cond=mu_c)
                    x_gen_eval = trainer.decoder(z_gen_eval)
                    eval_generated_graphs.append(Data(x=x_gen_eval.cpu(), edge_index=real_g_template.edge_index.cpu()))
            
            if eval_real_graphs and eval_generated_graphs and len(eval_real_graphs) == len(eval_generated_graphs):
                eval_metrics = trainer.evaluate(eval_real_graphs, eval_generated_graphs)
                print(f"Intermediate Eval Metrics (Train Subset): {eval_metrics}")
            print("-------------------------------------------\n")


    print("Training completed.")

    # --- Final Generation & Evaluation on Test Set ---
    print("\n--- Final Evaluation on Test Set ---")
    try:
        print(f"Attempting to load test dataset from root: {TEST_DATA_ROOT}")
        test_dataset = MerfishCellGraphDataset(csv_path='data/merfish_test.csv', k=K_NEIGHBORS, root=TEST_DATA_ROOT, train=False)
        if len(test_dataset) == 0:
            print("WARNING: Test dataset is empty. Skipping final evaluation.")
            test_dataset = [] 
        else:
            print(f"Test dataset loaded: {len(test_dataset)} graphs.")
    except FileNotFoundError:
        print("WARNING: Test data CSV 'data/merfish_test.csv' not found. Skipping test set evaluation.")
        test_dataset = [] 
    except Exception as e:
        print(f"ERROR: Could not load or process test dataset: {e}")
        import traceback
        traceback.print_exc()
        test_dataset = []

    if test_dataset and len(test_dataset) > 0:
        trainer.encoder.eval()
        trainer.decoder.eval()
        trainer.diff.score_model.eval() # score_model is self.denoiser

        test_real_graphs = [test_dataset[i].to(device) for i in range(len(test_dataset))]
        generated_graphs_test = []

        for test_g_template in test_real_graphs:
            if test_g_template.x is None or test_g_template.x.size(0) == 0:
                print("Skipping generation for an empty test graph template.")
                # Add a placeholder or ensure evaluation handles potentially fewer generated graphs
                # For now, just skip, which might lead to mismatched list lengths if not handled in evaluate
                continue
            with torch.no_grad():
                lap_pe_test_template = test_g_template.lap_pe.to(device)
                # Basic check for PE, can be more robust
                if lap_pe_test_template.size(0) != test_g_template.x.size(0):
                     lap_pe_test_template = torch.zeros(test_g_template.x.size(0), PE_DIM, device=device, dtype=test_g_template.x.dtype)

                mu_cond_test, _ = trainer.encoder(
                    test_g_template.x, 
                    test_g_template.edge_index, 
                    lap_pe_test_template
                )
                
                z_gen_shape_test = (test_g_template.num_nodes, LATENT_DIM)
                z_generated_test = trainer.diff.sample(z_gen_shape_test, cond=mu_cond_test)
                x_generated_test = trainer.decoder(z_generated_test)
                
                # Create Data object on CPU for evaluation to save GPU memory
                gen_data_test = Data(x=x_generated_test.cpu(), edge_index=test_g_template.edge_index.cpu())
                generated_graphs_test.append(gen_data_test)
        
        # Ensure lists are not empty and have matching lengths before evaluation
        if test_real_graphs and generated_graphs_test and len(test_real_graphs) == len(generated_graphs_test):
            test_eval_results = trainer.evaluate(test_real_graphs, generated_graphs_test) # Pass CPU graphs
            print('Final Evaluation Metrics on Test Set:', test_eval_results)
        elif not test_real_graphs or not generated_graphs_test:
             print("Skipping test set evaluation: Real or generated graph list is empty.")
        else: # Mismatched lengths
            print(f"Skipping test set evaluation: Mismatched number of real ({len(test_real_graphs)}) and generated ({len(generated_graphs_test)}) graphs.")
    else:
        print("Test dataset is empty or could not be loaded. Skipping final evaluation.")

    print("Script finished.")
