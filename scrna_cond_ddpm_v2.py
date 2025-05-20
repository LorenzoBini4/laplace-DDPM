import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from torch_geometric.nn import ChebConv
import torch_geometric.utils
import os
import traceback
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist # For pairwise distances in Wasserstein and MMD
import sys # For checking installed modules
import random

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
except ImportError as e:
    print(f"Missing required library: {e}. Please install it using pip.")
    print("Example: pip install torch torch-geometric torch-scatter scanpy anndata pandas numpy scikit-learn scipy POT")
    sys.exit("Required libraries not found.")


torch.backends.cudnn.benchmark = True
# Attempt to set device, fall back to CPU if CUDA is not available or fails
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    print(f"Error setting up CUDA device: {e}. Falling back to CPU.")
    device = torch.device('cpu')

print(f"Using device: {device}")

def set_seed(seed):
    """Set random seeds for reproducibility."""
    if seed is not None:
        print(f"Setting random seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

# Helper function for Laplacian PE
def compute_lap_pe(edge_index, num_nodes, k=10):
    """
    Computes Laplacian Positional Encoding for a graph.
    Args:
        edge_index (torch.Tensor): Graph connectivity in COO format.
        num_nodes (int): Number of nodes in the graph.
        k (int): Number of eigenvectors to compute for the PE.
    Returns:
        torch.Tensor: Laplacian PE of shape (num_nodes, k).
    """
    target_device = edge_index.device if edge_index.numel() > 0 else torch.device('cpu')
    if num_nodes == 0: return torch.zeros((0, k), device=target_device, dtype=torch.float)
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    edge_index_np = edge_index.cpu().numpy()
    if edge_index_np.max() >= num_nodes or edge_index_np.min() < 0:
         print(f"Warning: Edge index out of bounds ({edge_index_np.min()},{edge_index_np.max()}). Num nodes: {num_nodes}. Clamping.")
         edge_index_np = np.clip(edge_index_np, 0, num_nodes - 1)
         valid_edges_mask = edge_index_np[0] != edge_index_np[1]
         edge_index_np = edge_index_np[:, valid_edges_mask]
         if edge_index_np.size == 0:
              print("Warning: All edges removed after clamping. Returning zero PEs.")
              return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    data_np = np.ones(edge_index_np.shape[1])
    try:
        row, col = edge_index_np
        adj = sp.coo_matrix((data_np, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    except Exception as e:
        print(f"Error creating sparse adj matrix for PE: {e}"); return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    adj = adj + adj.T
    adj.data[adj.data > 1] = 1

    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    deg_inv_sqrt_mat = sp.diags(deg_inv_sqrt)
    L = sp.eye(num_nodes, dtype=np.float32) - deg_inv_sqrt_mat @ adj @ deg_inv_sqrt_mat

    num_eigenvectors_to_compute = min(k + 1, num_nodes)
    if num_eigenvectors_to_compute <= 1:
        return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    try:
        eigvals, eigvecs = eigsh(L, k=num_eigenvectors_to_compute, which='SM', tol=1e-4,
                                 ncv=min(num_nodes, max(2 * num_eigenvectors_to_compute + 1, 20)))
        sorted_indices = np.argsort(eigvals)
        eigvecs = eigvecs[:, sorted_indices]
    except Exception as e:
        print(f"Eigenvalue computation failed for PE ({num_nodes} nodes, k={num_eigenvectors_to_compute}): {e}. Returning zero PEs.");
        traceback.print_exc()
        return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    start_idx = 1 if eigvecs.shape[1] > 1 else 0
    actual_k_to_use = min(k, eigvecs.shape[1] - start_idx)

    if actual_k_to_use <= 0:
        pe = torch.zeros((num_nodes, k), dtype=torch.float)
    else:
        pe = torch.from_numpy(eigvecs[:, start_idx : start_idx + actual_k_to_use]).float()
        if pe.shape[1] < k:
            padding = torch.zeros((num_nodes, k - pe.shape[1]), dtype=torch.float)
            pe = torch.cat((pe, padding), dim=1)
    return pe.to(target_device)


class PBMC3KDataset(InMemoryDataset):
    def __init__(self, h5ad_path, k_neighbors=15, pe_dim=10, root='data/pbmc3k', train=True, gene_threshold=20, pca_neighbors=50):
        self.h5ad_path = h5ad_path
        self.k_neighbors = k_neighbors
        self.pe_dim = pe_dim
        self.train = train
        self.gene_threshold = gene_threshold
        self.pca_neighbors = pca_neighbors
        self.filtered_gene_names = []

        processed_file = f'pbmc3k_{"train" if train else "test"}_k{k_neighbors}_pe{pe_dim}_gt{gene_threshold}_pca{pca_neighbors}.pt'
        self.processed_file_names_list = [processed_file]
        super().__init__(root=root, transform=None, pre_transform=None)

        if not os.path.exists(self.processed_paths[0]):
            print(f"Processed file not found at {self.processed_paths[0]}. Dataset will be processed.")
            self.process()
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(f"Successfully loaded processed data from {self.processed_paths[0]}")
            if self.data is None or self.data.num_nodes == 0:
                 print("Warning: Loaded processed data is empty or has no nodes.")
        except Exception as e:
            print(f"Error loading processed data from {self.processed_paths[0]}: {e}. Attempting to re-process.")
            traceback.print_exc()
            if os.path.exists(self.processed_paths[0]):
                 os.remove(self.processed_paths[0])
            self.process()

    @property
    def raw_file_names(self):
        return [os.path.basename(self.h5ad_path)]

    @property
    def processed_file_names(self):
        return self.processed_file_names_list

    def download(self):
        if not os.path.exists(self.h5ad_path):
            print(f"FATAL ERROR: H5AD file not found at {self.h5ad_path}");
            raise FileNotFoundError(f"H5AD file not found: {self.h5ad_path}")
        pass

    def process(self):
        print(f"Processing data from H5AD: {self.h5ad_path} for {'train' if self.train else 'test'} set.")
        print(f"Parameters: k={self.k_neighbors}, PE_dim={self.pe_dim}, Gene Threshold={self.gene_threshold}, PCA Neighbors={self.pca_neighbors}")

        try:
            adata = sc.read_h5ad(self.h5ad_path)
            if not isinstance(adata.X, (np.ndarray, sp.spmatrix)):
                 print(f"Warning: adata.X is of type {type(adata.X)}. Attempting conversion.")
                 try:
                     adata.X = sp.csr_matrix(adata.X)
                 except Exception as e:
                     print(f"Could not convert adata.X: {e}")
                     try:
                         adata.X = np.array(adata.X)
                     except Exception as e_dense:
                         print(f"FATAL ERROR: Could not convert adata.X to sparse or dense: {e_dense}"); raise e_dense
        except FileNotFoundError:
            print(f"FATAL ERROR: H5AD file not found at {self.h5ad_path}"); raise
        except Exception as e:
            print(f"FATAL ERROR reading H5AD file {self.h5ad_path}: {e}"); traceback.print_exc(); raise

        counts = adata.X
        num_cells, initial_num_genes = counts.shape
        print(f"Initial data shape: {num_cells} cells, {initial_num_genes} genes.")

        if num_cells == 0 or initial_num_genes == 0:
             print("Warning: Input data is empty. Creating empty Data object.")
             data = Data(x=torch.empty((0, initial_num_genes), dtype=torch.float),
                         edge_index=torch.empty((2,0), dtype=torch.long),
                         cell_type=torch.empty(0, dtype=torch.long),
                         lap_pe=torch.empty((0, self.pe_dim), dtype=torch.float),
                         num_nodes=0)
             data_list = [data]
             data_to_save, slices_to_save = self.collate(data_list)
             torch.save((data_to_save, slices_to_save), self.processed_paths[0])
             print(f"Processed and saved empty data to {self.processed_paths[0]}")
             return

        print(f"Filtering genes expressed in fewer than {self.gene_threshold} cells.")
        if sp.issparse(counts):
            genes_expressed_count = np.asarray((counts > 0).sum(axis=0)).flatten()
            genes_to_keep_mask = genes_expressed_count >= self.gene_threshold
            counts = counts[:, genes_to_keep_mask]
            self.filtered_gene_names = adata.var_names[genes_to_keep_mask].tolist()
        else:
            genes_expressed_count = np.count_nonzero(counts, axis=0)
            genes_to_keep_mask = genes_expressed_count >= self.gene_threshold
            counts = counts[:, genes_to_keep_mask]
            if hasattr(adata, 'var_names') and adata.var_names is not None and len(adata.var_names) == initial_num_genes:
                 self.filtered_gene_names = [adata.var_names[i] for i, keep in enumerate(genes_to_keep_mask) if keep]
            else:
                 print("Warning: Could not retrieve gene names from adata.var_names for filtered genes.")
                 self.filtered_gene_names = [f'gene_{i}' for i in np.where(genes_to_keep_mask)[0]]

        num_genes_after_filtering = counts.shape[1]
        print(f"Number of genes after filtering: {num_genes_after_filtering}")

        if num_genes_after_filtering == 0:
            print("FATAL ERROR: All genes filtered out. Creating empty Data object.")
            self.filtered_gene_names = []
            data = Data(x=torch.empty((num_cells, 0), dtype=torch.float),
                        edge_index=torch.empty((2,0), dtype=torch.long),
                        cell_type=torch.empty(num_cells, dtype=torch.long),
                        lap_pe=torch.empty((num_cells, self.pe_dim), dtype=torch.float),
                        num_nodes=num_cells)
            data_list = [data]
            data_to_save, slices_to_save = self.collate(data_list)
            torch.save((data_to_save, slices_to_save), self.processed_paths[0])
            print(f"Processed and saved empty data to {self.processed_paths[0]}")
            return

        if 'cell_type' not in adata.obs.columns:
             print("Warning: 'cell_type' not found in adata.obs.columns. Proceeding without cell type labels.")
             cell_type_labels = np.zeros(num_cells, dtype=int)
             num_cell_types = 1
             cell_type_categories = ["Unknown"]
        else:
            cell_type_series = adata.obs['cell_type']
            if not pd.api.types.is_categorical_dtype(cell_type_series):
                 try:
                     cell_type_series = cell_type_series.astype('category')
                 except Exception as e:
                      print(f"Error converting cell_type to categorical: {e}. Using raw values.")
                      try:
                           unique_types, cell_type_labels = np.unique(cell_type_series.values, return_inverse=True)
                           num_cell_types = len(unique_types)
                           cell_type_categories = unique_types.tolist()
                           print(f"Found {num_cell_types} cell types (processed as inverse indices).")
                      except Exception as e_inv:
                           print(f"FATAL ERROR: Could not process cell types as categorical or inverse indices: {e_inv}"); traceback.print_exc(); raise e_inv
            else:
                cell_type_labels = cell_type_series.cat.codes.values
                num_cell_types = len(cell_type_series.cat.categories)
                cell_type_categories = cell_type_series.cat.categories.tolist()
                print(f"Found {num_cell_types} cell types.")

        self.num_cell_types = num_cell_types
        self.cell_type_categories = cell_type_categories

        edge_index = torch.empty((2,0), dtype=torch.long)
        lap_pe = torch.zeros((num_cells, self.pe_dim), dtype=torch.float)

        if num_cells > 1 and self.k_neighbors > 0 and num_genes_after_filtering > 0:
            actual_k_for_knn = min(self.k_neighbors, num_cells - 1)
            print(f"Building KNN graph with k={actual_k_for_knn} on {num_cells} cells based on PCA-reduced expression.")
            pca_input = counts
            if sp.issparse(pca_input):
                try:
                    print("Converting sparse counts to dense for PCA.")
                    pca_input_dense = pca_input.toarray()
                except Exception as e:
                     print(f"Error converting sparse to dense for PCA: {e}. Using raw counts for KNN."); traceback.print_exc()
                     pca_input_dense = counts
            else:
                 pca_input_dense = pca_input

            pca_coords = None
            if num_cells > 1 and pca_input_dense.shape[1] > 0:
                 n_components_pca = min(self.pca_neighbors, pca_input_dense.shape[0] - 1, pca_input_dense.shape[1])
                 if n_components_pca > 0:
                     try:
                         pca = PCA(n_components=n_components_pca, random_state=0)
                         pca_coords = pca.fit_transform(pca_input_dense)
                         print(f"PCA reduced data shape: {pca_coords.shape}")
                     except Exception as e:
                         print(f"Error during PCA for KNN graph: {e}. Using raw counts for KNN."); traceback.print_exc()
                         pca_coords = pca_input_dense
                 else:
                      print("Warning: PCA components <= 0. Using raw counts for KNN.")
                      pca_coords = pca_input_dense
            else:
                 print("Warning: Cannot perform PCA or KNN. Insufficient cells or features.")

            knn_input_coords = pca_coords if pca_coords is not None else pca_input_dense
            if knn_input_coords is not None and knn_input_coords.shape[0] > 1 and actual_k_for_knn > 0 and knn_input_coords.shape[1] > 0:
                try:
                    nbrs = NearestNeighbors(n_neighbors=actual_k_for_knn + 1, algorithm='auto', metric='euclidean').fit(knn_input_coords)
                    distances, indices = nbrs.kneighbors(knn_input_coords)
                    source_nodes = np.repeat(np.arange(num_cells), actual_k_for_knn)
                    target_nodes = indices[:, 1:].flatten()
                    edges = np.stack([source_nodes, target_nodes], axis=0)
                    edge_index = torch.tensor(edges, dtype=torch.long)
                    edge_index = torch_geometric.utils.to_undirected(edge_index, num_nodes=num_cells)
                    edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
                    print(f"KNN graph built with {edge_index.size(1)} edges.")
                except Exception as e:
                    print(f"Error during KNN graph construction: {e}. Creating empty graph."); traceback.print_exc()
                    edge_index = torch.empty((2,0), dtype=torch.long)
            else:
                 print("Warning: Cannot build KNN graph. Creating empty graph.")
                 edge_index = torch.empty((2,0), dtype=torch.long)

            if num_cells > 0 and edge_index.numel() > 0:
                 print(f"Computing Laplacian PE with dim {self.pe_dim} for {num_cells} nodes.")
                 try:
                     lap_pe = compute_lap_pe(edge_index.cpu(), num_cells, k=self.pe_dim).to(device)
                     lap_pe = lap_pe.to(torch.float32)
                 except Exception as e:
                      print(f"Error computing Laplacian PE: {e}. Returning zero PEs."); traceback.print_exc()
                      lap_pe = torch.zeros((num_cells, self.pe_dim), device=device, dtype=torch.float32)
            else:
                 print("Skipping Laplacian PE computation.")
                 lap_pe = torch.zeros((num_cells, self.pe_dim), device=device, dtype=torch.float32)

        if sp.issparse(counts):
            x = torch.from_numpy(counts.toarray().copy()).float()
        else:
            x = torch.from_numpy(counts.copy()).float()
        cell_type = torch.from_numpy(cell_type_labels.copy()).long()
        data = Data(x=x, edge_index=edge_index, lap_pe=lap_pe, cell_type=cell_type, num_nodes=num_cells)

        if data.x.size(0) != num_cells: print(f"Warning: Data.x size mismatch.")
        if data.lap_pe.size(0) != num_cells: print(f"Warning: Data.lap_pe size mismatch.")
        if data.cell_type.size(0) != num_cells: print(f"Warning: Data.cell_type size mismatch.")
        if data.edge_index.numel() > 0 and data.edge_index.max() >= num_cells: print(f"Warning: Edge index out of bounds.")

        data_list = [data]
        try:
            data_to_save, slices_to_save = self.collate(data_list)
            torch.save((data_to_save, slices_to_save), self.processed_paths[0])
            print(f"Processed and saved data to {self.processed_paths[0]}")
        except Exception as e:
            print(f"FATAL ERROR saving processed data to {self.processed_paths[0]}: {e}"); traceback.print_exc()
            if os.path.exists(self.processed_paths[0]): os.remove(self.processed_paths[0])
            raise

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
            time_t_tensor = torch.tensor([time_t], device=current_device, dtype=zt.dtype)
        else:
            if time_t.device != current_device: time_t = time_t.to(current_device)
            if time_t.dtype != zt.dtype: time_t = time_t.to(zt.dtype)
            time_t_tensor = time_t

        if time_t_tensor.ndim == 0:
            if num_nodes > 0: time_t_processed = time_t_tensor.unsqueeze(0).expand(num_nodes, 1)
            else: time_t_processed = time_t_tensor.unsqueeze(0).unsqueeze(1)
        elif time_t_tensor.ndim == 1:
            if time_t_tensor.size(0) != num_nodes: raise ValueError(f"Time tensor 1D shape mismatch")
            time_t_processed = time_t_tensor.unsqueeze(1)
        elif time_t_tensor.ndim == 2 and time_t_tensor.size(1) == 1:
            if time_t_tensor.size(0) != num_nodes: raise ValueError(f"Time tensor 2D shape mismatch")
            time_t_processed = time_t_tensor
        else: raise ValueError(f"Unexpected time_t tensor shape")

        time_embedding = self.time_mlp(time_t_processed)

        if cell_type_labels is not None:
            if cell_type_labels.ndim == 0: cell_type_labels = cell_type_labels.unsqueeze(0)
            if cell_type_labels.size(0) != num_nodes and num_nodes > 0:
                 raise ValueError(f"Batch size mismatch for cell_type_labels and zt")
            is_dropout = self.training and torch.rand(1).item() < self.cond_drop_prob
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
            t_tensor_for_model = torch.full((num_samples,), t_val_float, device=current_device, dtype=z.dtype)
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

    def train_epoch(self, loader, current_epoch_num): # Added current_epoch_num for logging
        self.encoder.train(); self.denoiser.train(); self.decoder.train()
        total_loss_val, total_loss_diff_val, total_loss_kl_val, total_loss_rec_val = 0.0, 0.0, 0.0, 0.0
        num_batches_processed = 0

        for data in loader:
            data = data.to(device)
            num_nodes_in_batch = data.x.size(0)
            if num_nodes_in_batch == 0 or data.x is None or data.x.numel() == 0: continue

            # --- Optional: Input Gene Masking ---
            original_x = data.x # Keep original for reconstruction target
            masked_x = data.x.clone()
            if self.encoder.training and self.input_masking_fraction > 0.0 and self.input_masking_fraction < 1.0:
                # Create a mask for each cell independently
                mask = torch.rand_like(masked_x) < self.input_masking_fraction
                masked_x[mask] = 0.0 # Set masked genes to zero
                # print(f"[DEBUG] Applied input masking. Fraction: {self.input_masking_fraction}, {(mask.sum() / masked_x.numel()):.3f} actual masked elements.")


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
                    # The original code's formula was equivalent but perhaps prone to small negative due to precision.
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

                # --- Reconstruction Loss (use original_x as target) ---
                decoded_log_rates = self.decoder(mu)
                target_counts = original_x.float() # Use original, unmasked counts
                if decoded_log_rates.shape != target_counts.shape:
                     print(f"Warning: Decoder output shape mismatch. Skipping reconstruction loss.")
                     loss_rec = torch.tensor(0.0, device=device)
                elif torch.isnan(decoded_log_rates).any() or torch.isinf(decoded_log_rates).any():
                     print("Warning: Decoder output contains NaN/Inf. Skipping reconstruction loss.")
                     loss_rec = torch.tensor(0.0, device=device)
                else:
                    loss_rec = F.poisson_nll_loss(decoded_log_rates, target_counts, log_input=True, reduction='mean')

                final_loss = (self.loss_weights.get('diff', 1.0) * loss_diff +
                              self.loss_weights.get('kl', 0.1) * kl_div + # Consider KL annealing here
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

            # Debug prints (can be made less frequent)
            if self.current_step % 10 == 0 or num_batches_processed < 5 : # Print more often at start
                lr_val = self.optim.param_groups[0]['lr']
                print(f"[DEBUG] Epoch {current_epoch_num} | Batch Step {self.current_step} (Overall) | Optim Steps (approx): {self.optim._step_count} | Scheduler Steps: {self.scheduler._step_count} | LR: {lr_val:.3e}")
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
            # This print will be the main summary for the epoch
            # print(f"Epoch {current_epoch_num} Averages -> Total Loss: {avg_total_loss:.4f}, Diff Loss: {avg_diff_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, Rec Loss: {avg_rec_loss:.4f}")
            return avg_total_loss, avg_diff_loss, avg_kl_loss, avg_rec_loss # Return all for detailed logging
        else:
             print(f"Warning: No batches processed in epoch {current_epoch_num}.")
             return 0.0, 0.0, 0.0, 0.0

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
        # This function remains largely the same. Ensure it's robust.
        # Key is that real_adata should be the *filtered* test data.
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
                    gamma = 1.0 / (2. * scale**2 + 1e-9) # Add epsilon for scale=0 case
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
                        cond_w_agg.append(np.sqrt(max(0, w2_type))) # Ensure non-negative before sqrt
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


if __name__ == '__main__':
    BATCH_SIZE = 64 # For DataLoader, but PBMC3KDataset is InMemory, so effectively 1 batch of the whole graph
    LEARNING_RATE = 1e-3
    EPOCHS = 1500 # Reduced epochs as a starting point for testing
    HIDDEN_DIM = 1024
    LATENT_DIM = 512
    PE_DIM = 20
    K_NEIGHBORS = 20
    PCA_NEIGHBORS = 50
    GENE_THRESHOLD = 20
    TIMESTEPS_DIFFUSION = 1000
    GLOBAL_SEED = 69
    set_seed(GLOBAL_SEED)

    # Consider annealing KL weight: start small (e.g., 0 or 1e-4) and increase to target over epochs.
    loss_weights = {'diff': 1.0, 'kl': 0.5, 'rec': 10.0} # Adjusted KL weight
    INPUT_MASKING_FRACTION = 0.3 # Fraction of input genes to mask during training (0.0 to disable)

    TRAIN_H5AD = 'data/pbmc3k_train.h5ad'
    TEST_H5AD = 'data/pbmc3k_test.h5ad'
    DATA_ROOT = 'data/pbmc3k_processed'
    os.makedirs(DATA_ROOT, exist_ok=True)
    TRAIN_DATA_ROOT = os.path.join(DATA_ROOT, f'train_k{K_NEIGHBORS}_pe{PE_DIM}_gt{GENE_THRESHOLD}_pca{PCA_NEIGHBORS}_mask{INPUT_MASKING_FRACTION}')
    TEST_DATA_ROOT = os.path.join(DATA_ROOT, f'test_k{K_NEIGHBORS}_pe{PE_DIM}_gt{GENE_THRESHOLD}_pca{PCA_NEIGHBORS}') # Test data processing shouldn't change with masking
    os.makedirs(os.path.join(TRAIN_DATA_ROOT, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_ROOT, 'processed'), exist_ok=True)

    train_dataset = None
    input_feature_dim, num_cell_types = 0, 1
    filtered_gene_names_from_train = []

    try:
        print(f"Loading/Processing training data from: {TRAIN_H5AD} into {TRAIN_DATA_ROOT}")
        train_dataset = PBMC3KDataset(h5ad_path=TRAIN_H5AD, k_neighbors=K_NEIGHBORS, pe_dim=PE_DIM, root=TRAIN_DATA_ROOT, train=True, gene_threshold=GENE_THRESHOLD, pca_neighbors=PCA_NEIGHBORS)
        if train_dataset and len(train_dataset) > 0 and train_dataset.get(0) and train_dataset.get(0).num_nodes > 0:
            num_train_cells = train_dataset.get(0).num_nodes
            input_feature_dim = train_dataset.get(0).x.size(1)
            num_cell_types = train_dataset.num_cell_types
            filtered_gene_names_from_train = train_dataset.filtered_gene_names # Get from dataset object
            print(f"Training data: {num_train_cells} cells, {input_feature_dim} genes, {num_cell_types} types. Filtered genes: {len(filtered_gene_names_from_train)}")
        else: raise ValueError("Training data is empty or invalid after processing.")
    except Exception as e:
        print(f"FATAL ERROR loading training data: {e}"); traceback.print_exc(); sys.exit(1)


    if num_train_cells > 0 and input_feature_dim > 0:
        # Note: For InMemoryDataset, DataLoader typically yields the whole dataset as one batch.
        # If batch_size is set, it's often for compatibility but doesn't create mini-batches unless dataset is a list of Data objects.
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False) # batch_size=1 for clarity with InMemoryDataset
        
        TOTAL_TRAINING_STEPS = len(train_loader) * EPOCHS # len(train_loader) will be 1
        WARMUP_STEPS = max(1, int(0.05 * TOTAL_TRAINING_STEPS)) # 5% warmup
        WARMUP_STEPS = min(WARMUP_STEPS, TOTAL_TRAINING_STEPS // 2)

        trainer = Trainer(in_dim=input_feature_dim, hid_dim=HIDDEN_DIM, lat_dim=LATENT_DIM, num_cell_types=num_cell_types,
                          pe_dim=PE_DIM, timesteps=TIMESTEPS_DIFFUSION, lr=LEARNING_RATE, warmup_steps=WARMUP_STEPS,
                          total_steps=TOTAL_TRAINING_STEPS, loss_weights=loss_weights, input_masking_fraction=INPUT_MASKING_FRACTION)

        if TOTAL_TRAINING_STEPS > 0:
            print(f"\nStarting training for {EPOCHS} epochs. Total steps: {TOTAL_TRAINING_STEPS}, Warmup: {WARMUP_STEPS}. Initial LR: {LEARNING_RATE:.2e}")
            # Implement a simple validation scheme or early stopping here if you have validation data.
            # For now, just training for fixed epochs.
            for epoch in range(1, EPOCHS + 1):
                # Pass epoch number for logging inside train_epoch
                avg_total_loss, avg_diff_loss, avg_kl_loss, avg_rec_loss = trainer.train_epoch(train_loader, epoch)
                current_lr = trainer.optim.param_groups[0]["lr"]
                print(f"Epoch {epoch:03d}/{EPOCHS} Summary -> AvgTotal: {avg_total_loss:.4f}, AvgDiff: {avg_diff_loss:.4f}, AvgKL: {avg_kl_loss:.4f}, AvgRec: {avg_rec_loss:.4f}, LR: {current_lr:.3e}")
                # Add checkpoint saving if needed
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
                    else:
                        _, cell_type_condition_for_gen = np.unique(test_adata.obs['cell_type'].values, return_inverse=True)
                    if cell_type_condition_for_gen is not None: print("Generating conditionally based on real test set cell types.")
                if cell_type_condition_for_gen is None: print("Generating unconditionally for evaluation.")

                for i in range(3): # Generate 3 datasets
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
                                  for scale_key, mmd_val in metrics["MMD"].items(): # Iterate through whatever MMD results are there
                                      # Try to parse scale from key like 'Conditional_Avg_Scale_0.1' or 'Unconditional_Scale_0.1'
                                      try:
                                          scale_val_float = float(scale_key.split('_')[-1])
                                          if scale_val_float in all_mmd_results_per_scale and not np.isnan(mmd_val):
                                              all_mmd_results_per_scale[scale_val_float].append(mmd_val)
                                      except ValueError: pass # Key doesn't end with a float scale
                             if metrics and "Wasserstein" in metrics and metrics["Wasserstein"]:
                                 for w_key, w_val in metrics["Wasserstein"].items(): # e.g. 'Conditional_Avg' or 'Unconditional'
                                     if not np.isnan(w_val): all_wasserstein_results.append(w_val) # Aggregate all valid W-distances
                         except Exception as e_eval_loop: print(f"Error evaluating dataset {i+1}: {e_eval_loop}")
                
                print("\n--- Averaged Evaluation Metrics over Generated Datasets ---")
                averaged_metrics = {"MMD_Averages": {}, "Wasserstein_Average": np.nan}
                for scale, results_list in all_mmd_results_per_scale.items():
                    if results_list: averaged_metrics["MMD_Averages"][f'Avg_Scale_{scale}'] = np.mean(results_list)
                    else: averaged_metrics["MMD_Averages"][f'Avg_Scale_{scale}'] = np.nan
                if all_wasserstein_results: averaged_metrics["Wasserstein_Average"] = np.mean(all_wasserstein_results)
                print(averaged_metrics)
        else: print("\nSkipping final evaluation: Trainer not initialized.")
    else: print("\nSkipping final evaluation: Test data problematic or filtered genes unavailable.")
    print("\nScript execution finished.")

