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
import torch_geometric.utils
import os
import traceback
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist # For pairwise distances in Wasserstein and MMD
import sys # For checking installed modules
import random
import torch.distributions as dist
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.cell_type_categories = ["Unknown"] # Default
        self.num_cell_types = 1 # Default

        processed_file = f'pbmc3k_{"train" if train else "test"}_k{k_neighbors}_pe{pe_dim}_gt{gene_threshold}_pca{pca_neighbors}.pt'
        self.processed_file_names_list = [processed_file]
        super().__init__(root=root, transform=None, pre_transform=None)

        # Ensure processed data and metadata exist or process them
        metadata_path = self.processed_paths[0].replace(".pt", "_metadata.pt")
        if not os.path.exists(self.processed_paths[0]) or not os.path.exists(metadata_path):
            print(f"Processed file or metadata not found. Dataset will be processed.")
            if os.path.exists(self.processed_paths[0]): os.remove(self.processed_paths[0]) # Clean up if one exists but not other
            if os.path.exists(metadata_path): os.remove(metadata_path)
            self.process()
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(f"Successfully loaded processed data from {self.processed_paths[0]}")
            if os.path.exists(metadata_path):
                metadata = torch.load(metadata_path)
                self.filtered_gene_names = metadata.get('filtered_gene_names', [])
                self.cell_type_categories = metadata.get('cell_type_categories', ["Unknown"])
                self.num_cell_types = metadata.get('num_cell_types', 1)
                print(f"Loaded metadata: {len(self.filtered_gene_names)} filtered gene names, {self.num_cell_types} cell types ({self.cell_type_categories[:5]}...).")
            else:
                print(f"Warning: Metadata file {metadata_path} not found. Attributes might be default or inferred if possible.")
                if self.data and hasattr(self.data, 'x') and self.data.x is not None:
                    if not self.filtered_gene_names and self.data.x.shape[1] > 0 : self.filtered_gene_names = [f"gene_{i}" for i in range(self.data.x.shape[1])]
                    if hasattr(self.data, 'cell_type') and self.data.cell_type is not None:
                        unique_codes = torch.unique(self.data.cell_type).cpu().numpy()
                        self.num_cell_types = len(unique_codes)
                        self.cell_type_categories = [f"Type_{code}" for code in sorted(unique_codes)]
            if self.data is None or self.data.num_nodes == 0:
                 print("Warning: Loaded processed data is empty or has no nodes.")
        
        except Exception as e:
            print(f"Error loading processed data or metadata from {self.processed_paths[0]}: {e}. Attempting to re-process.")
            traceback.print_exc()
            if os.path.exists(self.processed_paths[0]): os.remove(self.processed_paths[0])
            if os.path.exists(metadata_path): os.remove(metadata_path)
            self.process() # Re-process on loading error

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
            metadata = {
                'filtered_gene_names': self.filtered_gene_names,
                'cell_type_categories': self.cell_type_categories,
                'num_cell_types': self.num_cell_types
            }
            metadata_path = self.processed_paths[0].replace(".pt", "_metadata.pt")
            torch.save(metadata, metadata_path)
            print(f"Saved metadata to {metadata_path}")

        except Exception as e:
            print(f"FATAL ERROR during processing or saving data/metadata: {e}"); traceback.print_exc()
            if os.path.exists(self.processed_paths[0]): os.remove(self.processed_paths[0])
            metadata_path_check = self.processed_paths[0].replace(".pt", "_metadata.pt")
            if os.path.exists(metadata_path_check): os.remove(metadata_path_check)
            raise

class LaplacianPerturb:
    def __init__(self, alpha_min=1e-4, alpha_max=1e-3):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def sample(self, edge_index, num_nodes):
        current_device = edge_index.device if edge_index.numel() > 0 else torch.device('cpu')
        if edge_index.numel() == 0: return torch.empty((0,), device=current_device)
        alpha = torch.rand(1, device=current_device, dtype=torch.float32) * (self.alpha_max - self.alpha_min) + self.alpha_min
        signs = torch.randint(0, 2, (edge_index.size(1),), device=current_device, dtype=torch.float32) * 2.0 - 1.0
        return 1.0 + alpha * signs

    def adversarial(self, x, edge_index, current_weights, xi=1e-6, epsilon=0.1, ip=3):
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


# --- New: Graph Transformer Layer ---
class GraphTransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(input_dim)
        # batch_first=True means input (batch, seq, feature)
        self.attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim), # Feed-forward expansion
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * input_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # x: (num_nodes, input_dim)
        # edge_index: (2, num_edges)

        num_nodes = x.size(0)
        if num_nodes == 0:
            return torch.empty(0, x.size(1), device=x.device, dtype=x.dtype)
        
        # Create attention mask
        # MultiheadAttention expects mask where True means MASKED (set to -inf)
        if num_nodes > 1 and edge_index.numel() > 0:
            adj_matrix = torch.zeros((num_nodes, num_nodes), device=x.device, dtype=torch.bool)
            adj_matrix[edge_index[0], edge_index[1]] = True
            adj_matrix[torch.arange(num_nodes), torch.arange(num_nodes)] = True # Allow self-loops
            attn_mask = ~adj_matrix # Invert: True where we want to mask (non-neighbors)
        elif num_nodes > 1 and edge_index.numel() == 0: # Multiple nodes but no edges, mask all but self-loops
            attn_mask = torch.full((num_nodes, num_nodes), True, device=x.device)
            torch.diagonal(attn_mask).fill_(False) # Allow self-attention
        else: # Single node or no nodes, no masking needed
            attn_mask = None

        # Self-attention block
        # MultiheadAttention expects (batch_size, sequence_length, embedding_dim)
        # Here, batch_size=1 (since we process one graph at a time), sequence_length=num_nodes
        x_attn_input = self.norm1(x).unsqueeze(0) # Add batch dimension
        attn_output, _ = self.attn(x_attn_input, x_attn_input, x_attn_input, attn_mask=attn_mask)
        attn_output = attn_output.squeeze(0) # Remove batch dimension
        x = x + self.dropout(attn_output) # Residual connection

        # Feed-forward block
        x_mlp_input = self.norm2(x)
        mlp_output = self.mlp(x_mlp_input)
        x = x + self.dropout(mlp_output) # Residual connection

        return x

# Redesigned ScoreNet to use GraphTransformerLayer
class ScoreNet(nn.Module):
    def __init__(self, in_dim, num_cell_types, pe_dim, time_embed_dim=32, hid_dim_gnn=512, num_transformer_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.num_cell_types = num_cell_types
        self.pe_dim = pe_dim
        self.time_embed_dim = time_embed_dim
        self.hid_dim_gnn = hid_dim_gnn # Store for input dimension check

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.cell_type_embed = nn.Embedding(num_cell_types, time_embed_dim)

        # Initial linear layer to project combined input to hid_dim_gnn
        # Input: noisy_x (in_dim) + time_embed (time_embed_dim) + cell_type_embed (time_embed_dim) + lap_pe (pe_dim)
        self.input_projection = nn.Linear(in_dim + time_embed_dim + time_embed_dim + pe_dim, hid_dim_gnn)

        # Stack GraphTransformerLayers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(hid_dim_gnn, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hid_dim_gnn) # Added for stability
        self.output_layer = nn.Linear(hid_dim_gnn, in_dim) # Output: predicted noise in gene space
        self.cond_drop_prob = 0.1

    def forward(self, x_t, time_t, cell_type_labels, edge_index, lap_pe, edge_weight=None):
        current_device = x_t.device
        num_nodes = x_t.size(0)
        if num_nodes == 0:
            return torch.empty(0, self.in_dim, device=current_device, dtype=x_t.dtype)

        # Time embedding
        if not isinstance(time_t, torch.Tensor):
            time_t_tensor = torch.tensor([time_t], device=current_device, dtype=x_t.dtype)
        else:
            if time_t.device != current_device: time_t = time_t.to(current_device)
            if time_t.dtype != x_t.dtype: time_t = time_t.to(x_t.dtype)
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

        # Cell type embedding with conditional dropout
        if cell_type_labels is not None:
            if cell_type_labels.ndim == 0: cell_type_labels = cell_type_labels.unsqueeze(0)
            if cell_type_labels.size(0) != num_nodes and num_nodes > 0:
                raise ValueError(f"Batch size mismatch for cell_type_labels and x_t")
            is_dropout = self.training and torch.rand(1).item() < self.cond_drop_prob
            if is_dropout:
                cell_type_embedding = torch.zeros(num_nodes, self.time_embed_dim, device=current_device, dtype=x_t.dtype)
            else:
                if cell_type_labels.max() >= self.num_cell_types or cell_type_labels.min() < 0:
                    cell_type_labels_clamped = torch.clamp(cell_type_labels, 0, self.num_cell_types - 1)
                    cell_type_embedding = self.cell_type_embed(cell_type_labels_clamped)
                else:
                    cell_type_embedding = self.cell_type_embed(cell_type_labels)
        else:
            cell_type_embedding = torch.zeros(num_nodes, self.time_embed_dim, device=current_device, dtype=x_t.dtype)

        # Laplacian PE
        if lap_pe is None or lap_pe.size(0) != num_nodes or lap_pe.size(1) != self.pe_dim:
            lap_pe = torch.zeros(num_nodes, self.pe_dim, device=current_device, dtype=x_t.dtype)

        # Combine all inputs for GNN
        combined_input = torch.cat([x_t, time_embedding, cell_type_embedding, lap_pe], dim=1)
        
        # Project to hidden dimension for transformer layers
        h = self.input_projection(combined_input)

        # Pass through GraphTransformerLayers
        for layer in self.transformer_layers:
            h = layer(h, edge_index) # Edge weights are not directly used by MultiheadAttention here

        # Apply final normalization
        h = self.final_norm(h)

        # Predict noise
        predicted_noise = self.output_layer(h)
        return predicted_noise

# ScoreSDE now operates on gene expression data directly
class ScoreSDE(nn.Module):
    def __init__(self, score_model, T=1.0, N=1000):
        super().__init__()
        self.score_model = score_model
        self.T = T
        self.N = N
        # Timesteps are decreasing from T to a small epsilon (e.g., 1e-5)
        self.timesteps = torch.linspace(T, 1e-5, N, device=device, dtype=torch.float32)

    def marginal_std(self, t):
        # This is sigma_t from VP-SDE: sqrt(1 - alpha_bar_t)
        # where alpha_bar_t = exp(-2t) as defined in the original code's marginal_std
        if not isinstance(t, torch.Tensor): t = torch.tensor(t, device=self.timesteps.device, dtype=torch.float32)
        elif t.device != self.timesteps.device: t = t.to(self.timesteps.device)
        if t.dtype != torch.float32 and t.dtype != torch.float64: t = t.float()
        if t.ndim == 0: t = t.unsqueeze(0)
        return torch.sqrt(1. - torch.exp(-2 * t) + 1e-8) # sigma_t

    def alpha_t(self, t):
        # This is sqrt(alpha_bar_t) from VP-SDE
        if not isinstance(t, torch.Tensor): t = torch.tensor(t, device=self.timesteps.device, dtype=torch.float32)
        elif t.device != self.timesteps.device: t = t.to(self.timesteps.device)
        if t.dtype != torch.float32 and t.dtype != torch.float64: t = t.float()
        if t.ndim == 0: t = t.unsqueeze(0)
        return torch.exp(-t) # alpha_t (which is sqrt(alpha_bar_t))

    @torch.no_grad()
    def sample(self, x_shape, cell_type_labels=None, edge_index=None, lap_pe=None):
        current_device = self.timesteps.device
        num_samples, in_dim = x_shape
        if num_samples == 0: return torch.empty(x_shape, device=current_device, dtype=torch.float32)

        # Start from pure noise (standard normal)
        x = torch.randn(x_shape, device=current_device, dtype=torch.float32)
        
        # Ensure graph data is on the correct device for sampling
        if edge_index is not None and edge_index.device != current_device: edge_index = edge_index.to(current_device)
        if lap_pe is not None and lap_pe.device != current_device: lap_pe = lap_pe.to(current_device)

        if cell_type_labels is not None:
            if cell_type_labels.device != current_device: cell_type_labels = cell_type_labels.to(current_device)
            if cell_type_labels.size(0) == 1 and num_samples > 1: cell_type_labels = cell_type_labels.repeat(num_samples)
            elif cell_type_labels.size(0) != num_samples: raise ValueError(f"Cell type condition size mismatch.")
            # Ensure cell_type_labels are long type for embedding lookup
            cell_type_labels = cell_type_labels.long()

        # For sampling, we need to ensure the graph structure is consistent with the number of samples being generated.
        # If edge_index/lap_pe are for a single large graph, and num_samples is for a batch of cells,
        # this needs careful handling. Assuming num_samples == num_nodes in the graph for now.
        if edge_index is None or lap_pe is None:
            print("Warning: Graph structure (edge_index or lap_pe) not provided for sampling. Generating without graph context.")
            # Create dummy graph structure if not provided, to avoid errors in ScoreNet
            temp_edge_index = torch.empty((2,0), dtype=torch.long, device=current_device)
            temp_lap_pe = torch.zeros(num_samples, self.score_model.pe_dim, device=current_device, dtype=x.dtype)
        else:
            temp_edge_index = edge_index
            temp_lap_pe = lap_pe


        for i in range(self.N):
            t_val_float = self.timesteps[i].item()
            t_tensor_for_model = torch.full((num_samples,), t_val_float, device=current_device, dtype=x.dtype)
            
            sigma_t = self.marginal_std(t_val_float) # Current noise level
            alpha_t_val = self.alpha_t(t_val_float) # Current signal level

            # Predict noise using the GNN-based score model
            predicted_epsilon = self.score_model(x, t_tensor_for_model, cell_type_labels,
                                                 temp_edge_index, temp_lap_pe)
            
            # Get next timestep's alpha and sigma
            if i < self.N - 1:
                t_next_val_float = self.timesteps[i+1].item()
                sigma_next = self.marginal_std(t_next_val_float)
                alpha_next = self.alpha_t(t_next_val_float)
            else: # Last step, denoise to x_0
                sigma_next = torch.tensor(0.0, device=current_device, dtype=x.dtype)
                alpha_next = torch.tensor(1.0, device=current_device, dtype=x.dtype)

            # Calculate x_0_prediction based on current noisy x and predicted noise
            # x_t = alpha_t * x_0 + sigma_t * epsilon
            # x_0_pred = (x_t - sigma_t * predicted_epsilon) / alpha_t
            x_0_pred = (x - sigma_t * predicted_epsilon) / (alpha_t_val + 1e-8) # Add epsilon for stability

            # Reconstruct x_prev using x_0_pred and the next timestep's noise level
            # x_prev = alpha_next * x_0_pred + sigma_next * z (where z is new noise)
            x = alpha_next * x_0_pred + sigma_next * torch.randn_like(x)
            
        # Clamp the log1p-transformed data to a reasonable range
        # Assuming log1p values typically range from 0 to around log1p(50000) ~ 10.8
        # A slightly wider range like -2 to 15 might be safer to avoid hard clipping
        x = torch.clamp(x, min=-2.0, max=15.0) 
        return x # Return continuous, log1p-transformed data

class Trainer:
    def __init__(self, in_dim, hid_dim_gnn, pe_dim, num_cell_types, timesteps, lr, warmup_steps, total_steps, input_masking_fraction=0.0, num_transformer_layers=2, num_heads=8, dropout=0.1):
        print("\nInitializing Trainer (Pure GraphDDPM with Transformer backbone)...")
        # ScoreNet is now the main model, taking gene expression, time, cell type, and graph info
        self.denoiser = ScoreNet(in_dim=in_dim, num_cell_types=num_cell_types, pe_dim=pe_dim,
                                 hid_dim_gnn=hid_dim_gnn, num_transformer_layers=num_transformer_layers,
                                 num_heads=num_heads, dropout=dropout).to(device)
        
        self.diff = ScoreSDE(self.denoiser, T=1.0, N=timesteps).to(device)
        self.lap_pert = LaplacianPerturb()
        
        self.all_params = list(self.denoiser.parameters())
        self.optim = torch.optim.Adam(self.all_params, lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.current_step = 0
        self.input_masking_fraction = input_masking_fraction 

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
        self.denoiser.train()
        total_loss_val = 0.0
        num_batches_processed = 0

        for data in loader:
            data = data.to(device)
            num_nodes_in_batch = data.x.size(0)
            if num_nodes_in_batch == 0 or data.x is None or data.x.numel() == 0: continue

            # --- Data Preprocessing for Diffusion (Normalization + Log1p) ---
            # Convert to AnnData for scanpy processing
            # Ensure data.x is numpy array for AnnData conversion
            if isinstance(data.x, torch.Tensor):
                temp_x_np = data.x.cpu().numpy()
            else:
                temp_x_np = data.x # Assume it's already numpy if not tensor

            adata_batch = ad.AnnData(X=temp_x_np)
            sc.pp.normalize_total(adata_batch, target_sum=1e4)
            sc.pp.log1p(adata_batch)
            original_x_processed = torch.from_numpy(adata_batch.X).to(device).float()

            # --- Apply Input Masking (optional, for robustness) ---
            # This masking is applied to the *input* to the diffusion process,
            # not necessarily the target for reconstruction.
            masked_x_input = original_x_processed.clone()
            if self.input_masking_fraction > 0.0 and self.input_masking_fraction < 1.0:
                mask = torch.rand_like(masked_x_input) < self.input_masking_fraction
                masked_x_input[mask] = 0.0

            lap_pe = data.lap_pe
            if lap_pe is None or lap_pe.size(0) != num_nodes_in_batch or lap_pe.size(1) != self.denoiser.pe_dim:
                lap_pe = torch.zeros(num_nodes_in_batch, self.denoiser.pe_dim, device=device, dtype=original_x_processed.dtype)

            cell_type_labels = data.cell_type
            if cell_type_labels is None or cell_type_labels.size(0) != num_nodes_in_batch:
                cell_type_labels = torch.zeros(num_nodes_in_batch, dtype=torch.long, device=device)
            if cell_type_labels.max() >= self.denoiser.num_cell_types or cell_type_labels.min() < 0:
                cell_type_labels = torch.clamp(cell_type_labels, 0, self.denoiser.num_cell_types - 1)

            # --- Laplacian Perturbations ---
            edge_weights = torch.ones(data.edge_index.size(1), device=device, dtype=original_x_processed.dtype) if data.edge_index.numel() > 0 else None
            if edge_weights is not None and edge_weights.numel() > 0:
                initial_perturbed_weights = self.lap_pert.sample(data.edge_index, num_nodes_in_batch)
                # Pass original_x_processed for adversarial perturbation calculation
                adversarially_perturbed_weights = self.lap_pert.adversarial(original_x_processed, data.edge_index, initial_perturbed_weights)
                adversarially_perturbed_weights = torch.nan_to_num(adversarially_perturbed_weights, nan=1.0, posinf=1.0, neginf=0.0)
                adversarially_perturbed_weights = torch.clamp(adversarially_perturbed_weights, min=1e-4)
            else: adversarially_perturbed_weights = None

            self.optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Sample timestep and noise
                t_indices = torch.randint(0, self.diff.N, (num_nodes_in_batch,), device=device).long()
                time_values_for_loss = self.diff.timesteps[t_indices]
                
                noise_target = torch.randn_like(original_x_processed) # Standard normal noise to predict
                
                # Apply noise to original_x_processed (the "x_0" for this batch)
                sigma_t_batch = self.diff.marginal_std(time_values_for_loss).unsqueeze(-1)
                alpha_t_batch = self.diff.alpha_t(time_values_for_loss).unsqueeze(-1)
                
                x_t_noisy = alpha_t_batch * original_x_processed + sigma_t_batch * noise_target

                # Predict noise using the GNN-based denoiser
                eps_predicted = self.denoiser(x_t_noisy, time_values_for_loss, cell_type_labels,
                                              data.edge_index, lap_pe, adversarially_perturbed_weights)
                
                # Diffusion Loss: MSE between predicted noise and actual noise
                loss_diff = F.mse_loss(eps_predicted, noise_target)
                
                final_loss = loss_diff # Only diffusion loss for pure DDPM

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
            if self.current_step % 10 == 0 or num_batches_processed < 5 :
                lr_val = self.optim.param_groups[0]['lr']
                print(f"[DEBUG] Epoch {current_epoch_num} | Batch Step {self.current_step} (Overall) | LR: {lr_val:.3e} | Loss: {final_loss.item():.4f}")

            total_loss_val += final_loss.item()
            num_batches_processed +=1

        if num_batches_processed > 0:
            avg_total_loss = total_loss_val / num_batches_processed
            return avg_total_loss, avg_total_loss, avg_total_loss, avg_total_loss # Return same value for all for consistency
        else:
             print(f"Warning: No batches processed in epoch {current_epoch_num}.")
             return 0.0, 0.0, 0.0, 0.0

    @torch.no_grad()
    def generate(self, num_samples, cell_type_condition=None, edge_index=None, lap_pe=None):
        print(f"\nGenerating {num_samples} samples...")
        self.denoiser.eval()
        
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

        # The shape for generation is directly the gene expression shape
        x_gen_shape = (num_samples, self.denoiser.in_dim)
        
        # Pass graph info to ScoreSDE.sample
        generated_log1p_tensor = self.diff.sample(x_gen_shape,
                                                  cell_type_labels=gen_cell_type_labels_tensor,
                                                  edge_index=edge_index,
                                                  lap_pe=lap_pe)
        
        # Convert log1p back to counts
        generated_counts_tensor = torch.expm1(generated_log1p_tensor)
        generated_counts_tensor = torch.relu(generated_counts_tensor) # Ensure non-negative
        generated_counts_tensor = generated_counts_tensor.round().int().float() # Round to nearest integer counts

        generated_counts_np = generated_counts_tensor.cpu().numpy()
        generated_cell_types_np = gen_cell_type_labels_tensor.cpu().numpy()
        print("Generation complete.")
        return generated_counts_np, generated_cell_types_np

    @torch.no_grad()
    def evaluate_generation(self, real_adata, generated_counts, generated_cell_types, n_pcs=30, mmd_scales=[0.01, 0.1, 1, 10, 100]):
        # This function remains largely the same. Ensure it's robust.
        # Key is that real_adata should be the *filtered* test data.
        print("\n--- Computing Evaluation Metrics ---")
        self.denoiser.eval()

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

def generate_qualitative_plots(real_adata_filtered, generated_counts, generated_cell_types,
                               train_cell_type_categories, train_filtered_gene_names,
                               output_dir="qualitative_plots", umap_neighbors=15, model_name="Our Model"):
    """
    Generates and saves qualitative evaluation plots.
    Args:
        real_adata_filtered (anndata.AnnData): Filtered real AnnData object (test set).
        generated_counts (np.ndarray): Generated gene expression counts.
        generated_cell_types (np.ndarray): Generated cell type labels (integer codes).
        train_cell_type_categories (list): List of cell type names from training data.
        train_filtered_gene_names (list): List of filtered gene names from training data.
        output_dir (str): Directory to save plots.
        umap_neighbors (int): Number of neighbors for UMAP.
        model_name (str): Name of the model for plot titles.
    """
    print(f"\n--- Generating Qualitative Plots in {output_dir} for {model_name} ---")
    os.makedirs(output_dir, exist_ok=True)

    if sp.issparse(real_adata_filtered.X):
        real_counts_np = real_adata_filtered.X.toarray()
    else:
        real_counts_np = np.asarray(real_adata_filtered.X)

    # --- Figure 1a: Mean-Variance Plot ---
    print("Plotting Figure 1a: Mean-Variance Relationship...")
    if real_counts_np.shape[0] > 0 and generated_counts.shape[0] > 0 and \
       real_counts_np.shape[1] > 0 and generated_counts.shape[1] > 0 and \
       real_counts_np.shape[1] == generated_counts.shape[1]:
        real_means = np.mean(real_counts_np, axis=0)
        real_vars = np.var(real_counts_np, axis=0)
        gen_means = np.mean(generated_counts, axis=0)
        gen_vars = np.var(generated_counts, axis=0)
        plt.figure(figsize=(8, 6))
        plt.scatter(real_means, real_vars, alpha=0.5, label='Real Data', s=10, c='blue', edgecolors='none')
        plt.scatter(gen_means, gen_vars, alpha=0.5, label=f'Generated ({model_name})', s=10, c='red', edgecolors='none')
        all_positive_means = np.concatenate([real_means[real_means > 1e-9], gen_means[gen_means > 1e-9]])
        all_positive_vars = np.concatenate([real_vars[real_vars > 1e-9], gen_vars[gen_vars > 1e-9]])
        min_val_plot = 1e-3; max_val_plot = 1.0
        if len(all_positive_means) > 0 and len(all_positive_vars) > 0:
            min_val_data = min(all_positive_means.min(), all_positive_vars.min())
            max_val_data = max(all_positive_means.max(), all_positive_vars.max())
            min_val_plot = max(1e-3, min_val_data); max_val_plot = max(1.0, max_val_data)
        elif len(all_positive_means) > 0 :
             min_val_plot = max(1e-3, all_positive_means.min()); max_val_plot = max(1.0, all_positive_means.max())
        plt.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'k--', alpha=0.7, label='Mean = Variance (Poisson-like)')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Mean Gene Expression (log scale)'); plt.ylabel('Variance Gene Expression (log scale)')
        plt.title('Figure 1a: Gene-wise Mean-Variance Relationship'); plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"figure1a_mean_variance_{model_name.replace(' ', '_')}.png"), dpi=300)
        plt.close(); print("Figure 1a saved.")
    else:
        print("Skipping Figure 1a: Real or generated data is empty, or gene dimensions mismatch.")

    # --- Figure 1b: Sparsity (Zeros per Cell) ---
    print("Plotting Figure 1b: Zeros per Cell Distribution...")
    if real_counts_np.shape[0] > 0 and generated_counts.shape[0] > 0:
        real_zeros_per_cell = (real_counts_np == 0).sum(axis=1)
        gen_zeros_per_cell = (generated_counts == 0).sum(axis=1)
        plt.figure(figsize=(8, 6))
        sns.histplot(real_zeros_per_cell, label='Real Data', stat='density', kde=True, alpha=0.6, bins=50, color='blue', element="step")
        sns.histplot(gen_zeros_per_cell, label=f'Generated ({model_name})', stat='density', kde=True, alpha=0.6, bins=50, color='red', element="step")
        plt.xlabel('Number of Zero Counts per Cell'); plt.ylabel('Density')
        plt.title('Figure 1b: Distribution of Zero Counts per Cell (Sparsity)'); plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"figure1b_zeros_per_cell_{model_name.replace(' ', '_')}.png"), dpi=300)
        plt.close(); print("Figure 1b saved.")
    else:
        print("Skipping Figure 1b: Real or generated data is empty.")

    # --- Figure 2: UMAP Visualization (Separate Plots for Real and Generated) ---
    print("Plotting Figure 2: Separate UMAP Visualizations for Real and Generated Data...")

    # --- Process Real Data for UMAP ---
    if real_counts_np.shape[0] > 0 and real_counts_np.shape[1] > 0:
        adata_real_processed = real_adata_filtered.copy()
        # Ensure var_names are strings to prevent ImplicitModificationWarning
        adata_real_processed.var_names = [str(name) for name in adata_real_processed.var_names] 

        if 'cell_type' in adata_real_processed.obs:
            if isinstance(adata_real_processed.obs['cell_type'].dtype, pd.CategoricalDtype):
                adata_real_processed.obs['cell_type_str_umap'] = adata_real_processed.obs['cell_type']
            elif pd.api.types.is_numeric_dtype(adata_real_processed.obs['cell_type']):
                adata_real_processed.obs['cell_type_str_umap'] = adata_real_processed.obs['cell_type'].apply(
                    lambda x: train_cell_type_categories[int(x)] if 0 <= int(x) < len(train_cell_type_categories) else f"UnknownReal_{int(x)}"
                ).astype('category')
            else:
                adata_real_processed.obs['cell_type_str_umap'] = adata_real_processed.obs['cell_type'].astype('category')
        else:
            adata_real_processed.obs['cell_type_str_umap'] = pd.Categorical(['Unknown_Real'] * adata_real_processed.shape[0])

        # UMAP pipeline for real data
        sc.pp.normalize_total(adata_real_processed, target_sum=1e4)
        sc.pp.log1p(adata_real_processed)
        
        try:
            n_top_genes_hvg_real = min(2000, adata_real_processed.shape[1] - 1 if adata_real_processed.shape[1] > 1 else 1)
            if n_top_genes_hvg_real > 0:
                print("Performing HVG selection for real data with flavor 'cell_ranger'.")
                # FIX: Use 'cell_ranger' flavor for HVG, as it handles non-integer data after normalization/log1p
                sc.pp.highly_variable_genes(adata_real_processed, n_top_genes=n_top_genes_hvg_real, flavor='cell_ranger')
                adata_real_hvg = adata_real_processed[:, adata_real_processed.var.highly_variable].copy()
            else:
                print("Warning: Not enough genes for HVG selection in real data. Using all genes.")
                adata_real_hvg = adata_real_processed.copy()
        except Exception as e_hvg_real:
            print(f"Error selecting HVGs for real data: {e_hvg_real}. Using all genes for PCA/UMAP.")
            adata_real_hvg = adata_real_processed.copy()

        if adata_real_hvg.shape[1] > 0:
            n_comps_pca_real = min(50, adata_real_hvg.shape[0] - 1 if adata_real_hvg.shape[0] > 1 else 50,
                                   adata_real_hvg.shape[1] - 1 if adata_real_hvg.shape[1] > 1 else 50)
            if n_comps_pca_real < 2:
                print(f"Warning: Not enough features/samples for PCA in real data ({n_comps_pca_real} components). Skipping UMAP for real data.")
            else:
                sc.pp.scale(adata_real_hvg, max_value=10)
                sc.tl.pca(adata_real_hvg, svd_solver='arpack', n_comps=n_comps_pca_real)
                current_umap_neighbors_real = min(umap_neighbors, adata_real_hvg.shape[0] - 1 if adata_real_hvg.shape[0] > 1 else umap_neighbors)
                if current_umap_neighbors_real < 2:
                    print(f"Warning: Not enough samples for UMAP neighbors in real data ({current_umap_neighbors_real}). Skipping UMAP for real data.")
                else:
                    sc.pp.neighbors(adata_real_hvg, n_neighbors=current_umap_neighbors_real, use_rep='X_pca')
                    sc.tl.umap(adata_real_hvg, min_dist=0.3)
                    
                    # Plot Real Data UMAP
                    plt.figure(figsize=(8, 8))
                    sc.pl.umap(adata_real_hvg, color='cell_type_str_umap',
                               frameon=False, legend_fontsize=8, legend_loc='on data', show=False,
                               title=f"UMAP of Real Data Cell Types",
                               save=f"_figure2_umap_real_{model_name.replace(' ', '_')}.png")
                    
                    default_save_path_real = f"figures/umap_figure2_umap_real_{model_name.replace(' ', '_')}.png"
                    target_save_path_real = os.path.join(output_dir, f"figure2_umap_real_cell_types_{model_name.replace(' ', '_')}.png")
                    if os.path.exists(default_save_path_real):
                        os.rename(default_save_path_real, target_save_path_real)
                        print(f"Figure 2 (Real Data UMAP) saved to {target_save_path_real}")
                    else:
                        print(f"Warning: Scanpy UMAP plot for real data not found at default location: {default_save_path_real}")
                    plt.close()
        else:
            print("Skipping UMAP for real data: No highly variable genes found or remaining.")
    else:
        print("Skipping UMAP for real data: Real data is empty.")


    # --- Process Generated Data for UMAP ---
    if generated_counts.shape[0] > 0 and generated_counts.shape[1] > 0:
        gen_cell_type_str_list = [
            train_cell_type_categories[int(code)] if 0 <= int(code) < len(train_cell_type_categories) else f"UnknownGen_{int(code)}"
            for code in generated_cell_types
        ]
        
        var_names_for_gen = [str(name) for name in train_filtered_gene_names]
        if not var_names_for_gen and generated_counts.shape[1] > 0:
            print("Warning: No gene names for generated data, creating dummy names for UMAP.")
            var_names_for_gen = [f"Gene_{i}" for i in range(generated_counts.shape[1])]
        elif generated_counts.shape[1] == 0:
            print("Skipping UMAP for generated data: Generated counts have 0 genes.")
            return # Exit plotting for UMAP if no genes

        adata_gen_processed = ad.AnnData(
            X=generated_counts,
            obs=pd.DataFrame({
                'cell_type_str_umap': pd.Categorical(gen_cell_type_str_list)
            }),
            var=pd.DataFrame(index=var_names_for_gen)
        )
        # Ensure var_names are strings to prevent ImplicitModificationWarning
        adata_gen_processed.var_names = [str(name) for name in adata_gen_processed.var_names] 

        # UMAP pipeline for generated data
        sc.pp.normalize_total(adata_gen_processed, target_sum=1e4)
        sc.pp.log1p(adata_gen_processed)

        try:
            n_top_genes_hvg_gen = min(2000, adata_gen_processed.shape[1] - 1 if adata_gen_processed.shape[1] > 1 else 1)
            if n_top_genes_hvg_gen > 0:
                print("Performing HVG selection for generated data with flavor 'cell_ranger'.")
                # FIX: Use 'cell_ranger' flavor for HVG
                sc.pp.highly_variable_genes(adata_gen_processed, n_top_genes=n_top_genes_hvg_gen, flavor='cell_ranger')
                adata_gen_hvg = adata_gen_processed[:, adata_gen_processed.var.highly_variable].copy()
            else:
                print("Warning: Not enough genes for HVG selection in generated data. Using all genes.")
                adata_gen_hvg = adata_gen_processed.copy()
        except Exception as e_hvg_gen:
            print(f"Error selecting HVGs for generated data: {e_hvg_gen}. Using all genes for PCA/UMAP.")
            adata_gen_hvg = adata_gen_processed.copy()

        if adata_gen_hvg.shape[1] > 0:
            n_comps_pca_gen = min(50, adata_gen_hvg.shape[0] - 1 if adata_gen_hvg.shape[0] > 1 else 50,
                                  adata_gen_hvg.shape[1] - 1 if adata_gen_hvg.shape[1] > 1 else 50)
            if n_comps_pca_gen < 2:
                print(f"Warning: Not enough features/samples for PCA in generated data ({n_comps_pca_gen} components). Skipping UMAP for generated data.")
            else:
                sc.pp.scale(adata_gen_hvg, max_value=10)
                sc.tl.pca(adata_gen_hvg, svd_solver='arpack', n_comps=n_comps_pca_gen)
                current_umap_neighbors_gen = min(umap_neighbors, adata_gen_hvg.shape[0] - 1 if adata_gen_hvg.shape[0] > 1 else umap_neighbors)
                if current_umap_neighbors_gen < 2:
                    print(f"Warning: Not enough samples for UMAP neighbors in generated data ({current_umap_neighbors_gen}). Skipping UMAP for generated data.")
                else:
                    sc.pp.neighbors(adata_gen_hvg, n_neighbors=current_umap_neighbors_gen, use_rep='X_pca')
                    sc.tl.umap(adata_gen_hvg, min_dist=0.3)

                    # Plot Generated Data UMAP
                    plt.figure(figsize=(8, 8))
                    sc.pl.umap(adata_gen_hvg, color='cell_type_str_umap',
                               frameon=False, legend_fontsize=8, legend_loc='on data', show=False,
                               title=f"UMAP of Generated ({model_name}) Cell Types",
                               save=f"_figure2_umap_gen_{model_name.replace(' ', '_')}.png")
                    
                    default_save_path_gen = f"figures/umap_figure2_umap_gen_{model_name.replace(' ', '_')}.png"
                    target_save_path_gen = os.path.join(output_dir, f"figure2_umap_generated_cell_types_{model_name.replace(' ', '_')}.png")
                    if os.path.exists(default_save_path_gen):
                        os.rename(default_save_path_gen, target_save_path_gen)
                        print(f"Figure 2 (Generated Data UMAP) saved to {target_save_path_gen}")
                    else:
                        print(f"Warning: Scanpy UMAP plot for generated data not found at default location: {default_save_path_gen}")
                    plt.close()
        else:
            print("Skipping UMAP for generated data: No highly variable genes found or remaining.")
    else:
        print("Skipping UMAP for generated data: Generated data is empty.")

    # Clean up the 'figures' directory if it's empty after moving
    if os.path.exists("./figures") and not os.listdir("./figures"):
        try:
            os.rmdir("./figures")
        except OSError:
            pass # Directory might not be empty if other plots are saved there by scanpy
            
    print(f"Qualitative plotting for {model_name} finished.")


if __name__ == '__main__':
    BATCH_SIZE = 64 # For DataLoader, but PBMC3KDataset is InMemory, so effectively 1 batch of the whole graph
    LEARNING_RATE = 1e-3
    EPOCHS = 1000 # Reduced epochs as a starting point for testing
    HIDDEN_DIM = 1536 # Increased hidden dimension
    PE_DIM = 20
    K_NEIGHBORS = 20
    PCA_NEIGHBORS = 50
    GENE_THRESHOLD = 20
    TIMESTEPS_DIFFUSION = 1000
    GLOBAL_SEED = 69
    set_seed(GLOBAL_SEED)

    INPUT_MASKING_FRACTION = 0.4 # Fraction of input genes to mask during training (0.0 to disable)

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
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
        
        TOTAL_TRAINING_STEPS = len(train_loader) * EPOCHS
        WARMUP_STEPS = max(1, int(0.05 * TOTAL_TRAINING_STEPS))
        WARMUP_STEPS = min(WARMUP_STEPS, TOTAL_TRAINING_STEPS // 2)

        trainer = Trainer(in_dim=input_feature_dim, hid_dim_gnn=HIDDEN_DIM, pe_dim=PE_DIM,
                          num_cell_types=num_cell_types, timesteps=TIMESTEPS_DIFFUSION, lr=LEARNING_RATE,
                          warmup_steps=WARMUP_STEPS, total_steps=TOTAL_TRAINING_STEPS,
                          input_masking_fraction=INPUT_MASKING_FRACTION,
                          num_transformer_layers=6, # Increased number of GraphTransformer layers
                          num_heads=8,
                          dropout=0.1)

        if TOTAL_TRAINING_STEPS > 0:
            print(f"\nStarting training for {EPOCHS} epochs. Total steps: {TOTAL_TRAINING_STEPS}, Warmup: {WARMUP_STEPS}. Initial LR: {LEARNING_RATE:.2e}")
            for epoch in range(1, EPOCHS + 1):
                avg_total_loss, _, _, _ = trainer.train_epoch(train_loader, epoch) # Only total loss is meaningful now
                current_lr = trainer.optim.param_groups[0]["lr"]
                print(f"Epoch {epoch:03d}/{EPOCHS} Summary -> AvgTotal Loss: {avg_total_loss:.4f}, LR: {current_lr:.3e}")
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
            # Check if the denoiser's input_dim matches the test data's gene dim
            if trainer.denoiser.in_dim != num_genes_eval:
                print(f"FATAL ERROR: Denoiser input dim ({trainer.denoiser.in_dim}) != test gene dim ({num_genes_eval}). Skipping eval.")
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

                # For generation, we need to pass the graph structure of the *test* data
                # This assumes the generated data should have a similar graph structure to the real data it's mimicking.
                # If you want to generate entirely novel graphs, this part would need to change.
                test_graph_data = None
                try:
                    # Create a dummy dataset object to get graph info for test_adata
                    temp_test_dataset = PBMC3KDataset(h5ad_path=TEST_H5AD, k_neighbors=K_NEIGHBORS, pe_dim=PE_DIM, root=TEST_DATA_ROOT, train=False, gene_threshold=GENE_THRESHOLD, pca_neighbors=PCA_NEIGHBORS)
                    if temp_test_dataset and len(temp_test_dataset) > 0:
                        test_graph_data = temp_test_dataset.get(0)
                        print(f"Loaded test graph data: {test_graph_data.num_nodes} nodes, {test_graph_data.edge_index.size(1)} edges.")
                    else:
                        print("Warning: Could not load test graph data for generation. Generating without graph context.")
                except Exception as e_graph_load:
                    print(f"Error loading test graph data for generation: {e_graph_load}. Generating without graph context.")

                for i in range(3): # Generate 3 datasets
                    print(f"Generating dataset {i+1}/3...")
                    try:
                        # Pass edge_index and lap_pe from test_graph_data to trainer.generate
                        gen_counts, gen_types = trainer.generate(num_samples=num_test_cells,
                                                                 cell_type_condition=cell_type_condition_for_gen,
                                                                 edge_index=test_graph_data.edge_index if test_graph_data else None,
                                                                 lap_pe=test_graph_data.lap_pe if test_graph_data else None)
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
                if all_wasserstein_results: averaged_metrics["Wasserstein_Average"] = np.nanmean(all_wasserstein_results)
                print(averaged_metrics)
                # --- >>> QUALITATIVE PLOTTING  <<< ---
                output_plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qualitative_evaluation_plots_pure_ddpm")
                current_train_cell_type_categories = ["Unknown"] # Default
                if 'train_dataset' in locals() and hasattr(train_dataset, 'cell_type_categories'):
                    current_train_cell_type_categories = train_dataset.cell_type_categories
                else:
                    print("Warning: 'train_dataset' object not found or 'cell_type_categories' attribute missing. Using default for plots.")
                
                if not filtered_gene_names_from_train:
                    print("Warning: 'filtered_gene_names_from_train' is empty. UMAP gene names might be incorrect.")

                # Use the last generated dataset for plotting
                if generated_datasets_counts and generated_datasets_counts[-1] is not None:
                    generate_qualitative_plots(
                        real_adata_filtered=test_adata, # Pass the filtered real AnnData
                        generated_counts=generated_datasets_counts[-1],
                        generated_cell_types=generated_datasets_cell_types[-1],
                        train_cell_type_categories=current_train_cell_type_categories,
                        train_filtered_gene_names=filtered_gene_names_from_train, # This should be available from training data loading
                        output_dir=output_plot_dir,
                        umap_neighbors=K_NEIGHBORS, # Use your global K_NEIGHBORS
                        model_name="GraphTransformerDDPM" # Updated model name for plots
                    )
                else:
                    print("Skipping qualitative plots: No valid generated data for plotting.")
                # --- END OF QUALITATIVE PLOTTING ---
        else: print("\nSkipping final evaluation: Trainer not initialized.")
    else: print("\nSkipping final evaluation: Test data problematic or filtered genes unavailable.")
    print("\nScript execution finished.")