import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans # Not used in the final evaluation metrics requested
# from sklearn.metrics import adjusted_rand_score # Not used in the final evaluation metrics requested
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

# Helper function for Laplacian PE (kept mostly the same, ensuring robustness)
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
        # If no edges but nodes exist, return zero PE
        return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    # Ensure indices are within bounds
    edge_index_np = edge_index.cpu().numpy()
    if edge_index_np.max() >= num_nodes or edge_index_np.min() < 0:
         print(f"Warning: Edge index out of bounds ({edge_index_np.min()},{edge_index_np.max()}). Num nodes: {num_nodes}. Clamping.")
         edge_index_np = np.clip(edge_index_np, 0, num_nodes - 1)
         # Filter out edges that might have become self-loops or invalid after clamping if num_nodes is very small
         valid_edges_mask = edge_index_np[0] != edge_index_np[1]
         edge_index_np = edge_index_np[:, valid_edges_mask]
         if edge_index_np.size == 0:
              print("Warning: All edges removed after clamping and filtering self-loops. Returning zero PEs.")
              return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)


    data_np = np.ones(edge_index_np.shape[1])
    try:
        # Build adjacency matrix
        row, col = edge_index_np
        adj = sp.coo_matrix((data_np, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    except Exception as e:
        print(f"Error creating sparse adj matrix for PE: {e}"); return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    # Ensure symmetry and remove duplicates
    adj = adj + adj.T
    adj.data[adj.data > 1] = 1 # Binarize if needed


    deg = np.array(adj.sum(axis=1)).flatten()
    # Avoid division by zero for isolated nodes
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    deg_inv_sqrt_mat = sp.diags(deg_inv_sqrt)

    # Compute Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    L = sp.eye(num_nodes, dtype=np.float32) - deg_inv_sqrt_mat @ adj @ deg_inv_sqrt_mat

    # Compute eigenvectors for the smallest non-zero eigenvalues
    num_eigenvectors_to_compute = min(k + 1, num_nodes) # We need k+1 for the first non-trivial eigenvectors
    if num_eigenvectors_to_compute <= 1:
        # Need at least 2 nodes and k>=1 to compute non-trivial LPE
        return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    try:
        # eigsh finds eigenvalues/vectors for symmetric matrices
        # which='SM' means find smallest magnitude eigenvalues
        # k specifies the number of eigenvalues to find
        eigvals, eigvecs = eigsh(L, k=num_eigenvectors_to_compute, which='SM', tol=1e-4,
                                 ncv=min(num_nodes, max(2 * num_eigenvectors_to_compute + 1, 20))) # ncv: number of Lanzcos vectors generated

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigvals)
        eigvecs = eigvecs[:, sorted_indices]
        eigvals = eigvals[sorted_indices]

    except Exception as e:
        print(f"Eigenvalue computation failed for PE ({num_nodes} nodes, k={num_eigenvectors_to_compute}): {e}. Returning zero PEs.");
        traceback.print_exc()
        # Return zeros with correct shape and device
        return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    # Use eigenvectors corresponding to the smallest non-zero eigenvalues (skip the first one for connected graphs)
    # If the graph is disconnected, there will be multiple eigenvalues of 0.
    # We take the k smallest non-zero eigenvalues' corresponding eigenvectors.
    # Simple approach: take the first k eigenvectors after skipping the first one (assuming connected).
    # More robust: find indices of non-zero eigenvalues and take top k.
    # Let's use the simple approach assuming mostly connected components or that the first few capture overall structure.
    start_idx = 1 if eigvecs.shape[1] > 1 else 0 # Skip the first eigenvector if more than one exists
    actual_k_to_use = min(k, eigvecs.shape[1] - start_idx)

    if actual_k_to_use <= 0:
        pe = torch.zeros((num_nodes, k), dtype=torch.float)
    else:
        pe = torch.from_numpy(eigvecs[:, start_idx : start_idx + actual_k_to_use]).float()
        if pe.shape[1] < k:
            # Pad with zeros if we got fewer than k non-trivial eigenvectors
            padding = torch.zeros((num_nodes, k - pe.shape[1]), dtype=torch.float)
            pe = torch.cat((pe, padding), dim=1)

    return pe.to(target_device)


class PBMC3KDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for PBMC3K scRNA-seq data from an .h5ad file.
    Builds a KNN graph on cells based on PCA-reduced expression and computes Laplacian PE.
    Stores raw counts and cell type labels.
    """
    def __init__(self, h5ad_path, k_neighbors=15, pe_dim=10, root='data/pbmc3k', train=True, gene_threshold=20, pca_neighbors=50):
        """
        Args:
            h5ad_path (str): Path to the input .h5ad file.
            k_neighbors (int): Number of neighbors for KNN graph construction.
            pe_dim (int): Dimension of Laplacian Positional Encoding.
            root (str): Root directory for processed data.
            train (bool): Whether this is the training or test dataset split.
            gene_threshold (int): Minimum number of cells a gene must be expressed in to be kept.
            pca_neighbors (int): Number of PCA components to use before KNN graph construction.
        """
        self.h5ad_path = h5ad_path
        self.k_neighbors = k_neighbors
        self.pe_dim = pe_dim
        self.train = train
        self.gene_threshold = gene_threshold
        self.pca_neighbors = pca_neighbors
        self.filtered_gene_names = [] # To store gene names after filtering

        # Determine processed file name based on parameters
        processed_file = f'pbmc3k_{"train" if train else "test"}_k{k_neighbors}_pe{pe_dim}_gt{gene_threshold}_pca{pca_neighbors}.pt'
        # Use a list for processed_file_names property
        self.processed_file_names_list = [processed_file]

        # The 'root' directory is where the 'processed' subdirectory will be created
        super().__init__(root=root, transform=None, pre_transform=None)

        # Check if processing is needed
        if not os.path.exists(self.processed_paths[0]):
            print(f"Processed file not found at {self.processed_paths[0]}. Dataset will be processed.")
            self.process() # Automatically calls process() if file is missing

        # Load the processed data
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(f"Successfully loaded processed data from {self.processed_paths[0]}")
            if self.data is None or self.data.num_nodes == 0:
                 print("Warning: Loaded processed data is empty or has no nodes.")
        except Exception as e:
            print(f"Error loading processed data from {self.processed_paths[0]}: {e}. Attempting to re-process.")
            traceback.print_exc()
            # Clear potentially incomplete files before reprocessing
            if os.path.exists(self.processed_paths[0]):
                 os.remove(self.processed_paths[0])
            self.process() # Re-process on loading error


    @property
    def raw_file_names(self):
        # The raw file is the .h5ad file itself. We assume it's in the root or accessible.
        # We don't need PyG to "download" it from a URL in this case.
        # We just list it here to acknowledge the input file.
        # However, for InMemoryDataset, raw_file_names isn't strictly used for download,
        # but define it for completeness if needed for other functionalities.
        return [os.path.basename(self.h5ad_path)]

    @property
    def processed_file_names(self):
        # Return the list containing the single processed file name defined in __init__
        return self.processed_file_names_list

    def download(self):
        # Assume .h5ad files are already present. If not, raise an error.
        if not os.path.exists(self.h5ad_path):
            print(f"FATAL ERROR: H5AD file not found at {self.h5ad_path}");
            raise FileNotFoundError(f"H5AD file not found: {self.h5ad_path}")
        pass # No external download needed

    def process(self):
        """Processes the raw .h5ad data into a PyG Data object."""
        print(f"Processing data from H5AD: {self.h5ad_path} for {'train' if self.train else 'test'} set.")
        print(f"Parameters: k={self.k_neighbors}, PE_dim={self.pe_dim}, Gene Threshold={self.gene_threshold}, PCA Neighbors={self.pca_neighbors}")

        try:
            adata = sc.read_h5ad(self.h5ad_path)
            # Ensure X is in a standard format (e.g., CSR or numpy)
            if not isinstance(adata.X, (np.ndarray, sp.spmatrix)):
                 print(f"Warning: adata.X is of type {type(adata.X)}. Attempting conversion.")
                 try:
                     # Try converting to sparse CSR first, then dense if needed later
                     adata.X = sp.csr_matrix(adata.X)
                 except Exception as e:
                     print(f"Could not convert adata.X: {e}")
                     # As a last resort, try converting to dense numpy
                     try:
                         adata.X = np.array(adata.X)
                     except Exception as e_dense:
                         print(f"FATAL ERROR: Could not convert adata.X to sparse or dense: {e_dense}"); raise e_dense


        except FileNotFoundError:
            print(f"FATAL ERROR: H5AD file not found at {self.h5ad_path}"); raise
        except Exception as e:
            print(f"FATAL ERROR reading H5AD file {self.h5ad_path}: {e}"); traceback.print_exc(); raise

        counts = adata.X # This should now be numpy array or sparse matrix
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
             return # Exit process method early


        # Gene Filtering based on expression in at least gene_threshold cells
        print(f"Filtering genes expressed in fewer than {self.gene_threshold} cells.")
        if sp.issparse(counts):
            # Count non-zero entries per column (gene) in sparse matrix
            genes_expressed_count = np.asarray((counts > 0).sum(axis=0)).flatten()
            genes_to_keep_mask = genes_expressed_count >= self.gene_threshold
            counts = counts[:, genes_to_keep_mask]
            # Update adata.var to match filtered counts if needed elsewhere, but not strictly required for Data object
            self.filtered_gene_names = adata.var_names[genes_to_keep_mask].tolist()
        else:
            # Count non-zero entries per column (gene) for dense array
            genes_expressed_count = np.count_nonzero(counts, axis=0)
            genes_to_keep_mask = genes_expressed_count >= self.gene_threshold
            counts = counts[:, genes_to_keep_mask]
            # Store filtered gene names
            # Ensure adata.var_names is accessible and matches counts columns before filtering
            if hasattr(adata, 'var_names') and adata.var_names is not None and len(adata.var_names) == initial_num_genes:
                 self.filtered_gene_names = [adata.var_names[i] for i, keep in enumerate(genes_to_keep_mask) if keep]
            else:
                 print("Warning: Could not retrieve gene names from adata.var_names for filtered genes.")
                 self.filtered_gene_names = [f'gene_{i}' for i in np.where(genes_to_keep_mask)[0]] # Create dummy names

        num_genes_after_filtering = counts.shape[1]
        print(f"Number of genes after filtering: {num_genes_after_filtering}")

        if num_genes_after_filtering == 0:
            print("FATAL ERROR: All genes filtered out. Creating empty Data object.")
            self.filtered_gene_names = []
            data = Data(x=torch.empty((num_cells, 0), dtype=torch.float),
                        edge_index=torch.empty((2,0), dtype=torch.long),
                        cell_type=torch.empty(num_cells, dtype=torch.long),
                        lap_pe=torch.empty((num_cells, self.pe_dim), dtype=torch.float), # Still create PE if nodes exist
                        num_nodes=num_cells)
            data_list = [data]
            data_to_save, slices_to_save = self.collate(data_list)
            torch.save((data_to_save, slices_to_save), self.processed_paths[0])
            print(f"Processed and saved empty data to {self.processed_paths[0]}")
            return # Exit process method early


        # Get cell type labels
        if 'cell_type' not in adata.obs.columns:
             print("Warning: 'cell_type' not found in adata.obs.columns. Proceeding without cell type labels.")
             # Create dummy labels if cell_type is missing
             cell_type_labels = np.zeros(num_cells, dtype=int)
             num_cell_types = 1 # Only one dummy type
             cell_type_categories = ["Unknown"]
        else:
            cell_type_series = adata.obs['cell_type']
            # Ensure cell_type is a categorical type for consistent indexing
            if not pd.api.types.is_categorical_dtype(cell_type_series):
                 try:
                     cell_type_series = cell_type_series.astype('category')
                 except Exception as e:
                      print(f"Error converting cell_type to categorical: {e}. Using raw values.")
                      # Fallback to using raw values if conversion fails, try to convert to integer codes directly
                      try:
                           unique_types, cell_type_labels = np.unique(cell_type_series.values, return_inverse=True)
                           num_cell_types = len(unique_types)
                           cell_type_categories = unique_types.tolist()
                           print(f"Found {num_cell_types} cell types (processed as inverse indices).")
                      except Exception as e_inv:
                           print(f"FATAL ERROR: Could not process cell types as categorical or inverse indices: {e_inv}"); traceback.print_exc(); raise e_inv
            else:
                cell_type_labels = cell_type_series.cat.codes.values # Convert categories to integer codes
                num_cell_types = len(cell_type_series.cat.categories)
                cell_type_categories = cell_type_series.cat.categories.tolist()
                print(f"Found {num_cell_types} cell types.")

        # Store the number of cell types and categories for external use (e.g., in Trainer)
        self.num_cell_types = num_cell_types
        self.cell_type_categories = cell_type_categories


        # --- Graph Construction (KNN on PCA-reduced expression) ---
        edge_index = torch.empty((2,0), dtype=torch.long) # Initialize empty edge_index
        lap_pe = torch.zeros((num_cells, self.pe_dim), dtype=torch.float) # Initialize zero PE tensor

        if num_cells > 1 and self.k_neighbors > 0 and num_genes_after_filtering > 0:
            actual_k_for_knn = min(self.k_neighbors, num_cells - 1)
            print(f"Building KNN graph with k={actual_k_for_knn} on {num_cells} cells based on PCA-reduced expression.")

            # Use PCA on expression for KNN
            pca_input = counts # Use filtered counts

            # Ensure pca_input is dense for PCA if necessary (scikit-learn PCA typically requires dense)
            if sp.issparse(pca_input):
                # Convert sparse to dense numpy array
                try:
                    print("Converting sparse counts to dense for PCA. This may require significant memory.")
                    pca_input_dense = pca_input.toarray()
                except Exception as e:
                     print(f"Error converting sparse to dense for PCA: {e}. Skipping PCA and using raw counts for KNN (might be slow)."); traceback.print_exc()
                     pca_input_dense = counts # Fallback, will likely fail KNN if sparse
            else:
                 pca_input_dense = pca_input # Already dense numpy array

            pca_coords = None
            if num_cells > 1 and pca_input_dense.shape[1] > 0: # Ensure valid dimensions for PCA/KNN
                 # Ensure n_components for PCA is valid
                 n_components_pca = min(self.pca_neighbors, pca_input_dense.shape[0] - 1, pca_input_dense.shape[1])

                 if n_components_pca > 0:
                     try:
                         pca = PCA(n_components=n_components_pca, random_state=0)
                         pca_coords = pca.fit_transform(pca_input_dense)
                         print(f"PCA reduced data shape: {pca_coords.shape}")
                     except Exception as e:
                         print(f"Error during PCA for KNN graph: {e}. Using raw counts for KNN (might be slow)."); traceback.print_exc()
                         pca_coords = pca_input_dense # Fallback
                 else:
                      print("Warning: PCA components <= 0. Using raw counts for KNN (might be slow).")
                      pca_coords = pca_input_dense # Fallback
            else:
                 print("Warning: Cannot perform PCA or KNN. Insufficient cells or features. Creating empty graph.")
                 # pca_coords remains None


            # Build KNN graph using scikit-learn on pca_coords or pca_input_dense fallback
            knn_input_coords = pca_coords if pca_coords is not None else pca_input_dense

            if knn_input_coords is not None and knn_input_coords.shape[0] > 1 and actual_k_for_knn > 0 and knn_input_coords.shape[1] > 0:
                try:
                    # Use Euclidean distance for KNN
                    nbrs = NearestNeighbors(n_neighbors=actual_k_for_knn + 1, algorithm='auto', metric='euclidean').fit(knn_input_coords)
                    # Get indices of neighbors, excluding the point itself (distance 0)
                    distances, indices = nbrs.kneighbors(knn_input_coords)

                    # Construct edge_index (source node -> target node) from indices
                    source_nodes = np.repeat(np.arange(num_cells), actual_k_for_knn)
                    target_nodes = indices[:, 1:].flatten() # Exclude the first column (the node itself)

                    edges = np.stack([source_nodes, target_nodes], axis=0)
                    edge_index = torch.tensor(edges, dtype=torch.long)

                    # Ensure the graph is undirected and remove self-loops/duplicates after combining directions
                    edge_index = torch_geometric.utils.to_undirected(edge_index, num_nodes=num_cells)
                    # Remove self-loops explicitly if to_undirected doesn't guarantee it
                    edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)

                    print(f"KNN graph built with {edge_index.size(1)} edges.")

                except Exception as e:
                    print(f"Error during KNN graph construction: {e}. Creating empty graph."); traceback.print_exc()
                    edge_index = torch.empty((2,0), dtype=torch.long)
            else:
                 print("Warning: Cannot build KNN graph. Insufficient samples, k, or feature dimension. Creating empty graph.")
                 edge_index = torch.empty((2,0), dtype=torch.long)


            # Compute Laplacian PE
            if num_cells > 0 and edge_index.numel() > 0:
                 print(f"Computing Laplacian PE with dim {self.pe_dim} for {num_cells} nodes.")
                 try:
                     # Ensure edge_index is on CPU for numpy conversion if compute_lap_pe expects numpy
                     lap_pe = compute_lap_pe(edge_index.cpu(), num_cells, k=self.pe_dim).to(device) # Move PE to device
                     # Ensure PE has correct dtype
                     lap_pe = lap_pe.to(torch.float32)
                 except Exception as e:
                      print(f"Error computing Laplacian PE: {e}. Returning zero PEs."); traceback.print_exc()
                      lap_pe = torch.zeros((num_cells, self.pe_dim), device=device, dtype=torch.float32)
            else:
                 print("Skipping Laplacian PE computation. Number of cells is 0 or edge index is empty.")
                 lap_pe = torch.zeros((num_cells, self.pe_dim), device=device, dtype=torch.float32)


        # Convert counts and labels to tensors
        # Store raw counts as float32 as required for model input
        if sp.issparse(counts):
            counts_dense = counts.toarray()
            x = torch.from_numpy(counts_dense.copy()).float() # Use .copy() to avoid non-writable warning
            x = torch.from_numpy(counts.toarray()).float() # Convert sparse to dense tensor for GNN input
        else:
            x = torch.from_numpy(counts.copy()).float()

        # Convert cell type labels to long tensor
        cell_type = torch.from_numpy(cell_type_labels.copy()).long()

        # Create a single Data object for the entire dataset split
        data = Data(x=x, edge_index=edge_index, lap_pe=lap_pe, cell_type=cell_type, num_nodes=num_cells)

        # Check for potential issues in the created data object
        if data.x.size(0) != num_cells:
             print(f"Warning: Data.x size mismatch. Expected {num_cells}, got {data.x.size(0)}")
        if data.lap_pe.size(0) != num_cells:
             print(f"Warning: Data.lap_pe size mismatch. Expected {num_cells}, got {data.lap_pe.size(0)}")
        if data.cell_type.size(0) != num_cells:
             print(f"Warning: Data.cell_type size mismatch. Expected {num_cells}, got {data.cell_type.size(0)}")
        if data.edge_index.numel() > 0 and data.edge_index.max() >= num_cells:
             print(f"Warning: Edge index out of bounds in created Data object. Max index: {data.edge_index.max()}, Num nodes: {num_cells}")


        data_list = [data] # Wrap in a list for collate

        # Save processed data
        try:
            data_to_save, slices_to_save = self.collate(data_list)
            torch.save((data_to_save, slices_to_save), self.processed_paths[0])
            print(f"Processed and saved data to {self.processed_paths[0]}")
            # After successful save, if train dataset, store filtered gene names in main block
        except Exception as e:
            print(f"FATAL ERROR saving processed data to {self.processed_paths[0]}: {e}"); traceback.print_exc()
            # Clean up potentially incomplete file
            if os.path.exists(self.processed_paths[0]):
                 os.remove(self.processed_paths[0])
            raise # Re-raise the exception after cleaning up

# LaplacianPerturb remains mostly the same, used for perturbing graph edge weights during training
class LaplacianPerturb:
    """
    Perturbs graph edge weights based on Laplacian eigenvectors.
    Intended for adversarial training on graph structure.
    """
    def __init__(self, alpha_min=1e-3, alpha_max=1e-2):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def sample(self, edge_index, num_nodes):
        """Samples a random perturbation scale for each edge."""
        current_device = edge_index.device if edge_index.numel() > 0 else device
        if edge_index.numel() == 0: return torch.empty((0,), device=current_device)
        alpha = torch.rand(1, device=current_device, dtype=torch.float32) * (self.alpha_max - self.alpha_min) + self.alpha_min
        signs = torch.randint(0, 2, (edge_index.size(1),), device=current_device, dtype=torch.float32) * 2.0 - 1.0
        return 1.0 + alpha * signs # Return initial weights (1.0) + perturbation


    def adversarial(self, model_not_used, x, edge_index, current_weights, xi=1e-6, epsilon=0.1, ip=3):
        """Computes adversarial perturbation based on dominant eigenvector of Laplacian."""
        num_nodes = x.size(0)
        current_device = x.device

        if num_nodes == 0 or edge_index.numel() == 0 or current_weights is None or current_weights.numel() == 0:
            return current_weights.clone() if current_weights is not None else None # Return None if no weights or empty

        # Ensure weights are positive for adjacency matrix interpretation
        current_weights = torch.relu(current_weights) + 1e-8 # Add small epsilon for stability

        # Create sparse adjacency matrix from edge_index and weights
        # Ensure indices are within bounds for sparse tensor
        if edge_index.max() >= num_nodes or edge_index.min() < 0:
             print(f"Warning: Edge index out of bounds for adversarial perturbation ({edge_index.min()},{edge_index.max()}). Num nodes: {num_nodes}. Skipping adversarial perturbation.")
             return current_weights.clone() # Return unperturbed weights

        adj = torch.sparse_coo_tensor(edge_index, current_weights.float(), (num_nodes, num_nodes), device=current_device).coalesce()

        perturbed_weights = current_weights.clone()

        with torch.no_grad():
            # Power iteration to find the dominant eigenvector of the adjacency matrix
            # This is an approximation related to the Laplacian's eigenvectors
            # A more correct approach for Laplacian perturbation would involve the Laplacian itself.
            # Sticking to the original code's approach for minimal modification.
            v = torch.randn(num_nodes, 1, device=current_device, dtype=x.dtype) * 0.01
            v = v / (v.norm() + 1e-8) # Normalize, add epsilon for stability

            for _ in range(ip):
                v_new = torch.sparse.mm(adj, v)
                v_norm = v_new.norm()
                if v_norm > 1e-9: v = v_new / v_norm
                else:
                    # Re-initialize if norm is too small (e.g., on disconnected components)
                    v = torch.randn(num_nodes, 1, device=current_device, dtype=x.dtype) * 0.01; v = v / (v.norm() + 1e-8)

            # Compute edge-specific perturbation term
            if edge_index.size(1) == 0:
                edge_specific_perturbation_term = torch.empty(0, device=current_device, dtype=x.dtype)
            else:
                 v_i = v[edge_index[0]] # Shape [num_edges, 1]
                 v_j = v[edge_index[1]] # Shape [num_edges, 1]
                 # The perturbation magnitude is related to the product of eigenvector components at edge endpoints
                 edge_specific_perturbation_term = xi * v_i.squeeze(-1) * v_j.squeeze(-1) # Shape [num_edges]

                 # Ensure perturbation term has the same shape as current_weights
                 if edge_specific_perturbation_term.shape != current_weights.shape:
                     print(f"Warning: Shape mismatch in adversarial perturbation term. Expected {current_weights.shape}, got {edge_specific_perturbation_term.shape}. Skipping adversarial perturbation.")
                     return current_weights.clone() # Return unperturbed weights


            # Apply scaled perturbation
            perturbed_weights = current_weights + epsilon * edge_specific_perturbation_term

        # Clamp to a reasonable range to prevent extreme values
        return perturbed_weights.clamp(min=1e-4, max=10.0)


class SpectralEncoder(nn.Module):
    """
    Spectral Graph Neural Network Encoder.
    Processes node features using graph structure and outputs per-node latent features.
    """
    def __init__(self, in_dim, hid_dim, lat_dim, pe_dim):
        """
        Args:
            in_dim (int): Input feature dimension (number of genes).
            hid_dim (int): Hidden layer dimension.
            lat_dim (int): Output latent dimension per node.
            pe_dim (int): Dimension of Laplacian Positional Encoding.
        """
        super().__init__()
        self.pe_dim = pe_dim
        # ChebConv layers process node features using graph structure and Laplacian spectrum
        # Input dimension is combined feature dimension (gene expression + PE)
        self.cheb1 = ChebConv(in_dim + pe_dim, hid_dim, K=4)
        self.cheb2 = ChebConv(hid_dim, hid_dim, K=4)
        # Output layers predict per-node mu and logvar for the latent space
        self.mu_net = nn.Linear(hid_dim, lat_dim)
        self.logvar_net = nn.Linear(hid_dim, lat_dim)

    def forward(self, x, edge_index, lap_pe, edge_weight=None):
        """
        Args:
            x (torch.Tensor): Node features (gene expression) [num_nodes, in_dim].
            edge_index (torch.Tensor): Graph connectivity [2, num_edges].
            lap_pe (torch.Tensor): Laplacian Positional Encoding [num_nodes, pe_dim].
            edge_weight (torch.Tensor, optional): Edge weights. Defaults to None (assumes binary graph).

        Returns:
            tuple: (mu, logvar) for per-node latent features [num_nodes, lat_dim].
        """
        current_device = x.device
        num_nodes = x.size(0)

        if num_nodes == 0:
             # Return empty tensors with correct latent dimension
             return torch.empty(0, self.mu_net.out_features, device=current_device, dtype=x.dtype), \
                    torch.empty(0, self.logvar_net.out_features, device=current_device, dtype=x.dtype)

        # Ensure lap_pe is valid or use zeros
        if lap_pe is None or lap_pe.size(0) != num_nodes or lap_pe.size(1) != self.pe_dim:
            # print(f"Warning: Invalid lap_pe shape ({lap_pe.shape if lap_pe is not None else 'None'}). Expected ({num_nodes}, {self.pe_dim}). Using zeros.")
            lap_pe = torch.zeros(num_nodes, self.pe_dim, device=current_device, dtype=x.dtype)

        # Concatenate gene features and positional encoding
        x_combined = torch.cat([x, lap_pe], dim=1) # [num_nodes, in_dim + pe_dim]

        # Add self-loops for ChebConv if edge_index is not empty
        # ChebConv requires self-loops implicitly or explicitly depending on implementation details.
        # torch_geometric.utils.add_self_loops adds self-loops with a specified fill_value (default 1.0 for weights)
        edge_index_sl, edge_weight_sl = edge_index, edge_weight # Initialize with original
        if edge_index.numel() > 0:
            edge_index_sl, edge_weight_sl = torch_geometric.utils.add_self_loops(edge_index, edge_weight, num_nodes=num_nodes, fill_value=1.0 if edge_weight is None else 1.0) # Ensure fill_value is float

        elif num_nodes > 0: # Only add self loops if there are nodes but no edges
             # If no edges, create a graph with only self-loops
             edge_index_sl = torch.arange(num_nodes, device=current_device).repeat(2, 1)
             edge_weight_sl = torch.ones(num_nodes, device=current_device, dtype=x.dtype)
        else: # num_nodes == 0
             edge_index_sl = torch.empty((2,0), dtype=torch.long, device=current_device)
             edge_weight_sl = None # No weights needed for empty edges


        # Pass through ChebConv layers
        # Ensure edge_weight_sl is on the correct device and has correct dtype
        if edge_weight_sl is not None:
             if edge_weight_sl.device != current_device:
                  edge_weight_sl = edge_weight_sl.to(current_device)
             if edge_weight_sl.dtype != x_combined.dtype:
                  # print(f"Warning: edge_weight_sl dtype mismatch ({edge_weight_sl.dtype}) with input features ({x_combined.dtype}). Converting.")
                  edge_weight_sl = edge_weight_sl.to(x_combined.dtype)


        if edge_index_sl.numel() == 0 and num_nodes > 0:
             # Case with nodes but no edges after processing (shouldn't happen if num_nodes>0)
             # If this happens, ChebConv might require a dummy edge_index and weight
             # However, if num_nodes > 0, add_self_loops should add edges.
             # This branch might indicate an issue in previous steps or a graph with only isolated nodes.
             # In theory, ChebConv should handle edge_index=empty if num_nodes>0 by just applying linear transformations.
             # Let's pass the (potentially empty but non-None) edge_index_sl and edge_weight_sl
             # If num_nodes > 0, edge_index_sl should at least contain self-loops.
             # If num_nodes == 0, the initial check handles it.
             h = F.relu(self.cheb1(x_combined, edge_index_sl, edge_weight_sl))
             h = F.relu(self.cheb2(h, edge_index_sl, edge_weight_sl))

        else:
             # Normal ChebConv application
             h = F.relu(self.cheb1(x_combined, edge_index_sl, edge_weight_sl))
             h = F.relu(self.cheb2(h, edge_index_sl, edge_weight_sl))

        # Output per-node latent parameters
        mu = self.mu_net(h) # [num_nodes, lat_dim]
        logvar = self.logvar_net(h) # [num_nodes, lat_dim]

        return mu, logvar # mu, logvar are now per-node latent representations


class ScoreNet(nn.Module):
    """
    Score/Noise Prediction Network for the Diffusion Model.
    Operates on per-node latent features and incorporates time and cell type conditioning.
    """
    def __init__(self, lat_dim, num_cell_types, time_embed_dim=32):
        """
        Args:
            lat_dim (int): Latent feature dimension per node.
            num_cell_types (int): Number of cell types for conditioning.
            time_embed_dim (int): Dimension for time embeddings.
        """
        super().__init__()
        self.lat_dim = lat_dim
        self.num_cell_types = num_cell_types
        self.time_embed_dim = time_embed_dim

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Cell type embedding layer
        # Using an embedding layer allows the model to learn a distinct vector for each cell type
        # Add a special embedding for unconditional generation if using classifier-free guidance
        # The embedding layer size should be num_cell_types + 1 if using a dedicated unconditional token (index num_cell_types)
        # For simplicity, let's use a zero vector for unconditional case in the forward pass,
        # so the embedding layer size is just num_cell_types.
        self.cell_type_embed = nn.Embedding(num_cell_types, time_embed_dim)


        # MLP for score/noise prediction
        # Input: latent feature (lat_dim) + time embedding (time_embed_dim) + cell type embedding (time_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(lat_dim + time_embed_dim + time_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, lat_dim) # Predict noise/score for the latent feature
        )
        self.cond_drop_prob = 0.1 # For classifier-free guidance dropout during training

    def forward(self, zt, time_t, cell_type_labels=None):
        """
        Args:
            zt (torch.Tensor): Noisy latent features [num_nodes, lat_dim].
            time_t (torch.Tensor or float): Time value(s) [num_nodes] or scalar.
            cell_type_labels (torch.Tensor, optional): Cell type labels [num_nodes] (long tensor).

        Returns:
            torch.Tensor: Predicted noise/score [num_nodes, lat_dim].
        """
        current_device = zt.device
        num_nodes = zt.size(0)

        if num_nodes == 0:
            return torch.empty(0, self.lat_dim, device=current_device, dtype=zt.dtype)

        # Process time to ensure shape is [num_nodes, 1] before feeding to time_mlp
        if not isinstance(time_t, torch.Tensor):
            # Convert scalar time to tensor on the correct device with correct dtype
            time_t_tensor = torch.tensor([time_t], device=current_device, dtype=zt.dtype) # Shape [1]
        else:
            # Ensure time_t tensor is on the correct device and dtype
            if time_t.device != current_device: time_t = time_t.to(current_device)
            if time_t.dtype != zt.dtype: time_t = time_t.to(zt.dtype)
            time_t_tensor = time_t # Shape could be [], [1], [num_nodes], [num_nodes, 1]

        # Ensure the time tensor has shape [num_nodes, 1]
        if time_t_tensor.ndim == 0:
            # Scalar case (e.g., when time_t is passed as a single float)
            # Expand scalar to match number of nodes and add feature dimension
            if num_nodes > 0:
                 time_t_processed = time_t_tensor.unsqueeze(0).expand(num_nodes, 1) # Scalar -> [1] -> [1, 1] -> [num_nodes, 1]
            else:
                 time_t_processed = time_t_tensor.unsqueeze(0).unsqueeze(1) # Scalar -> [1, 1] (for num_nodes=0 case, although handled above)

        elif time_t_tensor.ndim == 1:
            # 1D tensor case (e.g., [num_nodes])
            # Ensure size matches num_nodes and add feature dimension
            if time_t_tensor.size(0) != num_nodes:
                 # This should ideally not happen if time_t is [num_nodes]
                 raise ValueError(f"Time tensor 1D shape mismatch: {time_t_tensor.shape} vs num_nodes {num_nodes}")
            time_t_processed = time_t_tensor.unsqueeze(1) # [num_nodes] -> [num_nodes, 1]

        elif time_t_tensor.ndim == 2 and time_t_tensor.size(1) == 1:
            # Already in the desired [num_nodes, 1] format
            if time_t_tensor.size(0) != num_nodes:
                 raise ValueError(f"Time tensor 2D shape mismatch: {time_t_tensor.shape} vs num_nodes {num_nodes}")
            time_t_processed = time_t_tensor # Keep as is

        else:
             # Unexpected shape
             raise ValueError(f"Unexpected time_t tensor shape: {time_t_tensor.shape}. Expected scalar, [N], or [N, 1].")


        time_embedding = self.time_mlp(time_t_processed) # [num_nodes, time_embed_dim]

        # Process cell type condition with dropout for classifier-free guidance
        if cell_type_labels is not None:
            if cell_type_labels.ndim == 0: cell_type_labels = cell_type_labels.unsqueeze(0) # Scalar to [1]
            if cell_type_labels.size(0) != num_nodes and num_nodes > 0:
                 raise ValueError(f"Batch size mismatch for cell_type_labels ({cell_type_labels.shape}) and zt ({zt.shape})")

            # Apply classifier-free guidance dropout during training
            # During inference, can use interpolation between conditional and unconditional outputs
            is_dropout = self.training and torch.rand(1).item() < self.cond_drop_prob

            if is_dropout:
                # Use a learned embedding for the "unconditional" class or zeros
                # Using zeros as a simple way to represent unconditional
                cell_type_embedding = torch.zeros(num_nodes, self.time_embed_dim, device=current_device, dtype=zt.dtype)
            else:
                 # Ensure cell_type_labels are within the valid range for the embedding layer
                 if cell_type_labels.max() >= self.num_cell_types or cell_type_labels.min() < 0:
                     # print(f"Warning: Cell type label out of bounds ({cell_type_labels.min()},{cell_type_labels.max()}). Num cell types: {self.num_cell_types}. Clamping labels.")
                     # Clamp labels to valid range to avoid embedding layer errors
                     cell_type_labels_clamped = torch.clamp(cell_type_labels, 0, self.num_cell_types - 1)
                     cell_type_embedding = self.cell_type_embed(cell_type_labels_clamped) # [num_nodes, time_embed_dim]
                 else:
                    cell_type_embedding = self.cell_type_embed(cell_type_labels) # [num_nodes, time_embed_dim]

        else:
            # Unconditional case (no cell type provided)
            cell_type_embedding = torch.zeros(num_nodes, self.time_embed_dim, device=current_device, dtype=zt.dtype) # Use zeros for unconditional

        # Concatenate latent feature, time embedding, and cell type embedding
        combined_input = torch.cat([zt, time_embedding, cell_type_embedding], dim=1) # [num_nodes, lat_dim + 2*time_embed_dim]

        # Pass through MLP to predict noise/score
        score_val = self.mlp(combined_input) # [num_nodes, lat_dim]

        return score_val

class ScoreSDE(nn.Module):
    """
    Score-based Diffusion Model SDE Solver.
    Handles the forward diffusion process marginal standard deviation and the reverse sampling process.
    Operates on per-node latent features.
    """
    def __init__(self, score_model, T=1.0, N=1000):
        """
        Args:
            score_model (nn.Module): The ScoreNet model to use.
            T (float): Final time of the diffusion process.
            N (int): Number of diffusion steps.
        """
        super().__init__()
        self.score_model = score_model # This is the denoiser (ScoreNet)
        self.T = T
        self.N = N
        # Diffusion timesteps (from T down to a small value near 0)
        # Ensure timesteps are float32 and on the correct device
        # The SDE is typically solved from T to epsilon > 0.
        # Using linspace from T to T/N is one convention. Another is T down to 1/N or 0.
        # Let's stick to the original code's linspace for consistency with its marginal_std.
        self.timesteps = torch.linspace(T, T / N, N, device=device, dtype=torch.float32)


    def marginal_std(self, t):
        """
        Computes the standard deviation of the marginal distribution p_t(x_t | x_0).
        Assumes the SDE d x = f(x,t) dt + g(t) dw has a form where the marginal
        is Gaussian with time-dependent mean and variance.
        The original code used sqrt(1. - torch.exp(-2 * t)). Let's keep this formula.
        t is time_values_for_loss (from self.timesteps) or t_tensor_for_model.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.timesteps.device, dtype=torch.float32)
        elif t.device != self.timesteps.device:
            t = t.to(self.timesteps.device)

        if t.dtype != torch.float32 and t.dtype != torch.float64:
             # print(f"Warning: marginal_std input time_t has dtype {t.dtype}. Converting to float32.")
             t = t.float()

        # Ensure t is at least 1D for element-wise operations
        if t.ndim == 0: t = t.unsqueeze(0)

        # Formula for standard deviation - adding a small epsilon for numerical stability with sqrt.
        return torch.sqrt(1. - torch.exp(-2 * t) + 1e-8)


    @torch.no_grad()
    def sample(self, z_shape, cell_type_labels=None):
        """
        Samples latent features by solving the reverse SDE/ODE using Euler-Maruyama.

        Args:
            z_shape (tuple): Shape of the latent features to sample (num_samples, lat_dim).
            cell_type_labels (torch.Tensor, optional): Cell type labels for conditional sampling [num_samples].

        Returns:
            torch.Tensor: Sampled latent features [num_samples, lat_dim].
        """
        current_device = self.timesteps.device
        num_samples = z_shape[0]
        lat_dim = z_shape[1]

        if num_samples == 0:
             return torch.empty(z_shape, device=current_device, dtype=torch.float32)

        # Start from random noise at the final time T
        z = torch.randn(z_shape, device=current_device, dtype=torch.float32) # z is [num_samples, lat_dim]

        dt = self.T / self.N # Time step size

        # Ensure cell_type_labels are on the correct device if provided
        if cell_type_labels is not None:
            if cell_type_labels.device != current_device:
                cell_type_labels = cell_type_labels.to(current_device)
            # Expand cell_type_labels if a single label is provided for multiple samples
            if cell_type_labels.size(0) == 1 and num_samples > 1:
                 cell_type_labels = cell_type_labels.repeat(num_samples)
            elif cell_type_labels.size(0) != num_samples:
                 raise ValueError(f"Number of cell type labels ({cell_type_labels.size(0)}) must match number of samples to generate ({num_samples}) or be 1.")


        # Iterate backwards through time steps
        for i in range(self.N):
            # Time value for the current step (from T down to T/N)
            t_val_float = self.timesteps[i].item() # Scalar float time value
            # Time value tensor for the model (ScoreNet) - needs to be size [num_samples]
            t_tensor_for_model = torch.full((num_samples,), t_val_float, device=current_device, dtype=z.dtype)

            # Calculate the marginal standard deviation at this time step
            sigma_t = self.marginal_std(t_val_float) # Scalar sigma_t

            # Get the predicted noise/score from the model at (z, t, condition)
            # The ScoreNet predicts epsilon (noise) in the DDPM formulation
            predicted_epsilon = self.score_model(z, t_tensor_for_model, cell_type_labels) # [num_samples, lat_dim]

            # SDE update using Euler-Maruyama
            # Based on the reverse SDE for VP-SDEs, where the model predicts epsilon:
            # dx = [ -beta(t)/2 * x - beta(t) * (x - exp(-integral beta)) / (1 - exp(-2 integral beta)) ] dt + sqrt(beta(t)) dw
            # If beta(t) = 2t, integral beta = t^2, exp(-integral beta) = exp(-t^2). Does not match original marginal_std.
            # If marginal_std = sqrt(1 - exp(-2t)), this corresponds to beta(t) = 2.
            # d x = -x dt + sqrt(2) dw. Reverse SDE: dx = [x - 2 * score] dt + sqrt(2) dw
            # Score = -epsilon / sqrt(1-exp(-2t)) = -epsilon / sigma_t
            # dx = [x + 2 * epsilon / sigma_t] dt + sqrt(2) dw

            # Let's follow the update form from the original code, assuming score_val is related to the score.
            # Original update: z = z + score_val * dt + randn * sqrt(dt)
            # This looks like dx = score * dt + dw, which is the VE SDE with constant diffusion coefficient 1, and score = score_val.
            # However, the marginal_std sqrt(1 - exp(-2t)) implies a VP SDE.
            # The original code's loss (MSE(eps_predicted, noise_target)) trains the model to predict epsilon.
            # The original code's sample update is inconsistent with predicting epsilon for a VP SDE.

            # Let's implement sampling consistent with predicting epsilon for a VP SDE where marginal_std = sqrt(1 - exp(-2t))
            # The reverse SDE for x_t when predicting epsilon is:
            # dx = [ -1/2 * beta(t) * x_t + beta(t) * (x_t - exp(-t) * x_0) / (1 - exp(-2t)) ] dt + sqrt(beta(t)) dw
            # Where epsilon_predicted is the prediction of (x_t - exp(-t) * x_0) / sqrt(1 - exp(-2t))
            # Rearranging the reverse SDE term involving epsilon:
            # beta(t) * (x_t - exp(-t) * x_0) / (1 - exp(-2t)) = beta(t) * epsilon_predicted / sqrt(1 - exp(-2t)) = beta(t) * epsilon_predicted / sigma_t
            # With beta(t) = 2 (from marginal_std derivation), this term is 2 * epsilon_predicted / sigma_t.
            # The term -1/2 * beta(t) * x_t is -1 * x_t.
            # So reverse SDE: dx = [-x_t + 2 * epsilon_predicted / sigma_t] dt + sqrt(2) dw.
            # Euler-Maruyama update: z_new = z + [-z + 2 * predicted_epsilon / sigma_t] * dt + sqrt(2 * dt) * randn
            # Simplified update assuming dt is small: z_new = z + 2 * predicted_epsilon / sigma_t * dt + sqrt(2 * dt) * randn

            # Let's use a common sampler for VP SDEs where the model predicts epsilon:
            # z_new = z - sigma_t^2 * score(z, t) * dt + sigma_t * sqrt(dt) * dw
            # Score = -epsilon / sigma_t
            # z_new = z - sigma_t^2 * (-predicted_epsilon / sigma_t) * dt + sigma_t * sqrt(dt) * dw
            # z_new = z + sigma_t * predicted_epsilon * dt + sigma_t * sqrt(dt) * dw
            # Noise term coefficient: sqrt(2 * dt) is more standard for beta(t) = 2.
            # Let's use: z_new = z + 2 * predicted_epsilon / sigma_t * dt + sqrt(2 * dt) * randn_like(z)

            # Check for sigma_t near zero
            sigma_t_safe = sigma_t + 1e-8 # Add epsilon for division stability
            drift = 2 * predicted_epsilon / sigma_t_safe # [num_samples, lat_dim]
            diffusion_coeff_input = torch.tensor(2 * dt + 1e-8, device=current_device, dtype=z.dtype)
            diffusion_coeff = torch.sqrt(diffusion_coeff_input) # Scalar diffusion coefficient

            # Euler-Maruyama update
            z = z + drift * dt + diffusion_coeff * torch.randn_like(z) # [num_samples, lat_dim]

        return z # Final sampled latent features [num_samples, lat_dim]


class FeatureDecoder(nn.Module):
    """
    Decoder Network to transform latent features back to gene expression parameters.
    Takes per-node latent features and outputs parameters for a discrete distribution
    over gene counts for each node.
    """
    def __init__(self, lat_dim, hid_dim, out_dim):
        """
        Args:
            lat_dim (int): Input latent feature dimension per node.
            hid_dim (int): Hidden layer dimension.
            out_dim (int): Output dimension per node (number of genes).
        """
        super().__init__()
        # MLP takes per-node latent features and outputs per-gene parameters
        self.decoder_mlp = nn.Sequential(
            nn.Linear(lat_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim) # Output a value for each gene
        )
        # Output values are interpreted as log rates for a Poisson distribution

    def forward(self, z):
        """
        Args:
            z (torch.Tensor): Latent features per node [num_nodes, lat_dim].

        Returns:
            torch.Tensor: Decoded parameters (log rates) per node [num_nodes, out_dim].
        """
        # z: [num_nodes, lat_dim] - per-node latent features
        if z.size(0) == 0:
            return torch.empty(0, self.decoder_mlp[-1].out_features, device=z.device, dtype=z.dtype)

        # Output raw values, interpret as log rates for a discrete distribution (Poisson)
        log_rates = self.decoder_mlp(z) # [num_nodes, out_dim] - out_dim is number of genes

        return log_rates # Return log rates


# --- Trainer Class Modifications ---
class Trainer:
    """
    Manages training and evaluation of the SpectralGNN DDPM for scRNA-seq generation.
    """
    def __init__(self, in_dim, hid_dim, lat_dim, num_cell_types, pe_dim, timesteps, lr, warmup_steps, total_steps, loss_weights=None):
        """
        Args:
            in_dim (int): Input feature dimension (number of genes).
            hid_dim (int): Model hidden dimension.
            lat_dim (int): Latent space dimension.
            num_cell_types (int): Number of cell types for conditioning.
            pe_dim (int): Positional encoding dimension.
            timesteps (int): Number of diffusion timesteps.
            lr (float): Learning rate.
            warmup_steps (int): Number of steps for learning rate warmup.
            total_steps (int): Total number of training steps.
            loss_weights (dict, optional): Weights for different loss components ('diff', 'kl', 'rec').
                                          Defaults to {'diff': 1.0, 'kl': 0.1, 'rec': 10.0}.
        """
        print("\nInitializing Trainer...")
        self.encoder = SpectralEncoder(in_dim, hid_dim, lat_dim, pe_dim=pe_dim).to(device)
        # Pass num_cell_types to ScoreNet for conditioning
        self.denoiser = ScoreNet(lat_dim, num_cell_types=num_cell_types, time_embed_dim=32).to(device)
        # Decoder outputs parameters for 'in_dim' genes
        self.decoder = FeatureDecoder(lat_dim, hid_dim, in_dim).to(device)
        self.diff = ScoreSDE(self.denoiser, T=1.0, N=timesteps).to(device)

        # LaplacianPerturb is optional, can be used for adversarial graph perturbation
        self.lap_pert = LaplacianPerturb()

        # Gather all parameters for optimization
        self.all_params = list(self.encoder.parameters()) + \
                          list(self.denoiser.parameters()) + \
                          list(self.decoder.parameters())

        # Optimizer
        self.optim = torch.optim.Adam(self.all_params, lr=lr)

        # GradScaler for mixed precision training (if CUDA is available)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.current_step = 0

        # Loss weights
        if loss_weights is None:
            self.loss_weights = {'diff': 1.0, 'kl': 0.1, 'rec': 10.0}
        else:
            self.loss_weights = loss_weights
        print(f"Using loss weights: {self.loss_weights}")

        # Learning rate scheduler (Warmup and Cosine Decay)
        num_warmup_steps_captured = warmup_steps
        num_training_steps_captured = total_steps
        def lr_lambda_fn(current_scheduler_step):
            if num_training_steps_captured == 0: return 1.0 # Avoid division by zero
            actual_warmup_steps = min(num_warmup_steps_captured, num_training_steps_captured)
            if current_scheduler_step < actual_warmup_steps:
                # Linear warmup
                return float(current_scheduler_step + 1) / float(max(1, actual_warmup_steps))
            # Cosine decay after warmup
            decay_phase_duration = num_training_steps_captured - actual_warmup_steps
            if decay_phase_duration <= 0: return 0.0 # No decay phase
            current_step_in_decay = current_scheduler_step - actual_warmup_steps
            progress = float(current_step_in_decay) / float(max(1, decay_phase_duration))
            progress = min(1.0, progress) # Clamp progress
            return 0.5 * (1.0 + np.cos(np.pi * progress)) # Cosine decay from 1.0 to 0.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda_fn)
        print("Trainer initialized.")


    def train_epoch(self, loader):
        """
        Trains the model for one epoch.

        Args:
            loader (DataLoader): DataLoader providing batches of graph data.

        Returns:
            float: Average total loss for the epoch.
        """
        self.encoder.train(); self.denoiser.train(); self.decoder.train()
        total_loss_val, total_loss_diff_val, total_loss_kl_val, total_loss_rec_val = 0.0, 0.0, 0.0, 0.0
        num_batches_processed = 0

        # Assuming the DataLoader yields one large Data object per dataset split
        # If memory is an issue, manual mini-batching of nodes from data object would be needed here.
        for data in loader:
            data = data.to(device) # Move the whole Data object to device
            num_nodes_in_batch = data.x.size(0)

            if num_nodes_in_batch == 0 or data.x is None or data.x.numel() == 0:
                 # print("Skipping empty batch (no nodes or features).")
                 continue

            # Ensure lap_pe and cell_type are present and correctly shaped on device
            lap_pe = data.lap_pe
            if lap_pe is None or lap_pe.size(0) != num_nodes_in_batch or lap_pe.size(1) != self.encoder.pe_dim:
                # print(f"Warning: Invalid lap_pe shape ({lap_pe.shape if lap_pe is not None else 'None'}) in batch. Expected ({num_nodes_in_batch}, {self.encoder.pe_dim}). Using zeros.")
                lap_pe = torch.zeros(num_nodes_in_batch, self.encoder.pe_dim, device=device, dtype=data.x.dtype)

            cell_type_labels = data.cell_type
            if cell_type_labels is None or cell_type_labels.size(0) != num_nodes_in_batch:
                 print(f"Warning: Invalid cell_type_labels shape ({cell_type_labels.shape if cell_type_labels is not None else 'None'}) in batch. Expected ({num_nodes_in_batch},). Using dummy labels (0).")
                 cell_type_labels = torch.zeros(num_nodes_in_batch, dtype=torch.long, device=device) # Dummy labels

            # Ensure cell type labels are within the valid range for the embedding layer
            if cell_type_labels.max() >= self.denoiser.num_cell_types or cell_type_labels.min() < 0:
                 # print(f"Warning: Cell type label out of bounds ({cell_type_labels.min()},{cell_type_labels.max()}). Num cell types: {self.denoiser.num_cell_types}. Clamping labels.")
                 # Clamp labels to valid range before passing to the model
                 cell_type_labels = torch.clamp(cell_type_labels, 0, self.denoiser.num_cell_types - 1)


            # Apply Laplacian Perturbation to graph edges (optional)
            # Assumes binary edges initially, weights are 1.0
            edge_weights = torch.ones(data.edge_index.size(1), device=device, dtype=data.x.dtype) if data.edge_index.numel() > 0 else None

            if edge_weights is not None and edge_weights.numel() > 0:
                # Sample initial weights based on perturbation strength
                initial_perturbed_weights = self.lap_pert.sample(data.edge_index, num_nodes_in_batch)
                # Compute adversarial perturbation (modifies initial_perturbed_weights)
                adversarially_perturbed_weights = self.lap_pert.adversarial(
                    self.encoder, data.x, data.edge_index, initial_perturbed_weights
                )
                # Ensure weights are positive and not NaN/Inf before using in GCN
                adversarially_perturbed_weights = torch.nan_to_num(adversarially_perturbed_weights, nan=1.0, posinf=1.0, neginf=0.0)
                adversarially_perturbed_weights = torch.clamp(adversarially_perturbed_weights, min=1e-4) # Ensure minimum weight
            else:
                 adversarially_perturbed_weights = None


            self.optim.zero_grad(set_to_none=True) # More memory efficient

            # Using autocast for mixed precision training
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Encoder outputs per-node mu and logvar
                mu, logvar = self.encoder(data.x, data.edge_index, lap_pe, adversarially_perturbed_weights) # mu, logvar are [num_nodes, lat_dim]

                if mu.numel() == 0 or logvar.numel() == 0:
                     print("Encoder output is empty. Skipping batch.")
                     continue

                # Ensure mu and logvar have the same size as num_nodes_in_batch
                if mu.size(0) != num_nodes_in_batch or logvar.size(0) != num_nodes_in_batch:
                     print(f"Warning: Encoder output size mismatch. Expected ({num_nodes_in_batch}, {self.encoder.mu_net.out_features}), got mu {mu.shape}, logvar {logvar.shape}. Skipping batch.")
                     continue


                std = torch.exp(0.5 * logvar) # [num_nodes, lat_dim]

                # KL Divergence Loss (from encoder's latent distribution to standard normal)
                # KL is computed per node and then averaged over the batch
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1) # Sum over lat_dim -> [num_nodes]
                kl_div = kl_div.mean() # Mean over batch -> scalar


                # --- Diffusion Loss (per-node latent space) ---
                # Sample time steps per node/sample in the batch
                # Ensure t_indices has num_nodes_in_batch entries
                t_indices = torch.randint(0, self.diff.N, (num_nodes_in_batch,), device=device).long()
                # Get corresponding time values from the predefined timesteps
                time_values_for_loss = self.diff.timesteps[t_indices] # [num_nodes_in_batch]

                # Calculate the marginal std deviation for each time step
                sigma_t_batch = self.diff.marginal_std(time_values_for_loss) # [num_nodes_in_batch]

                # Ensure sigma_t_batch is broadcastable with latent features z (or mu)
                if sigma_t_batch.ndim == 1:
                     sigma_t_batch = sigma_t_batch.unsqueeze(-1) # [num_nodes_in_batch, 1]


                # Sample noise for the diffusion process
                noise_target = torch.randn_like(mu) # [num_nodes_in_batch, lat_dim] - per-node noise

                # Corrupt mu (considered as x_0 in the latent space) based on the forward process (DDPM-like)
                # x_t = exp(-t) * x_0 + sqrt(1 - exp(-2t)) * epsilon
                # zt_corrupted = alpha_t * mu + sigma_t * noise_target
                alpha_t = torch.exp(-time_values_for_loss).unsqueeze(-1) # [num_nodes_in_batch, 1]
                # Use mu.detach() here so that the diffusion loss does not backpropagate through mu's computation
                # It trains the denoiser to predict noise for a given mu, time, and corrupted mu.
                zt_corrupted = alpha_t * mu.detach() + sigma_t_batch * noise_target # [num_nodes_in_batch, lat_dim]


                # Denoiser predicts the noise (epsilon) from the corrupted latent state, time, and condition
                # Pass the *original* cell type labels for conditioning the denoiser
                eps_predicted = self.denoiser(zt_corrupted, time_values_for_loss, cell_type_labels) # [num_nodes_in_batch, lat_dim]

                # Diffusion Loss: MSE between predicted noise and target noise
                loss_diff = F.mse_loss(eps_predicted, noise_target)


                # --- Reconstruction Loss (from latent space mean mu to original counts) ---
                # Decode the mean of the latent space (mu) back to gene expression parameters (log rates)
                decoded_log_rates = self.decoder(mu) # [num_nodes, in_dim] - log_rates for each cell's genes

                # Target for reconstruction loss is the original raw counts data.x
                # Use Poisson Negative Log Likelihood (NLL) loss for discrete count data
                # torch.nn.functional.poisson_nll_loss expects log_input=True for log rates and full=False for mean over batch
                # The target should be a float tensor for this loss function
                target_counts = data.x.float() # Ensure target is float

                # Ensure decoded_log_rates and target_counts have compatible shapes
                if decoded_log_rates.shape != target_counts.shape:
                     print(f"Warning: Decoder output shape ({decoded_log_rates.shape}) mismatch with target counts shape ({target_counts.shape}). Skipping reconstruction loss.")
                     loss_rec = torch.tensor(0.0, device=device) # Set reconstruction loss to 0 if shapes don't match
                 # Ensure decoded_log_rates does not contain NaN/Inf before loss calculation
                elif torch.isnan(decoded_log_rates).any() or torch.isinf(decoded_log_rates).any():
                     print("Warning: Decoder output contains NaN/Inf. Skipping reconstruction loss.")
                     loss_rec = torch.tensor(0.0, device=device)
                else:
                    # Poisson NLL loss: -log P(target_counts | lambda=exp(decoded_log_rates))
                    loss_rec = F.poisson_nll_loss(decoded_log_rates, target_counts, log_input=True, reduction='mean') # 'mean' averages over batch and genes


                # Apply weights and sum losses
                final_loss = (self.loss_weights.get('diff', 0.0) * loss_diff + # Use .get() with default 0.0 if key is missing
                              self.loss_weights.get('kl', 0.0) * kl_div +
                              self.loss_weights.get('rec', 0.0) * loss_rec)

            # Check for NaN/Inf in the total loss before backprop
            if torch.isnan(final_loss) or torch.isinf(final_loss):
                print(f"Warning: NaN/Inf loss detected: {final_loss.item()}. Diff: {loss_diff.item():.4f}, KL: {kl_div.item():.4f}, Rec: {loss_rec.item():.4f}. Skipping batch.")
                # Optionally, save model state or parameters for debugging here
                # For now, just skip the backward pass and optimizer step for this batch
                continue

            # Backward pass and optimizer step with gradient scaling
            self.scaler.scale(final_loss).backward()
            self.scaler.unscale_(self.optim) # Unscale gradients before clipping
            # Clip gradients of all model parameters
            torch.nn.utils.clip_grad_norm_(self.all_params, max_norm=1.0)
            self.scaler.step(self.optim)
            self.scaler.update()
            self.scheduler.step() # Step the learning rate scheduler
            self.current_step += 1 # Increment the global step counter

            # Accumulate losses for reporting
            total_loss_val += final_loss.item()
            total_loss_diff_val += loss_diff.item()
            total_loss_kl_val += kl_div.item()
            total_loss_rec_val += loss_rec.item()
            num_batches_processed +=1

        # Report average losses for the epoch
        if num_batches_processed > 0:
            avg_total_loss = total_loss_val / num_batches_processed
            avg_diff_loss = total_loss_diff_val / num_batches_processed
            avg_kl_loss = total_loss_kl_val / num_batches_processed
            avg_rec_loss = total_loss_rec_val / num_batches_processed
            print(f"Epoch Averages -> Total Loss: {avg_total_loss:.4f}, Diff Loss: {avg_diff_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, Rec Loss: {avg_rec_loss:.4f}")
            return avg_total_loss
        else:
             print("Warning: No batches processed in this epoch.")
             return 0.0 # Return 0 if no batches processed


    @torch.no_grad()
    def generate(self, num_samples, cell_type_condition=None):
        """
        Generates synthetic scRNA-seq count data.

        Args:
            num_samples (int): Number of cells to generate.
            cell_type_condition (list, np.ndarray, torch.Tensor, or None):
                Cell type label(s) for conditional generation.
                If None, generates unconditionally.
                If a single value, applies to all samples.
                If a list/array/tensor, must have length num_samples.

        Returns:
            tuple: (generated_counts, generated_cell_types)
                   generated_counts (np.ndarray): Generated raw counts [num_samples, num_genes].
                   generated_cell_types (np.ndarray): Used cell type labels [num_samples].
        """
        print(f"\nGenerating {num_samples} samples...")
        self.denoiser.eval() # Set denoiser (ScoreNet) to evaluation mode
        self.decoder.eval() # Set decoder to evaluation mode

        # Determine cell type labels for generation
        if cell_type_condition is None:
            # Unconditional generation: create dummy cell type labels (e.g., all 0)
            # ScoreNet expects cell_type_labels to be a tensor if not None.
            print("Generating unconditionally.")
            gen_cell_type_labels_tensor = torch.zeros(num_samples, dtype=torch.long, device=device) # Use 0 as dummy index
        else:
            # Conditional generation
            if isinstance(cell_type_condition, (list, np.ndarray)):
                 gen_cell_type_labels_tensor = torch.tensor(cell_type_condition, dtype=torch.long, device=device)
            elif isinstance(cell_type_condition, torch.Tensor):
                 gen_cell_type_labels_tensor = cell_type_condition.to(device).long()
            else:
                 raise ValueError("cell_type_condition must be None, a list, numpy array, or torch.Tensor.")

            # Ensure cell type labels tensor size matches num_samples or is 1
            if gen_cell_type_labels_tensor.size(0) == 1 and num_samples > 1:
                 # Broadcast single condition to all samples
                 gen_cell_type_labels_tensor = gen_cell_type_labels_tensor.repeat(num_samples)
                 print(f"Generating {num_samples} cells with broadcasted cell type condition: {gen_cell_type_labels_tensor[0].item()}.")
            elif gen_cell_type_labels_tensor.size(0) != num_samples:
                 raise ValueError(f"Number of cell type conditions ({gen_cell_type_labels_tensor.size(0)}) must match number of samples to generate ({num_samples}) or be 1.")
            else:
                 print(f"Generating {num_samples} cells with specified cell types.")

            # Ensure cell type labels are within the valid range for the embedding layer in ScoreNet
            if gen_cell_type_labels_tensor.max() >= self.denoiser.num_cell_types or gen_cell_type_labels_tensor.min() < 0:
                 print(f"Warning: Generated cell type label out of bounds ({gen_cell_type_labels_tensor.min()},{gen_cell_type_labels_tensor.max()}). Num cell types in model: {self.denoiser.num_cell_types}. Clamping labels.")
                 gen_cell_type_labels_tensor = torch.clamp(gen_cell_type_labels_tensor, 0, self.denoiser.num_cell_types - 1)


        # Sample latent features from the diffusion model
        z_gen_shape = (num_samples, self.diff.score_model.lat_dim) # lat_dim from ScoreNet's init
        # Pass cell type labels to the sample method for conditional sampling
        z_generated = self.diff.sample(z_gen_shape, cell_type_labels=gen_cell_type_labels_tensor) # [num_samples, lat_dim]

        # Decode latent features back to gene expression parameters (log rates)
        decoded_log_rates = self.decoder(z_generated) # [num_samples, in_dim]

        # Sample counts from the discrete distribution (Poisson) using the decoded log rates
        try:
            # Convert log rates to rates (lambda for Poisson)
            # Use Softplus for numerical stability if rates could be very low, but Exp is standard for log rates
            rates = torch.exp(decoded_log_rates) # Rates must be positive

            # Ensure rates are positive and finite for Poisson distribution
            rates = torch.clamp(rates, min=1e-6) # Add small epsilon to rates to avoid log(0) if needed by distribution fn
            rates = torch.nan_to_num(rates, nan=1e-6, posinf=1e6, neginf=1e-6) # Replace potential NaN/Inf with safe values


            # Create Poisson distribution object
            poisson_dist = torch.distributions.Poisson(rates)

            # Sample discrete counts from the distribution
            generated_counts_tensor = poisson_dist.sample() # Sample discrete counts as float tensor
            generated_counts_tensor = generated_counts_tensor.int().float() # Convert to int and then back to float for consistency

        except Exception as e:
             print(f"Error during sampling from Poisson distribution: {e}. Returning zero counts."); traceback.print_exc()
             # As a fallback, return zero counts if sampling fails
             generated_counts_tensor = torch.zeros_like(decoded_log_rates) # Zero counts with correct shape and device


        # Move generated counts and cell types to CPU and convert to numpy arrays
        generated_counts_np = generated_counts_tensor.cpu().numpy()
        generated_cell_types_np = gen_cell_type_labels_tensor.cpu().numpy()

        print("Generation complete.")
        return generated_counts_np, generated_cell_types_np


    @torch.no_grad()
    def evaluate_generation(self, real_adata, generated_counts, generated_cell_types, n_pcs=30, mmd_scales=[0.01, 0.1, 1, 10, 100]):
        """
        Evaluates the generative performance using MMD and 2-Wasserstein distance on PCA projections.

        Args:
            real_adata (anndata.AnnData): AnnData object for the real test set (should be gene-filtered).
            generated_counts (np.ndarray): Generated raw counts [num_samples, num_genes].
            generated_cell_types (np.ndarray): Used cell type labels for generated data [num_samples].
            n_pcs (int): Number of principal components to use for projection.
            mmd_scales (list): List of scales (sigmas) for the RBF kernel MMD calculation.

        Returns:
            dict: Dictionary containing MMD and Wasserstein distance results.
        """
        print("\n--- Computing Evaluation Metrics ---")
        # Set models to evaluation mode (although not directly used in this function, good practice)
        self.denoiser.eval()
        self.decoder.eval()
        if hasattr(self, 'encoder'): self.encoder.eval()


        # Ensure real_adata contains raw counts and cell types
        if not hasattr(real_adata, 'X') or real_adata.X is None:
            print("Error: Real data AnnData object is missing .X (counts).")
            return {"MMD": {}, "Wasserstein": {}, "Notes": "Missing real data counts."}

        real_counts = real_adata.X # This should be the filtered counts

        # Check if real_adata has cell types and retrieve them
        if hasattr(real_adata, 'obs') and 'cell_type' in real_adata.obs.columns:
            real_cell_types_present = True
            real_cell_type_series = real_adata.obs['cell_type']
            # Ensure cell_type is categorical
            if not pd.api.types.is_categorical_dtype(real_cell_type_series):
                 try:
                     real_cell_type_series = real_cell_type_series.astype('category')
                 except Exception as e:
                     print(f"Error converting real cell_type to categorical: {e}. Using raw values for indexing.")
                     # Fallback to using raw values for indexing if categorical conversion fails
                     unique_types, real_cell_type_labels = np.unique(real_cell_type_series.values, return_inverse=True)
                     real_cell_type_categories = unique_types.tolist()

            else:
                real_cell_type_labels = real_cell_type_series.cat.codes.values # Integer codes
                real_cell_type_categories = real_cell_type_series.cat.categories.tolist() # Category names

            print(f"Found {len(real_cell_type_categories)} cell types in real data.")

        else:
            print("Warning: 'cell_type' not found in real_adata.obs. Conditional evaluation will be skipped.")
            real_cell_types_present = False
            real_cell_type_labels = np.zeros(real_counts.shape[0], dtype=int) # Dummy labels
            real_cell_type_categories = ["Unknown"] # Dummy category


        # Ensure generated_counts is a numpy array with correct dtype
        if isinstance(generated_counts, torch.Tensor):
             generated_counts_np = generated_counts.cpu().numpy()
        else:
             generated_counts_np = generated_counts # Assume it's already numpy

        if isinstance(generated_cell_types, torch.Tensor):
             generated_cell_types_np = generated_cell_types.cpu().numpy()
        else:
             generated_cell_types_np = generated_cell_types # Assume it's already numpy


        # Ensure gene dimensions match between real and generated data
        if real_counts.shape[1] != generated_counts_np.shape[1]:
             print(f"Error: Gene dimension mismatch between real ({real_counts.shape[1]}) and generated ({generated_counts_np.shape[1]}) data. Cannot compute metrics.")
             return {"MMD": {}, "Wasserstein": {}, "Notes": "Gene dimension mismatch."}

        if real_counts.shape[0] == 0 or generated_counts_np.shape[0] == 0:
             print("Warning: Real or generated data is empty. Cannot compute metrics.")
             return {"MMD": {}, "Wasserstein": {}, "Notes": "Real or generated data empty."}


        # --- Preprocessing for Evaluation (Normalization and Log-transformation) ---
        print("Applying normalization (to 1e4) and log1p transformation for evaluation.")

        def normalize_and_log(counts):
            """Normalizes counts to 1e4 total counts per cell and applies log1p."""
            # Ensure counts is a numpy array (handle sparse if necessary)
            if sp.issparse(counts):
                 counts_dense = counts.toarray()
            elif isinstance(counts, torch.Tensor):
                 counts_dense = counts.cpu().numpy()
            else:
                 counts_dense = counts

            # Normalize to a total of 1e4 counts per cell
            cell_totals = counts_dense.sum(axis=1, keepdims=True)
            # Avoid division by zero for cells with zero total counts
            cell_totals[cell_totals == 0] = 1.0 # Replace 0 sums with 1 to avoid Inf/NaN
            normalized_counts = counts_dense / cell_totals * 1e4

            # Apply log1p transformation
            log1p_counts = np.log1p(normalized_counts)
            return log1p_counts

        # Preprocess real data counts
        real_log1p = normalize_and_log(real_counts) # [num_real_cells, num_genes]

        # Preprocess generated counts
        generated_log1p = normalize_and_log(generated_counts_np) # [num_gen_cells, num_genes]


        # --- PCA Projection ---
        print(f"Performing PCA to {n_pcs} dimensions on real data and projecting generated data.")
        # Ensure valid dimensions for PCA
        actual_n_pcs = min(n_pcs, real_log1p.shape[0] - 1, real_log1p.shape[1])

        if actual_n_pcs <= 0:
             print("Warning: Cannot perform PCA with 0 or fewer components. Skipping PCA-based metrics.")
             return {"MMD": {}, "Wasserstein": {}, "Notes": "PCA components <= 0."}

        try:
            pca = PCA(n_components=actual_n_pcs, random_state=0)
            # Fit PCA only on real data
            real_pca = pca.fit_transform(real_log1p) # [num_real_cells, actual_n_pcs]
            # Transform generated data using real data's PCA loadings
            generated_pca = pca.transform(generated_log1p) # [num_gen_cells, actual_n_pcs]
        except Exception as e:
            print(f"Error during PCA computation: {e}. Skipping PCA-based metrics."); traceback.print_exc()
            return {"MMD": {}, "Wasserstein": {}, "Notes": f"PCA failed: {e}"}

        # Ensure projected data are numpy arrays and not empty
        if real_pca.shape[0] == 0 or generated_pca.shape[0] == 0:
             print("Error: PCA projected data is empty.")
             return {"MMD": {}, "Wasserstein": {}, "Notes": "PCA projected data empty."}


        # --- Compute Distribution Distances ---

        mmd_results = {}
        wasserstein_results = {}

        # Conditional Evaluation (per cell type)
        print("Computing conditional metrics (per cell type).")

        # Get unique cell types present in both real and generated data for conditional evaluation
        if real_cell_types_present:
             unique_real_cell_types = np.unique(real_cell_type_labels)
             unique_gen_cell_types = np.unique(generated_cell_types_np)
             # Find common cell type codes
             common_cell_type_codes = np.intersect1d(unique_real_cell_types, unique_gen_cell_types)

             if len(common_cell_type_codes) == 0:
                  print("Warning: No common cell types found between real and generated data based on labels. Skipping conditional metrics per type.")
                  # Can still proceed with unconditional if needed
                  pass
             else:
                 print(f"Computing conditional metrics for {len(common_cell_type_codes)} common cell types.")
                 conditional_mmd = {scale: [] for scale in mmd_scales}
                 conditional_wasserstein = []

                 for cell_type_code in common_cell_type_codes:
                     try:
                         # Get cell type name for logging
                         cell_type_name = real_cell_type_categories[cell_type_code] # Use real data categories for names

                         # Filter data for the current cell type based on integer codes
                         real_pca_type = real_pca[real_cell_type_labels == cell_type_code]
                         generated_pca_type = generated_pca[generated_cell_types_np == cell_type_code]

                         if real_pca_type.shape[0] < 2 or generated_pca_type.shape[0] < 2:
                             print(f"Warning: Skipping metrics for cell type '{cell_type_name}' (Code {cell_type_code}): Insufficient samples (Real: {real_pca_type.shape[0]}, Gen: {generated_pca_type.shape[0]}). Need at least 2 samples.")
                             continue

                         print(f"Computing metrics for cell type: '{cell_type_name}' (Code {cell_type_code}) - Real samples: {real_pca_type.shape[0]}, Gen samples: {generated_pca_type.shape[0]}")


                         # --- Compute MMD for the cell type ---
                         def rbf_kernel_mmd(X, Y, scales):
                             """Computes RBF-kernel MMD^2 for given scales."""
                             if X.shape[0] == 0 or Y.shape[0] == 0: return {scale: 0.0 for scale in scales}

                             n, m = X.shape[0], Y.shape[0]
                             # Compute pairwise squared Euclidean distances (once for all scales)
                             try:
                                  pairwise_sq_dist_xx = cdist(X, X, 'sqeuclidean')
                                  pairwise_sq_dist_yy = cdist(Y, Y, 'sqeuclidean')
                                  pairwise_sq_dist_xy = cdist(X, Y, 'sqeuclidean')
                             except Exception as e:
                                  print(f"Error computing pairwise distances for MMD in type {cell_type_name}: {e}. Returning zeros."); traceback.print_exc()
                                  return {scale: 0.0 for scale in scales}


                             mmd_results_scales = {}
                             for scale in scales:
                                 # Gamma for RBF kernel: gamma = 1 / (2 * sigma^2). Here, scale is sigma.
                                 gamma = 1.0 / (2. * scale**2)
                                 # Avoid division by zero if scale is zero (should not happen with requested scales)
                                 if scale == 0: gamma = 1.0 # Or handle as error

                                 # Compute kernel matrices using the same gamma for all distance matrices
                                 K_xx = np.exp(-gamma * pairwise_sq_dist_xx)
                                 K_yy = np.exp(-gamma * pairwise_sq_dist_yy)
                                 K_xy = np.exp(-gamma * pairwise_sq_dist_xy)

                                 # Compute MMD^2 (unbiased estimate)
                                 # MMD^2_u(X, Y) = 1/(n(n-1)) sum_{i!=j} k(x_i, x_j) + 1/(m(m-1)) sum_{i!=j} k(y_i, y_j) - 2/(nm) sum_{i,j} k(x_i, y_j)
                                 # For the prompt's formula (1/n^2, 1/m^2), it's the biased estimate, which is simpler:
                                 # MMD^2_b(X, Y) = 1/n^2 sum_{i,j} k(x_i, x_j) + 1/m^2 sum_{i,j} k(y_i, y_j) - 2/(nm) sum_{i,j} k(x_i, y_j)
                                 # K_xx.mean() is 1/n^2 sum k(x_i, x_j), etc.
                                 mmd2 = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

                                 # MMD^2 can be negative due to numerical precision or estimation bias, clip at 0
                                 mmd_results_scales[scale] = max(0, mmd2) # Report MMD^2

                             return mmd_results_scales


                         mmd_type_scales = rbf_kernel_mmd(real_pca_type, generated_pca_type, mmd_scales)
                         for scale, mmd_val in mmd_type_scales.items():
                              conditional_mmd[scale].append(mmd_val)


                         # --- Compute 2-Wasserstein distance for the cell type using POT ---
                         try:
                             # Compute cost matrix (squared Euclidean distance in PC space)
                             # M = cdist(real_pca_type, generated_pca_type, 'sqeuclidean') # Already computed as pairwise_sq_dist_xy
                             M = cdist(real_pca_type, generated_pca_type, 'sqeuclidean')

                             # Compute optimal transport plan and Wasserstein distance
                             # Use uniform weights for each sample
                             a = np.ones((real_pca_type.shape[0],), dtype=np.float64) / real_pca_type.shape[0]
                             b = np.ones((generated_pca_type.shape[0],), dtype=np.float64) / generated_pca_type.shape[0]

                             # Use EMD2 for exact Wasserstein-2 distance
                             # ot.emd2(a, b, M) returns the squared Wasserstein-2 distance when M is squared Euclidean distance
                             wasserstein2_type = ot.emd2(a, b, M)
                             # The 2-Wasserstein distance is the square root of this
                             wasserstein_type = np.sqrt(wasserstein2_type)
                             conditional_wasserstein.append(wasserstein_type)

                         except Exception as e:
                             print(f"Error computing 2-Wasserstein distance for cell type '{cell_type_name}': {e}. Returning NaN."); traceback.print_exc()
                             conditional_wasserstein.append(np.nan) # Append NaN if computation fails

                     except Exception as e_outer:
                          print(f"An error occurred processing cell type code {cell_type_code}: {e_outer}. Skipping."); traceback.print_exc()
                          # Ensure NaNs are appended for scales and Wasserstein if a type fails completely
                          for scale in mmd_scales: conditional_mmd[scale].append(np.nan)
                          conditional_wasserstein.append(np.nan)


                 # Average conditional metrics over cell types
                 print("Averaging conditional metrics over cell types.")
                 for scale in mmd_scales:
                     valid_mmd = [m for m in conditional_mmd[scale] if not np.isnan(m)]
                     if valid_mmd:
                          mmd_results[f'Conditional_Avg_Scale_{scale}'] = np.mean(valid_mmd)
                     else: mmd_results[f'Conditional_Avg_Scale_{scale}'] = np.nan # Indicate no valid data


                 valid_wasserstein = [w for w in conditional_wasserstein if not np.isnan(w)]
                 if valid_wasserstein:
                     wasserstein_results['Conditional_Avg'] = np.mean(valid_wasserstein)
                 else: wasserstein_results['Conditional_Avg'] = np.nan # Indicate no valid data

        else:
            print("Skipping conditional metrics: Real data is missing cell type labels.")


        # Unconditional Evaluation (on the full dataset or sampled batches)
        print("Computing unconditional metrics.")

        # Sample batches of 5,000 cells for unconditional evaluation if datasets are large
        sample_size_unconditional = 5000
        real_pca_uncond = real_pca
        generated_pca_uncond = generated_pca

        # If dataset is larger than sample_size_unconditional, sample randomly
        if real_pca_uncond.shape[0] > sample_size_unconditional:
            print(f"Sampling {sample_size_unconditional} real cells for unconditional metrics.")
            real_pca_uncond_indices = np.random.choice(real_pca_uncond.shape[0], size=sample_size_unconditional, replace=False)
            real_pca_uncond = real_pca_uncond[real_pca_uncond_indices]

        if generated_pca_uncond.shape[0] > sample_size_unconditional:
            print(f"Sampling {sample_size_unconditional} generated cells for unconditional metrics.")
            generated_pca_uncond_indices = np.random.choice(generated_pca_uncond.shape[0], size=sample_size_unconditional, replace=False)
            generated_pca_uncond = generated_pca_uncond[generated_pca_uncond_indices]

        if real_pca_uncond.shape[0] < 2 or generated_pca_uncond.shape[0] < 2:
             print("Warning: Skipping unconditional metrics: Insufficient samples after sampling (Real: {real_pca_uncond.shape[0]}, Gen: {generated_pca_uncond.shape[0]}). Need at least 2 samples.")
             # Skip unconditional metrics
        else:
             print(f"Unconditional evaluation samples: Real {real_pca_uncond.shape[0]}, Gen {generated_pca_uncond.shape[0]}.")
             # Compute MMD for unconditional
             mmd_uncond_scales = rbf_kernel_mmd(real_pca_uncond, generated_pca_uncond, mmd_scales)
             for scale, mmd_val in mmd_uncond_scales.items():
                  mmd_results[f'Unconditional_Scale_{scale}'] = mmd_val

             # Compute 2-Wasserstein distance for unconditional
             try:
                 # Compute cost matrix
                 M_uncond = cdist(real_pca_uncond, generated_pca_uncond, 'sqeuclidean')
                 a_uncond = np.ones((real_pca_uncond.shape[0],), dtype=np.float64) / real_pca_uncond.shape[0]
                 b_uncond = np.ones((generated_pca_uncond.shape[0],), dtype=np.float64) / generated_pca_uncond.shape[0]

                 # ot.emd2 returns the squared Wasserstein-2 distance
                 wasserstein2_uncond = ot.emd2(a_uncond, b_uncond, M_uncond)
                 # The 2-Wasserstein distance is the square root
                 wasserstein_uncond = np.sqrt(wasserstein2_uncond)
                 wasserstein_results['Unconditional'] = wasserstein_uncond

             except Exception as e:
                 print(f"Error computing unconditional 2-Wasserstein distance: {e}. Returning NaN."); traceback.print_exc()
                 wasserstein_results['Unconditional'] = np.nan


        print("Evaluation metrics computation complete.")
        return {"MMD": mmd_results, "Wasserstein": wasserstein_results}


if __name__ == '__main__':
    # --- Configuration ---
    # BATCH_SIZE is typically 1 for DataLoader when loading a single large graph (dataset split)
    # If processing multiple smaller graphs or implementing manual minibatching of nodes, this would change.
    BATCH_SIZE = 1

    LEARNING_RATE = 5e-5 # Learning rate for Adam optimizer
    EPOCHS = 3000 # Number of training epochs
    HIDDEN_DIM = 1024 # Hidden dimension for GNN and MLPs
    LATENT_DIM = 512 # Dimension of the latent space for diffusion
    PE_DIM = 20 # Dimension of Laplacian Positional Encoding
    K_NEIGHBORS = 20 # Number of neighbors for KNN graph on cells
    PCA_NEIGHBORS = 50 # Number of PCA components to use before KNN graph construction
    GENE_THRESHOLD = 20 # Minimum number of cells a gene must be expressed in (applied across the loaded split)
    TIMESTEPS_DIFFUSION = 1500 # Number of diffusion timesteps (N in ScoreSDE)

    # Loss Weights - balance diffusion, KL, and reconstruction losses
    # These may need tuning based on observed loss values during training
    loss_weights = {'diff': 0.001, 'kl': 0.07, 'rec': 11.0} # Increased reconstruction weight as it's on raw counts

    # Dataset Paths (Relative to script location or absolute)
    # Assumes pbmc3k_test.h5ad and pbmc3k_train.h5ad are in a 'data' subdirectory
    TRAIN_H5AD = 'data/pbmc3k_train.h5ad'
    TEST_H5AD = 'data/pbmc3k_test.h5ad'

    # --- Dynamic Path for Processed Data ---
    # Processed data root directory includes parameters to distinguish different processed versions
    DATA_ROOT = 'data/pbmc3k_processed'
    os.makedirs(DATA_ROOT, exist_ok=True) # Ensure root exists

    # Create specific root directories for train and test processed data
    TRAIN_DATA_ROOT = os.path.join(DATA_ROOT, f'train_k{K_NEIGHBORS}_pe{PE_DIM}_gt{GENE_THRESHOLD}_pca{PCA_NEIGHBORS}')
    TEST_DATA_ROOT = os.path.join(DATA_ROOT, f'test_k{K_NEIGHBORS}_pe{PE_DIM}_gt{GENE_THRESHOLD}_pca{PCA_NEIGHBORS}')
    # Ensure 'processed' subdirectory exists within each root
    os.makedirs(os.path.join(TRAIN_DATA_ROOT, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_ROOT, 'processed'), exist_ok=True)


    # --- Load Training Data ---
    train_dataset = None
    input_feature_dim = 0 # Initialize dimensions in case of error
    num_cell_types = 1
    filtered_gene_names_from_train = []

    try:
        print(f"Loading/Processing training data from: {TRAIN_H5AD} into {TRAIN_DATA_ROOT}")
        train_dataset = PBMC3KDataset(
            h5ad_path=TRAIN_H5AD,
            k_neighbors=K_NEIGHBORS,
            pe_dim=PE_DIM,
            root=TRAIN_DATA_ROOT,
            train=True,
            gene_threshold=GENE_THRESHOLD,
            pca_neighbors=PCA_NEIGHBORS
        )

        if train_dataset is None or len(train_dataset) == 0 or (len(train_dataset) == 1 and train_dataset.get(0) is None) or (len(train_dataset) == 1 and train_dataset.get(0).num_nodes == 0):
            print("ERROR: Training dataset is empty or contains no nodes after processing. Check H5AD file or processing logic.");
            num_train_cells = 0
            # input_feature_dim and num_cell_types remain initialized to 0 and 1
            print("Setting dimensions to 0 and proceeding, but training is not possible.")
        else:
            num_train_cells = train_dataset.get(0).num_nodes
            input_feature_dim = train_dataset.get(0).x.size(1)
            num_cell_types = train_dataset.num_cell_types
            train_cell_type_categories = train_dataset.cell_type_categories
            # --- Retrieve the filtered gene names directly from the processed dataset object ---
            filtered_gene_names_from_train = train_dataset.filtered_gene_names
            print(f"Training dataset loaded. Number of cells: {num_train_cells}, Input features (genes): {input_feature_dim}, Number of cell types: {num_cell_types}")
            print(f"Retrieved {len(filtered_gene_names_from_train)} filtered gene names from training dataset object.")

            # --- Get the list of genes kept after filtering the TRAINING data ---
            # We need to reload the raw adata to get original gene names and apply the filter logic again
            try:
                train_adata_raw = sc.read_h5ad(TRAIN_H5AD)
                initial_train_counts = train_adata_raw.X
                if sp.issparse(initial_train_counts):
                    genes_expressed_count = np.asarray((initial_train_counts > 0).sum(axis=0)).flatten()
                else:
                    genes_expressed_count = np.count_nonzero(initial_train_counts, axis=0)
                genes_to_keep_mask = genes_expressed_count >= GENE_THRESHOLD
                # Store the names of the genes that passed the filter in the training data
                filtered_gene_names = train_adata_raw.var_names[genes_to_keep_mask].tolist()
                print(f"Identified {len(filtered_gene_names)} genes after filtering training data (threshold >= {GENE_THRESHOLD}).")

            except Exception as e_genes:
                 print(f"Warning: Could not get filtered gene names from training data: {e_genes}. Evaluation gene check might fail."); traceback.print_exc()
                 # Keep filtered_gene_names as empty if failed, the evaluation check will then trigger the error

    except Exception as e:
        print(f"FATAL ERROR loading or processing training data: {e}"); traceback.print_exc(); # Use print_exc() for detailed error
        num_train_cells = 0
        input_feature_dim = 0
        num_cell_types = 1
        train_dataset = None
        filtered_gene_names = [] # Ensure empty list on error
        print("Fatal error during training data loading/processing. Training will be skipped.")

    # Proceed only if training data was loaded successfully and is not empty
    if num_train_cells > 0 and input_feature_dim > 0:
        # Set num_workers to 0 for potentially better compatibility with PyTorch Geometric and CUDA
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
        if len(train_loader) == 0: print("Warning: DataLoader is empty. Check dataset or batch size."); # Training will be skipped

        # Calculate total training steps and warmup steps for the scheduler
        TOTAL_TRAINING_STEPS = len(train_loader) * EPOCHS if len(train_loader) > 0 else 0
        WARMUP_STEPS = max(1, int(0.05 * TOTAL_TRAINING_STEPS)) if TOTAL_TRAINING_STEPS > 0 else 0 # 5% warmup
        WARMUP_STEPS = min(WARMUP_STEPS, TOTAL_TRAINING_STEPS // 2) # Cap warmup steps at half total steps

        # Initialize the Trainer
        trainer = Trainer(
            in_dim=input_feature_dim,
            hid_dim=HIDDEN_DIM,
            lat_dim=LATENT_DIM,
            num_cell_types=num_cell_types, # Pass the actual number of cell types
            pe_dim=PE_DIM,
            timesteps=TIMESTEPS_DIFFUSION,
            lr=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            total_steps=TOTAL_TRAINING_STEPS,
            loss_weights=loss_weights
        )

        # --- Training Loop ---
        if TOTAL_TRAINING_STEPS > 0:
            print(f"\nStarting training for {EPOCHS} epochs, {TOTAL_TRAINING_STEPS} total steps (warmup: {WARMUP_STEPS}). Initial LR: {LEARNING_RATE:.2e}")
            for epoch in range(1, EPOCHS + 1):
                print(f"\nEpoch {epoch}/{EPOCHS}")
                avg_epoch_loss = trainer.train_epoch(train_loader) # This now prints sub-losses
                current_lr = trainer.optim.param_groups[0]["lr"]
                # Print epoch summary including average loss and current learning rate
                # Printing is already handled inside train_epoch now.
                # print(f'Epoch {epoch:03d}/{EPOCHS}, AvgTotalLoss: {avg_epoch_loss:.4f}, Current LR: {current_lr:.3e}')

                # You can add periodic saving of the model here if desired
                # if epoch % 100 == 0 or epoch == EPOCHS:
                #     torch.save({
                #         'epoch': epoch,
                #         'encoder_state_dict': trainer.encoder.state_dict(),
                #         'denoiser_state_dict': trainer.denoiser.state_dict(),
                #         'decoder_state_dict': trainer.decoder.state_dict(),
                #         'optimizer_state_dict': trainer.optim.state_dict(),
                #         'scheduler_state_dict': trainer.scheduler.state_dict(),
                #         'scaler_state_dict': trainer.scaler.state_dict(),
                #         'current_step': trainer.current_step,
                #     }, f'checkpoint_epoch_{epoch}.pt')
                #     print(f"Model checkpoint saved at epoch {epoch}")


            print("\nTraining completed.")
        else:
             print("\nSkipping training: No training steps available.")

    else:
        print("\nSkipping training: Training data is empty or has no features.")


    # --- Final Evaluation on Test Set ---
    print("\n--- Starting Final Evaluation on Test Set ---")
    test_adata = None # Initialize test_adata to None
    num_genes_eval = 0 # Initialize for the check

    # Ensure filtered_gene_names from training is available
    # This list is populated in the training data loading section.
    if not filtered_gene_names_from_train:
        print("ERROR: Filtered gene names from training data are not available. Cannot perform consistent filtering for evaluation. Skipping evaluation.")
    else:
        try:
            print(f"Loading test data from: {TEST_H5AD}")
            test_adata_raw = sc.read_h5ad(TEST_H5AD)

            # --- Apply gene filtering to test data consistently with training data ---
            # Filter the raw test data to keep only genes present in the filtered training data names
            # This ensures the gene set matches exactly what the model was trained on.
            if not hasattr(test_adata_raw, 'var_names') or test_adata_raw.var_names is None:
                 print("FATAL ERROR: Raw test adata is missing .var_names. Cannot match training genes. Skipping evaluation.")
                 test_adata = None # Invalidate test_adata
            else:
                 # Find genes in the raw test data that are also in the training data's filtered gene names
                 genes_to_keep_in_test_mask = test_adata_raw.var_names.isin(filtered_gene_names_from_train)

                 if np.sum(genes_to_keep_in_test_mask) == 0:
                      print("FATAL ERROR: No genes in the raw test data match the filtered training genes. Skipping evaluation.")
                      test_adata = None # Invalidate test_adata
                 else:
                      # Apply this filter to the test data
                      test_adata = test_adata_raw[:, genes_to_keep_in_test_mask].copy()
                      num_genes_eval = test_adata.shape[1]
                      print(f"Test data shape after consistent gene filtering (matching training genes): {test_adata.shape[0]} cells, {num_genes_eval} genes.")

                      # Check if 'cell_type' is present in test data for conditional evaluation
                      if 'cell_type' not in test_adata.obs.columns:
                           print("Warning: 'cell_type' column missing in test_adata. Conditional evaluation will rely on generated cell types or be skipped if no common types.")
                           # You might want to create a dummy cell_type column here if absolutely necessary for downstream steps
                           # test_adata.obs['cell_type'] = pd.Categorical(['Unknown'] * test_adata.shape[0])

        except FileNotFoundError:
            print(f"FATAL ERROR: Test H5AD file not found at {TEST_H5AD}. Skipping final evaluation."); traceback.print_exc(); test_adata = None
        except Exception as e:
            print(f"FATAL ERROR loading or filtering test AnnData for evaluation: {e}. Skipping final evaluation."); traceback.print_exc(); test_adata = None


    # Proceed with evaluation only if test data was loaded and filtered successfully AND filtered_gene_names_from_train were available
    if test_adata is not None and test_adata.shape[0] > 0 and test_adata.shape[1] > 0 and filtered_gene_names_from_train:
        # Ensure the trainer object exists and models were initialized (requires successful training data load)
        if 'trainer' in locals() and trainer is not None:

            # Check if the decoder output dimension matches the number of genes in the consistently filtered test data
            # This check should now pass if the logic above worked correctly
            if trainer.decoder.decoder_mlp[-1].out_features != num_genes_eval:
                 # This error should ideally not be hit now, but as a safeguard:
                 print(f"FATAL ERROR: Decoder output dimension ({trainer.decoder.decoder_mlp[-1].out_features}) still does not match number of genes in consistently filtered test data ({num_genes_eval}). Cannot proceed with evaluation.")
                 # You might want to raise an error here or exit if this happens, as it indicates a logic issue
                 # raise ValueError("Gene dimension mismatch after consistent filtering.")
            else:
                # --- Generate Data for Evaluation ---
                # Generate three datasets matching the size of the original test set
                num_test_cells = test_adata.shape[0] # Get the number of cells from the consistently filtered test data
                print(f"\nGenerating 3 datasets of size {num_test_cells} for evaluation.")
                generated_datasets_counts = []
                generated_datasets_cell_types = []

                # Get real test cell types for conditional generation templates
                # Use cell types from the loaded and consistently filtered test_adata object
                cell_type_condition_for_gen = None # Default to unconditional
                if 'cell_type' in test_adata.obs.columns:
                     # Ensure categories are consistent if possible, or just use codes directly
                     if pd.api.types.is_categorical_dtype(test_adata.obs['cell_type']):
                          # Check if there are any valid cell types
                          if not test_adata.obs['cell_type'].empty:
                               real_test_cell_type_labels = test_adata.obs['cell_type'].cat.codes.values
                               print("Generating conditionally based on real test set cell types.")
                               cell_type_condition_for_gen = real_test_cell_type_labels
                          else:
                               print("Warning: 'cell_type' column found but is empty in test_adata.obs. Generating unconditionally.")

                     else:
                          # Convert to codes if not categorical, but only if not empty
                          if not test_adata.obs['cell_type'].empty:
                               unique_types, real_test_cell_type_labels = np.unique(test_adata.obs['cell_type'].values, return_inverse=True)
                               print("Generating conditionally based on real test set cell types.")
                               cell_type_condition_for_gen = real_test_cell_type_labels
                          else:
                              print("Warning: 'cell_type' column found but is empty in test_adata.obs. Generating unconditionally.")

                if cell_type_condition_for_gen is None:
                     print("Generating unconditionally.")


                # Generate the specified number of datasets
                num_datasets_to_generate = 3
                for i in range(num_datasets_to_generate):
                    print(f"Generating dataset {i+1}/{num_datasets_to_generate}...")
                    try:
                        # Pass the determined cell_type_condition_for_gen (either array or None)
                        gen_counts, gen_types = trainer.generate(num_samples=num_test_cells, cell_type_condition=cell_type_condition_for_gen)
                        generated_datasets_counts.append(gen_counts)
                        generated_datasets_cell_types.append(gen_types)
                        print(f"Dataset {i+1} generated.")
                    except Exception as e:
                        print(f"Error generating dataset {i+1}: {e}. Skipping this dataset."); traceback.print_exc();
                        # Append None or empty arrays to maintain list length if generation fails
                        generated_datasets_counts.append(None)
                        generated_datasets_cell_types.append(None)


                # --- Compute and Report Metrics ---
                print("\nComputing metrics for generated datasets...")
                # Initialize lists to store results across datasets
                all_mmd_results_per_scale = {scale: [] for scale in [0.01, 0.1, 1, 10, 100]}
                all_wasserstein_results = []
                individual_metrics_reports = [] # To store results for each generated dataset


                for i in range(len(generated_datasets_counts)):
                     gen_counts = generated_datasets_counts[i]
                     gen_types = generated_datasets_cell_types[i]

                     # Only evaluate if generation was successful for this dataset and has data
                     if gen_counts is not None and gen_types is not None and gen_counts.shape[0] > 0 and gen_counts.shape[1] > 0:
                         print(f"\nEvaluating generated dataset {i+1}/{len(generated_datasets_counts)} (Shape: {gen_counts.shape[0]} cells, {gen_counts.shape[1]} genes)...")

                         try:
                             # Compute evaluation metrics for the current generated dataset
                             metrics = trainer.evaluate_generation(
                                 real_adata=test_adata, # Use the consistently filtered test AnnData object
                                 generated_counts=gen_counts,
                                 generated_cell_types=gen_types,
                                 n_pcs=30, # Number of PCA components as specified
                                 mmd_scales=[0.01, 0.1, 1, 10, 100] # MMD scales as specified
                             )
                             individual_metrics_reports.append(metrics)
                             print(f"Metrics for dataset {i+1}: {metrics}")

                             # Aggregate results across the 3 generated datasets for averaging
                             if metrics and "MMD" in metrics and metrics["MMD"]:
                                  for scale in all_mmd_results_per_scale.keys():
                                       # Prioritize conditional average if available, otherwise use unconditional
                                       if f'Conditional_Avg_Scale_{scale}' in metrics["MMD"] and not np.isnan(metrics["MMD"][f'Conditional_Avg_Scale_{scale}']):
                                            all_mmd_results_per_scale[scale].append(metrics["MMD"][f'Conditional_Avg_Scale_{scale}'])
                                       elif f'Unconditional_Scale_{scale}' in metrics["MMD"] and not np.isnan(metrics["MMD"][f'Unconditional_Scale_{scale}']):
                                            all_mmd_results_per_scale[scale].append(metrics["MMD"][f'Unconditional_Scale_{scale}'])
                                       # If neither is present/valid for this scale, skip appending


                             if metrics and "Wasserstein" in metrics and metrics["Wasserstein"]:
                                 # Prioritize conditional average if available, otherwise use unconditional
                                 if 'Conditional_Avg' in metrics["Wasserstein"] and not np.isnan(metrics["Wasserstein"]['Conditional_Avg']):
                                     all_wasserstein_results.append(metrics["Wasserstein"]['Conditional_Avg'])
                                 elif 'Unconditional' in metrics["Wasserstein"] and not np.isnan(metrics["Wasserstein"]['Unconditional']):
                                     all_wasserstein_results.append(metrics["Wasserstein"]['Unconditional'])
                                 # If neither is present/valid, skip appending

                         except Exception as e_eval:
                              print(f"Error during evaluation of dataset {i+1}: {e_eval}. Skipping evaluation for this dataset."); traceback.print_exc();
                              individual_metrics_reports.append({"Notes": f"Evaluation failed: {e_eval}"}) # Append failure note


                     else:
                         print(f"\nSkipping evaluation for dataset {i+1} as generation failed or produced empty data.")
                         individual_metrics_reports.append({"Notes": "Generation failed or produced empty data."})


                # Report averaged metrics over the valid generated datasets
                print("\n--- Averaged Evaluation Metrics over Generated Datasets ---")
                averaged_metrics = {"MMD": {}, "Wasserstein": {}}

                # Average MMD results per scale
                for scale, results in all_mmd_results_per_scale.items():
                     if results: # Check if the list is not empty for this scale
                         averaged_metrics["MMD"][f'Average_Scale_{scale}'] = np.mean(results)
                         # Optional: add std dev
                         # averaged_metrics["MMD"][f'Average_Scale_{scale}_std'] = np.std(results)
                     else:
                          averaged_metrics["MMD"][f'Average_Scale_{scale}'] = np.nan # Report NaN if no valid results for this scale


                # Average Wasserstein results
                if all_wasserstein_results: # Check if the list is not empty
                     averaged_metrics["Wasserstein"]['Average'] = np.mean(all_wasserstein_results)
                     # Optional: add std dev
                     # averaged_metrics["Wasserstein"]['Average_std'] = np.std(all_wasserstein_results)
                else:
                     averaged_metrics["Wasserstein"]['Average'] = np.nan # Report NaN if no valid results

                # Print the final averaged metrics
                print("\nFinal Averaged Metrics:")
                print(averaged_metrics)

                # Optional: Print individual reports as well
                # print("\nIndividual Dataset Metrics Reports:")
                # for i, report in enumerate(individual_metrics_reports):
                #     print(f"Dataset {i+1}: {report}")


        else:
            print("\nSkipping final evaluation: Trainer object not initialized (training data loading failed).")

    else:
        print("\nSkipping final evaluation: Test data is empty or has no features after filtering, or filtered gene names were not available.")


    print("\nScript execution finished.")
