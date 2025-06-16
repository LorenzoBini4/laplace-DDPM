import os
import traceback
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops, to_undirected
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from utils import compute_lap_pe
import torch_geometric
from torch_geometric.data import download_url, extract_zip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
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
