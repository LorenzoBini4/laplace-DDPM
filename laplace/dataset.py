import os
import traceback
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
import scipy.sparse as sp
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops, to_undirected
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from .utils import compute_lap_pe
import torch_geometric
from torch_geometric.data import download_url, extract_zip
import warnings
from collections.abc import Mapping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
class LaplaceDataset(InMemoryDataset):
    def __init__(self, h5ad_path, k_neighbors=15, pe_dim=10, root='data/laplace', train=True, gene_threshold=20, pca_neighbors=50, seed=42, dataset_name='laplace', force_reprocess=False, use_pseudo_labels=False,
                 cell_type_path=None, cell_type_col="cell_type", barcode_col="barcode", cell_type_unknown="Unknown"):
        self.h5ad_path = h5ad_path
        self.k_neighbors = k_neighbors
        self.pe_dim = pe_dim
        self.train = train
        self.gene_threshold = gene_threshold
        self.pca_neighbors = pca_neighbors
        self.seed = seed
        self.force_reprocess = force_reprocess
        self.use_pseudo_labels = use_pseudo_labels
        self.cell_type_path = cell_type_path
        self.cell_type_col = cell_type_col
        self.barcode_col = barcode_col
        self.cell_type_unknown = cell_type_unknown
        self.filtered_gene_names = []
        self.cell_type_categories = ["Unknown"] # Default
        self.num_cell_types = 1 # Default

        processed_file = f'{dataset_name}_{"train" if train else "test"}_k{k_neighbors}_pe{pe_dim}_gt{gene_threshold}_pca{pca_neighbors}.pt'
        self.processed_file_names_list = [processed_file]
        super().__init__(root=root, transform=None, pre_transform=None)

        # Ensure processed data and metadata exist or process them
        metadata_path = self.processed_paths[0].replace(".pt", "_metadata.pt")
        if self.force_reprocess:
            if os.path.exists(self.processed_paths[0]):
                os.remove(self.processed_paths[0])
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
        if not os.path.exists(self.processed_paths[0]) or not os.path.exists(metadata_path):
            print(f"Processed file or metadata not found. Dataset will be processed.")
            if os.path.exists(self.processed_paths[0]): os.remove(self.processed_paths[0]) # Clean up if one exists but not other
            if os.path.exists(metadata_path): os.remove(metadata_path)
            self.process()
        try:
            data, slices = torch.load(self.processed_paths[0], weights_only=False)
            self._data = data
            self.slices = slices
            print(f"Successfully loaded processed data from {self.processed_paths[0]}")
            if os.path.exists(metadata_path):
                metadata = torch.load(metadata_path, weights_only=False)
                self.filtered_gene_names = metadata.get('filtered_gene_names', [])
                self.cell_type_categories = metadata.get('cell_type_categories', ["Unknown"])
                self.num_cell_types = metadata.get('num_cell_types', 1)
                self.log_libsize_mean = metadata.get('log_libsize_mean', None)
                self.log_libsize_std = metadata.get('log_libsize_std', None)
                print(f"Loaded metadata: {len(self.filtered_gene_names)} filtered gene names, {self.num_cell_types} cell types ({self.cell_type_categories[:5]}...).")
            else:
                print(f"Warning: Metadata file {metadata_path} not found. Attributes might be default or inferred if possible.")
                data_ref = self._data if hasattr(self, "_data") and self._data is not None else None
                if data_ref and hasattr(data_ref, 'x') and data_ref.x is not None:
                    if not self.filtered_gene_names and data_ref.x.shape[1] > 0 : self.filtered_gene_names = [f"gene_{i}" for i in range(data_ref.x.shape[1])]
                    if hasattr(data_ref, 'cell_type') and data_ref.cell_type is not None:
                        unique_codes = torch.unique(data_ref.cell_type).cpu().numpy()
                        self.num_cell_types = len(unique_codes)
                        self.cell_type_categories = [f"Type_{code}" for code in sorted(unique_codes)]
            data_ref = self._data if hasattr(self, "_data") and self._data is not None else None
            if data_ref is None or data_ref.num_nodes == 0:
                 print("Warning: Loaded processed data is empty or has no nodes.")
        
        except Exception as e:
            print(f"Error loading processed data or metadata from {self.processed_paths[0]}: {e}. Attempting to re-process.")
            traceback.print_exc()
            if os.path.exists(self.processed_paths[0]): os.remove(self.processed_paths[0])
            if os.path.exists(metadata_path): os.remove(metadata_path)
            self.process() # Re-process on loading error
            # Retry load after re-process
            if os.path.exists(self.processed_paths[0]):
                data, slices = torch.load(self.processed_paths[0], weights_only=False)
                self._data = data
                self.slices = slices
            else:
                raise RuntimeError(f"Re-process failed to create {self.processed_paths[0]}")

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
        print(f"Processing data from H5/H5AD: {self.h5ad_path} for {'train' if self.train else 'test'} set.")
        print(f"Parameters: k={self.k_neighbors}, PE_dim={self.pe_dim}, Gene Threshold={self.gene_threshold}, PCA Neighbors={self.pca_neighbors}")

        adata = None
        chromatin_emb = None
        
        try:
            # Check if this is the Raw Multiome H5
            if self.h5ad_path.endswith(".h5"):
                print("Detected .h5 file. Assuming 10x Multiome format.")
                # Suppress "Variable names are not unique" warning
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Variable names are not unique", category=UserWarning)
                    adata = sc.read_10x_h5(self.h5ad_path, gex_only=False)
                
                adata.var_names_make_unique()
                # 1. Filter Cells (Raw -> ~14k)
                target_cells = 14500 # Heuristic for "14k sorted nuclei"
                print(f"Raw data shape: {adata.shape}. Filtering to top {target_cells} cells by counts.")
                sc.pp.calculate_qc_metrics(adata, inplace=True)
                cutoff = np.partition(adata.obs['total_counts'], -target_cells)[-target_cells]
                sc.pp.filter_cells(adata, min_counts=cutoff)
                sc.pp.filter_cells(adata, min_counts=cutoff)
                print(f"Filtered data shape: {adata.shape}")
                
                # --- TRAIN/TEST SPLIT ---
                n_total = adata.shape[0]
                indices = np.arange(n_total)
                np.random.seed(self.seed) # Use instance seed
                np.random.shuffle(indices)
                
                split_idx = int(0.8 * n_total)
                if self.train:
                    indices_to_use = indices[:split_idx]
                    print(f"SPLIT: Using TRAINING set (first 80%): {len(indices_to_use)} cells.")
                else:
                    indices_to_use = indices[split_idx:]
                    print(f"SPLIT: Using TEST set (last 20%): {len(indices_to_use)} cells.")
                
                adata = adata[indices_to_use].copy()
                # ----------------------------------------

                # 2. Split Modalities
                print("Splitting Gene Expression and Peaks...")
                raw_var = adata.var
                if 'feature_types' in raw_var:
                    gex_mask = raw_var['feature_types'] == 'Gene Expression'
                    atac_mask = raw_var['feature_types'] == 'Peaks'
                else:
                    # Fallback if feature_types missing (unlikely for 10x h5)
                    print("Warning: 'feature_types' not found. Assuming all GEX or using var names.")
                    gex_mask = np.ones(adata.shape[1], dtype=bool) 
                    atac_mask = np.zeros(adata.shape[1], dtype=bool)

                adata_atac = adata[:, atac_mask].copy()
                adata = adata[:, gex_mask].copy() # Keep only GEX in main adata
                print(f"GEX Shape: {adata.shape}, ATAC Shape: {adata_atac.shape}")

                # 3. Process ATAC -> LSI -> Chromatin Embedding
                if adata_atac.shape[1] > 0:
                    print("Processing ATAC data with LSI (TF-IDF + SVD)...")
                    import sklearn.decomposition
                    from sklearn.feature_extraction.text import TfidfTransformer
                    
                    # Binarize
                    X_atac = adata_atac.X
                    if sp.issparse(X_atac):
                        X_atac.data = np.ones_like(X_atac.data)
                    else:
                        X_atac[X_atac > 0] = 1
                    
                    # TF-IDF
                    # "Term Frequency" in scATAC is just binary freq, IDF is standard.
                    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
                    X_tfidf = tfidf.fit_transform(X_atac)
                    
                    # SVD (PCA on sparse without mean centering usually)
                    # LSI usually means SVD on TF-IDF.
                    n_components = 50 
                    svd = sklearn.decomposition.TruncatedSVD(n_components=n_components, random_state=42)
                    chromatin_emb_np = svd.fit_transform(X_tfidf)
                    
                    # Remove 1st component if correlated with depth (standard practice in LSI)
                    # But for "context embedding" it might be fine. Let's keep all 50.
                    
                    chromatin_emb = torch.from_numpy(chromatin_emb_np).float()
                    print(f"Computed Chromatin Embeddings: {chromatin_emb.shape}")
                else:
                    print("No ATAC features found.")

            else:
                # Standard H5AD (GEX only usually)
                adata = sc.read_h5ad(self.h5ad_path)
                try:
                    adata.var_names_make_unique()
                except Exception:
                    pass
                
                # --- TRAIN/TEST SPLIT ---
                n_total = adata.shape[0]
                indices = np.arange(n_total)
                np.random.seed(self.seed) 
                np.random.shuffle(indices)
                split_idx = int(0.8 * n_total)
                
                if self.train:
                    indices_to_use = indices[:split_idx]
                    print(f"SPLIT: Using TRAINING set (first 80%): {len(indices_to_use)} cells.")
                else:
                    indices_to_use = indices[split_idx:]
                    print(f"SPLIT: Using TEST set (last 20%): {len(indices_to_use)} cells.")
                
                adata = adata[indices_to_use].copy()
                # ----------------------------------------
            if not isinstance(adata.X, (np.ndarray, sp.spmatrix)):
                 print(f"Warning: adata.X is of type {type(adata.X)}. Attempting conversion.")
                 try: adata.X = sp.csr_matrix(adata.X)
                 except: adata.X = np.array(adata.X)

        except FileNotFoundError:
            print(f"FATAL ERROR: File not found at {self.h5ad_path}"); raise
        except Exception as e:
            print(f"FATAL ERROR reading file {self.h5ad_path}: {e}"); traceback.print_exc(); raise

        def _load_cell_type_labels(a):
            if not self.cell_type_path:
                return
            if not os.path.exists(self.cell_type_path):
                print(f"Warning: cell_type_path not found: {self.cell_type_path}. Using existing labels/Unknown.")
                return
            try:
                df = pd.read_csv(self.cell_type_path, sep=None, engine="python")
            except Exception as e:
                print(f"Warning: failed to read cell_type_path {self.cell_type_path}: {e}. Skipping.")
                return
            if self.barcode_col in df.columns and self.cell_type_col in df.columns:
                mapping = dict(zip(df[self.barcode_col].astype(str), df[self.cell_type_col].astype(str)))
            elif df.shape[1] >= 2:
                mapping = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
            else:
                print(f"Warning: cell_type_path {self.cell_type_path} has insufficient columns.")
                return
            obs_names = a.obs_names.astype(str)
            labels = [mapping.get(bc, self.cell_type_unknown) for bc in obs_names]
            a.obs["cell_type"] = pd.Categorical(labels)
            missing = sum(1 for bc in obs_names if bc not in mapping)
            if missing > 0:
                print(f"Warning: {missing}/{len(obs_names)} barcodes missing in cell_type_path. Filled with {self.cell_type_unknown}.")

        def _is_integerish(x):
            if sp.issparse(x):
                if x.data.size == 0:
                    return True
                if np.any(x.data < 0):
                    return False
                frac = np.max(np.abs(x.data - np.rint(x.data)))
                return frac <= 1e-3
            x = np.asarray(x)
            if x.size == 0:
                return True
            if np.any(x < 0):
                return False
            frac = np.max(np.abs(x - np.rint(x)))
            return frac <= 1e-3

        def _resolve_counts_and_varnames(a):
            if isinstance(getattr(a, "layers", None), Mapping) and "counts" in a.layers:
                return a.layers["counts"], a.var_names, "layers['counts']"
            if getattr(a, "raw", None) is not None and getattr(a.raw, "X", None) is not None:
                return a.raw.X, a.raw.var_names, "raw.X"
            return a.X, a.var_names, "X"

        def _coerce_counts(x):
            if sp.issparse(x):
                if x.data.size == 0:
                    return x
                if np.any(x.data < 0):
                    print("Warning: Negative values found in counts; clipping to 0.")
                    x.data = np.clip(x.data, 0, None)
                frac = np.max(np.abs(x.data - np.rint(x.data)))
                if frac > 1e-3:
                    print("Warning: Non-integer counts detected; rounding to nearest integer.")
                    x.data = np.rint(x.data)
                return x
            x = np.asarray(x)
            if np.any(x < 0):
                print("Warning: Negative values found in counts; clipping to 0.")
                x = np.clip(x, 0, None)
            frac = np.max(np.abs(x - np.rint(x))) if x.size else 0.0
            if frac > 1e-3:
                print("Warning: Non-integer counts detected; rounding to nearest integer.")
                x = np.rint(x)
            return x

        _load_cell_type_labels(adata)
        counts, var_names, counts_source = _resolve_counts_and_varnames(adata)
        if counts_source == "X":
            if _is_integerish(adata.X):
                if isinstance(getattr(adata, "layers", None), Mapping):
                    adata.layers["counts"] = adata.X.copy() if not sp.issparse(adata.X) else adata.X.copy()
                else:
                    adata.layers = {"counts": adata.X.copy() if not sp.issparse(adata.X) else adata.X.copy()}
                counts, var_names, counts_source = _resolve_counts_and_varnames(adata)
                print("Warning: No counts layer/raw found. Inferred counts from X and stored in layers['counts'].")
            else:
                raise ValueError("Counts not found in layers['counts'] or raw.X, and adata.X is non-integer/non-counts. Please provide raw counts.")
        counts = _coerce_counts(counts)
        num_cells, initial_num_genes = counts.shape
        print(f"Initial GEX data shape: {num_cells} cells, {initial_num_genes} genes.")
        print(f"Using counts from: {counts_source}")

        if num_cells == 0 or initial_num_genes == 0:
             print("Warning: Input data is empty. Creating empty Data object.")
             return

        print(f"Filtering genes expressed in fewer than {self.gene_threshold} cells.")
        if sp.issparse(counts):
            genes_expressed_count = np.asarray((counts > 0).sum(axis=0)).flatten()
            genes_to_keep_mask = genes_expressed_count >= self.gene_threshold
            counts = counts[:, genes_to_keep_mask]
            self.filtered_gene_names = np.array(var_names)[genes_to_keep_mask].tolist()
        else:
            genes_expressed_count = np.count_nonzero(counts, axis=0)
            genes_to_keep_mask = genes_expressed_count >= self.gene_threshold
            counts = counts[:, genes_to_keep_mask]
            if var_names is not None:
                self.filtered_gene_names = [var_names[i] for i, k in enumerate(genes_to_keep_mask) if k]
            else:
                self.filtered_gene_names = [f'gene_{i}' for i in np.where(genes_to_keep_mask)[0]]

        num_genes_after_filtering = counts.shape[1]
        print(f"Number of genes after filtering: {num_genes_after_filtering}")
        # Library size statistics (for generation scaling)
        if sp.issparse(counts):
            libsize = np.asarray(counts.sum(axis=1)).flatten()
        else:
            libsize = counts.sum(axis=1)
        libsize = np.clip(libsize, 1.0, None)
        log_libsize = np.log(libsize)
        log_libsize_mean = float(np.mean(log_libsize))
        log_libsize_std = float(np.std(log_libsize) + 1e-6)
        
        # Check if chromatin_emb exists and matches num_cells (it should if we filtered same adata)
        if chromatin_emb is not None:
             if chromatin_emb.size(0) != num_cells:
                 print(f"Warning: Chromatin embedding size {chromatin_emb.size(0)} != num_cells {num_cells}. This implies filtering mismatch.")
                 chromatin_emb = None

        if 'cell_type' not in adata.obs.columns:
             if self.use_pseudo_labels:
                 print("Warning: 'cell_type' not found in adata.obs.columns. Deriving pseudo-labels with Leiden.")
                 cell_type_labels = np.zeros(num_cells, dtype=int)
                 num_cell_types = 1
                 cell_type_categories = ["Unknown"]
                 try:
                     adata_tmp = sc.AnnData(X=counts)
                     with warnings.catch_warnings():
                         warnings.filterwarnings("ignore", message="Some cells have zero counts", category=UserWarning)
                         sc.pp.normalize_total(adata_tmp, target_sum=1e4)
                     sc.pp.log1p(adata_tmp)
                     sc.pp.pca(adata_tmp, n_comps=min(50, adata_tmp.shape[1] - 1))
                     sc.pp.neighbors(adata_tmp, n_neighbors=min(15, adata_tmp.shape[0] - 1))
                     sc.tl.leiden(adata_tmp, resolution=1.0)
                     cell_type_series = adata_tmp.obs['leiden'].astype('category')
                     cell_type_labels = cell_type_series.cat.codes.values
                     num_cell_types = len(cell_type_series.cat.categories)
                     cell_type_categories = cell_type_series.cat.categories.tolist()
                 except Exception as e:
                     print(f"Warning: Leiden pseudo-labeling failed: {e}. Using Unknown label.")
             else:
                 print("Warning: 'cell_type' not found in adata.obs.columns. Using single Unknown label.")
                 cell_type_labels = np.zeros(num_cells, dtype=int)
                 num_cell_types = 1
                 cell_type_categories = ["Unknown"]
        else:
            cell_type_series = adata.obs['cell_type']
            if not pd.api.types.is_categorical_dtype(cell_type_series):
                 try: cell_type_series = cell_type_series.astype('category')
                 except: pass # fallback
            
            if pd.api.types.is_categorical_dtype(cell_type_series):
                cell_type_labels = cell_type_series.cat.codes.values
                num_cell_types = len(cell_type_series.cat.categories)
                cell_type_categories = cell_type_series.cat.categories.tolist()
            else:
                 unique_types, cell_type_labels = np.unique(cell_type_series.values, return_inverse=True)
                 num_cell_types = len(unique_types)
                 cell_type_categories = unique_types.tolist()

        self.num_cell_types = num_cell_types
        self.cell_type_categories = cell_type_categories

        edge_index_feat = torch.empty((2,0), dtype=torch.long)
        edge_index_phys = torch.empty((2,0), dtype=torch.long)
        pos = torch.zeros((num_cells, 2), dtype=torch.float)
        lap_pe = torch.zeros((num_cells, self.pe_dim), dtype=torch.float)

        # 1. Build FEATURE Graph (Gene Expression Space)
        if num_cells > 1 and self.k_neighbors > 0 and num_genes_after_filtering > 0:
            actual_k_for_knn = min(self.k_neighbors, num_cells - 1)
            print(f"Building FEATURE graph (k={actual_k_for_knn}) on PCA-reduced expression.")
            pca_input = counts
            if sp.issparse(pca_input):
                try:
                    # print("Converting sparse counts to dense for PCA.") 
                    # Use chunked/incremental PCA if needed for huge datasets, but validation sets are usually ok.
                    pca_input_dense = pca_input.toarray()
                except Exception as e:
                     print(f"Error converting sparse to dense for PCA: {e}. Using raw counts."); traceback.print_exc()
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
                     except Exception as e:
                         print(f"Error during PCA: {e}. Using raw counts."); 
                         pca_coords = pca_input_dense
                 else:
                      pca_coords = pca_input_dense
            
            knn_input_coords = pca_coords if pca_coords is not None else pca_input_dense
            if knn_input_coords is not None and knn_input_coords.shape[0] > 1:
                try:
                    nbrs = NearestNeighbors(n_neighbors=actual_k_for_knn + 1, algorithm='auto', metric='euclidean').fit(knn_input_coords)
                    distances, indices = nbrs.kneighbors(knn_input_coords)
                    source_nodes = np.repeat(np.arange(num_cells), actual_k_for_knn)
                    target_nodes = indices[:, 1:].flatten()
                    edges = np.stack([source_nodes, target_nodes], axis=0)
                    edge_index_feat = torch.tensor(edges, dtype=torch.long)
                    edge_index_feat = to_undirected(edge_index_feat, num_nodes=num_cells)
                    edge_index_feat, _ = remove_self_loops(edge_index_feat)
                    print(f"FEATURE Graph built: {edge_index_feat.size(1)} edges.")
                except Exception as e:
                    print(f"Error building FEATURE graph: {e}."); traceback.print_exc()

        # 2. Build PHYSICAL Graph (Spatial Space)
        if 'spatial' in adata.obsm:
            print(f"Building PHYSICAL graph (k={self.k_neighbors}) on Spatial Coordinates.")
            spatial_coords = adata.obsm['spatial']
            pos = torch.from_numpy(spatial_coords).float()
            
            try:
                # Constructing Spatial KNN.
                nbrs_phys = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='auto', metric='euclidean').fit(spatial_coords)
                _, indices_phys = nbrs_phys.kneighbors(spatial_coords)
                source_phys = np.repeat(np.arange(num_cells), self.k_neighbors)
                target_phys = indices_phys[:, 1:].flatten()
                
                edges_phys = np.stack([source_phys, target_phys], axis=0)
                edge_index_phys = torch.tensor(edges_phys, dtype=torch.long)
                edge_index_phys = to_undirected(edge_index_phys, num_nodes=num_cells)
                edge_index_phys, _ = remove_self_loops(edge_index_phys)
                print(f"PHYSICAL Graph built: {edge_index_phys.size(1)} edges.")
                
            except Exception as e:
                print(f"Error building PHYSICAL graph: {e}."); traceback.print_exc()
        else:
            print("No 'spatial' in obsm. PHYSICAL graph will be empty.")

        target_graph_for_pe = edge_index_feat if edge_index_feat.numel() > 0 else edge_index_phys
        
        if num_cells > 0 and target_graph_for_pe.numel() > 0:
             print(f"Computing Laplacian PE with dim {self.pe_dim}...")
             try:
                 lap_pe = compute_lap_pe(target_graph_for_pe.cpu(), num_cells, k=self.pe_dim).to(device)
                 lap_pe = lap_pe.to(torch.float32)
             except Exception as e:
                  print(f"Error computing PE: {e}."); traceback.print_exc()
        
        if sp.issparse(counts):
            x = torch.from_numpy(counts.toarray().copy()).float()
        else:
            x = torch.from_numpy(counts.copy()).float()
        cell_type = torch.from_numpy(cell_type_labels.copy()).long()
        if isinstance(adata.obs, pd.DataFrame) and 'batch' in adata.obs.columns:
            batch_ids = pd.Categorical(adata.obs['batch']).codes.astype(np.int64)
        else:
            batch_ids = np.zeros(num_cells, dtype=np.int64)
        batch_id = torch.from_numpy(batch_ids.copy()).long()
        
        if chromatin_emb is None:
             chromatin_emb = torch.zeros((num_cells, 0), dtype=torch.float)

        # Save both graphs in Data object
        # edge_index (standard) will be FEATURE graph. edge_index_phys will be PHYSICAL.
        data = Data(x=x, 
                    edge_index=edge_index_feat, 
                    edge_index_phys=edge_index_phys,
                    pos=pos,
                    lap_pe=lap_pe, 
                    cell_type=cell_type,
                    chromatin=chromatin_emb, # Added Chromatin Embedding
                    batch_id=batch_id,
                    num_nodes=num_cells)

        if data.x.size(0) != num_cells: print(f"Warning: Data.x size mismatch.")
        if data.lap_pe.size(0) != num_cells: print(f"Warning: Data.lap_pe size mismatch.")
        if data.cell_type.size(0) != num_cells: print(f"Warning: Data.cell_type size mismatch.")
        if data.chromatin.size(0) != num_cells: print(f"Warning: Data.chromatin size mismatch.")
        if data.batch_id.size(0) != num_cells: print(f"Warning: Data.batch_id size mismatch.")
        if data.edge_index.numel() > 0 and data.edge_index.max() >= num_cells: print(f"Warning: Edge index out of bounds.")

        data_list = [data]
        try:
            data_to_save, slices_to_save = self.collate(data_list)
            torch.save((data_to_save, slices_to_save), self.processed_paths[0])
            print(f"Processed and saved data to {self.processed_paths[0]}")
            metadata = {
                'filtered_gene_names': self.filtered_gene_names,
                'cell_type_categories': self.cell_type_categories,
                'num_cell_types': self.num_cell_types,
                'counts_source': counts_source,
                'context_dim': chromatin_emb.size(1), # Save context dim
                'log_libsize_mean': log_libsize_mean,
                'log_libsize_std': log_libsize_std
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
