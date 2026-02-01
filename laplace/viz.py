import os   
import numpy as np  
import pandas as pd 
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from typing import List, Union
import umap
from sklearn.decomposition import PCA

def set_plotting_style():
    """
    Set the plotting style for matplotlib and seaborn.
    """
    plt.style.use('seaborn-whitegrid')
    sns.set_context("notebook", font_scale=1.2)
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16


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

    # --- figure 1a: Mean-Variance Plot ---
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

    # --- figure 1b: Sparsity (Zeros per Cell) ---
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

    # --- figure 2: UMAP Visualization (Joint Embedding for Real and Generated) ---
    print("Plotting Figure 2: Joint UMAP for Real and Generated Data...")
    if real_counts_np.shape[0] > 0 and generated_counts.shape[0] > 0 and \
       real_counts_np.shape[1] > 0 and generated_counts.shape[1] > 0 and \
       real_counts_np.shape[1] == generated_counts.shape[1]:

        gen_cell_type_str_list = [
            train_cell_type_categories[int(code)] if 0 <= int(code) < len(train_cell_type_categories) else f"UnknownGen_{int(code)}"
            for code in generated_cell_types
        ]
        real_cell_type_str_list = None
        if 'cell_type' in real_adata_filtered.obs:
            if isinstance(real_adata_filtered.obs['cell_type'].dtype, pd.CategoricalDtype):
                real_cell_type_str_list = real_adata_filtered.obs['cell_type'].astype(str).tolist()
            elif pd.api.types.is_numeric_dtype(real_adata_filtered.obs['cell_type']):
                real_cell_type_str_list = real_adata_filtered.obs['cell_type'].apply(
                    lambda x: train_cell_type_categories[int(x)] if 0 <= int(x) < len(train_cell_type_categories) else f"UnknownReal_{int(x)}"
                ).astype(str).tolist()
            else:
                real_cell_type_str_list = real_adata_filtered.obs['cell_type'].astype(str).tolist()
        else:
            real_cell_type_str_list = ['Unknown_Real'] * real_counts_np.shape[0]

        var_names_all = [str(name) for name in train_filtered_gene_names]
        if not var_names_all and generated_counts.shape[1] > 0:
            var_names_all = [f"Gene_{i}" for i in range(generated_counts.shape[1])]

        adata_real = ad.AnnData(
            X=real_counts_np,
            obs=pd.DataFrame({'cell_type_str_umap': pd.Categorical(real_cell_type_str_list), 'source': 'real'}),
            var=pd.DataFrame(index=var_names_all)
        )
        adata_gen = ad.AnnData(
            X=generated_counts,
            obs=pd.DataFrame({'cell_type_str_umap': pd.Categorical(gen_cell_type_str_list), 'source': 'generated'}),
            var=pd.DataFrame(index=var_names_all)
        )
        adata_real.var_names = [str(v) for v in adata_real.var_names]
        adata_gen.var_names = [str(v) for v in adata_gen.var_names]
        adata_real.var_names_make_unique()
        adata_gen.var_names_make_unique()
        # drop zero-count cells to avoid scanpy normalization warnings
        try:
            sc.pp.filter_cells(adata_real, min_counts=1)
            sc.pp.filter_cells(adata_gen, min_counts=1)
        except Exception:
            pass

        sc.pp.normalize_total(adata_real, target_sum=1e4)
        sc.pp.normalize_total(adata_gen, target_sum=1e4)
        sc.pp.log1p(adata_real)
        sc.pp.log1p(adata_gen)

        n_top_genes = min(2000, adata_real.shape[1] - 1 if adata_real.shape[1] > 1 else 1)
        if n_top_genes > 0:
            X_hvg = adata_real.X
            if sp.issparse(X_hvg):
                mean = np.asarray(X_hvg.mean(axis=0)).ravel()
                mean_sq = np.asarray(X_hvg.multiply(X_hvg).mean(axis=0)).ravel()
                var = mean_sq - mean ** 2
            else:
                var = np.var(X_hvg, axis=0)
            var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
            top_idx = np.argsort(var)[-n_top_genes:]
            hvg_mask = np.zeros(adata_real.shape[1], dtype=bool)
            hvg_mask[top_idx] = True
            adata_real = adata_real[:, hvg_mask].copy()
            adata_gen = adata_gen[:, hvg_mask].copy()

        X_real = adata_real.X.A if sp.issparse(adata_real.X) else adata_real.X
        X_gen = adata_gen.X.A if sp.issparse(adata_gen.X) else adata_gen.X
        mean = X_real.mean(axis=0)
        std = X_real.std(axis=0)
        std[std == 0] = 1.0
        X_real = (X_real - mean) / std
        X_gen = (X_gen - mean) / std

        n_comps = min(50, X_real.shape[0] - 1 if X_real.shape[0] > 1 else 50,
                      X_real.shape[1] - 1 if X_real.shape[1] > 1 else 50)
        if n_comps < 2:
            print("Skipping UMAP: not enough samples/features.")
        else:
            pca = PCA(n_components=n_comps, random_state=0)
            X_real_pca = pca.fit_transform(X_real)
            X_gen_pca = pca.transform(X_gen)

            current_umap_neighbors = min(umap_neighbors, X_real_pca.shape[0] - 1 if X_real_pca.shape[0] > 1 else umap_neighbors)
            if current_umap_neighbors < 2:
                print("Skipping UMAP: not enough neighbors.")
            else:
                umap_model = umap.UMAP(n_neighbors=current_umap_neighbors, min_dist=0.3, random_state=0)
                umap_real = umap_model.fit_transform(X_real_pca)
                umap_gen = umap_model.transform(X_gen_pca)

                adata_real_plot = adata_real.copy()
                adata_gen_plot = adata_gen.copy()
                adata_real_plot.obsm["X_umap"] = umap_real
                adata_gen_plot.obsm["X_umap"] = umap_gen

                plt.figure(figsize=(8, 8))
                sc.pl.umap(adata_real_plot, color='cell_type_str_umap',
                           frameon=False, legend_fontsize=8, legend_loc='on data', show=False,
                           title=f"UMAP of Real Data Cell Types",
                           save=f"_figure2_umap_real_{model_name.replace(' ', '_')}.png")
                default_save_path_real = f"figures/umap_figure2_umap_real_{model_name.replace(' ', '_')}.png"
                target_save_path_real = os.path.join(output_dir, f"figure2_umap_real_cell_types_{model_name.replace(' ', '_')}.png")
                if os.path.exists(default_save_path_real):
                    os.rename(default_save_path_real, target_save_path_real)
                    print(f"Figure 2 (Real Data UMAP) saved to {target_save_path_real}")
                plt.close()

                plt.figure(figsize=(8, 8))
                sc.pl.umap(adata_gen_plot, color='cell_type_str_umap',
                           frameon=False, legend_fontsize=8, legend_loc='on data', show=False,
                           title=f"UMAP of Generated ({model_name}) Cell Types",
                           save=f"_figure2_umap_gen_{model_name.replace(' ', '_')}.png")
                default_save_path_gen = f"figures/umap_figure2_umap_gen_{model_name.replace(' ', '_')}.png"
                target_save_path_gen = os.path.join(output_dir, f"figure2_umap_generated_cell_types_{model_name.replace(' ', '_')}.png")
                if os.path.exists(default_save_path_gen):
                    os.rename(default_save_path_gen, target_save_path_gen)
                    print(f"Figure 2 (Generated Data UMAP) saved to {target_save_path_gen}")
                plt.close()
    else:
        print("Skipping UMAP: Real or generated data empty or gene mismatch.")

    # clean up the 'figures' directory if it's empty after moving
    if os.path.exists("./figures") and not os.listdir("./figures"):
        try:
            os.rmdir("./figures")
        except OSError:
            pass # directory might not be empty if other plots are saved there by scanpy
            
    print(f"Qualitative plotting for {model_name} finished.")
