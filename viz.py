import os   
import numpy as np  
import pandas as pd 
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
# Ensure that the plotting libraries are set up correctly
from typing import List, Union

# Set plotting style for consistent aesthetics
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

    # --- figure 2: UMAP Visualization (Separate Plots for Real and Generated) ---
    print("Plotting Figure 2: Separate UMAP Visualizations for Real and Generated Data...")

    # --- process Real Data for UMAP ---
    if real_counts_np.shape[0] > 0 and real_counts_np.shape[1] > 0:
        adata_real_processed = real_adata_filtered.copy()
        adata_real_processed.var_names = [str(name) for name in adata_real_processed.var_names] # ensure var_names are strings

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
                # use 'cell_ranger' flavor for HVG, as it handles non-integer data after normalization/log1p
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
                    
                    # plot Real Data UMAP
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


    # --- process Generated Data for UMAP ---
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
            return # exit plotting for UMAP if no genes

        adata_gen_processed = ad.AnnData(
            X=generated_counts,
            obs=pd.DataFrame({
                'cell_type_str_umap': pd.Categorical(gen_cell_type_str_list)
            }),
            var=pd.DataFrame(index=var_names_for_gen)
        )
        adata_gen_processed.var_names = [str(name) for name in adata_gen_processed.var_names] # ensure var_names are strings

        # UMAP pipeline for generated data
        sc.pp.normalize_total(adata_gen_processed, target_sum=1e4)
        sc.pp.log1p(adata_gen_processed)

        try:
            n_top_genes_hvg_gen = min(2000, adata_gen_processed.shape[1] - 1 if adata_gen_processed.shape[1] > 1 else 1)
            if n_top_genes_hvg_gen > 0:
                print("Performing HVG selection for generated data with flavor 'cell_ranger'.")
                # use 'cell_ranger' flavor for HVG
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

                    # plot Generated Data UMAP
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

    # clean up the 'figures' directory if it's empty after moving
    if os.path.exists("./figures") and not os.listdir("./figures"):
        try:
            os.rmdir("./figures")
        except OSError:
            pass # directory might not be empty if other plots are saved there by scanpy
            
    print(f"Qualitative plotting for {model_name} finished.")
