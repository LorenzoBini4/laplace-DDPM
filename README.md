# LapDDPM: A Conditional Graph Diffusion Model for scRNA-seq Generation with Spectral Adversarial Perturbations

This repository contains the official PyTorch implementation for the paper: [**"LapDDPM: A Conditional Graph Diffusion Model for scRNA-seq Generation with Spectral Adversarial Perturbations"**](https://arxiv.org/pdf/2506.13344), published at the ICML 2025 [GenBio Workshop: The Second Workshop on Generative AI and Biology, Vancouver](https://genbio-workshop.github.io/2025/).

## Abstract

Generating high-fidelity and biologically plausible synthetic single-cell RNA sequencing (scRNA-seq) data, especially with conditional control, is challenging due to its high dimensionality, sparsity, and complex biological variations. Existing generative models often struggle to capture these unique characteristics and ensure robustness to structural noise in cellular networks. We introduce LapDDPM, a novel conditional Graph Diffusion Probabilistic Model for robust and high-fidelity scRNA-seq generation. LapDDPM uniquely integrates graph-based representations with a score-based diffusion model, enhanced by a novel spectral adversarial perturbation mechanism on graph edge weights. Our contributions are threefold: we leverage Laplacian Positional Encodings (LPEs) to enrich the latent space with crucial cellular relationship information; we develop a conditional score-based diffusion model for effective learning and generation from complex scRNA-seq distributions; and we employ a unique spectral adversarial training scheme on graph edge weights, boosting robustness against structural variations. Extensive experiments on diverse scRNA-seq datasets demonstrate LapDDPM's superior performance, achieving high fidelity and generating biologically-plausible, cell-type-specific samples.

---

## Repository Structure (Current)

```
.
├── experiments/                 # All experiment code, slurm, logs, plots
│   ├── spatial/                 # Spatial (Visium) runs
│   ├── multiome/                # Multiome runs
│   ├── baselines/               # scVI/scANVI and MultiVI baselines
│   ├── label_transfer/          # Label transfer utilities and jobs
│   ├── sweeps/                  # Hyperparameter sweeps
│   ├── unimodal/                # PBMC3K / Dentate runs
│   └── scripts/                 # Plotting / figure scripts
├── data/
│   ├── raw/                     # Raw datasets (.h5ad, .h5)
│   ├── processed/               # Preprocessed graph data
│   └── labels/                  # Transferred cell-type labels
├── paper/CHIL/                  # Paper sources and figures
└── README.md
```

- **`experiments/*/slurm`** contains SLURM entrypoints.
- **`experiments/*/logs`** contains run logs.
- **`experiments/qualitative_evaluation_plots_v2`** contains generated plots.
- **`data/raw`** holds Visium + Multiome files.
- **`data/processed`** holds cached graph data.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/LorenzoBini4/laplace-DDPM.git
    cd laplace-DDPM
    ```

 2. **Create and activate the Conda environment:**
    You may install our dedicated Conda environment directly from the .yml file. This may take a few minutes.
    ```bash
    conda env create -f lapddpm.yml
    ```
    Once the environment is created, activate it:
    ```bash
    conda activate lapddpm

3.  **Install dependencies:**
    The code is built on PyTorch and PyTorch Geometric. We recommend using a virtual environment (like `conda` or `venv`). The required libraries can be installed via `pip`.

    ```bash
    pip install torch torch-geometric torch-scatter scanpy anndata pandas numpy scikit-learn scipy pot
    ```
    *Note: Ensure your `torch` installation is compatible with your CUDA version if you plan to use a GPU.*

---

## Data Preparation

1. Place raw datasets in `data/raw/`:
   - `data/raw/visium_lymph_node.h5ad`
   - `data/raw/lymph_node_lymphoma_14k_raw_feature_bc_matrix.h5`

2. Scripts handle preprocessing. On first run they will:
    -   Load the raw data.
    -   Filter genes based on the `GENE_THRESHOLD`.
    -   Construct a k-NN graph and compute Laplacian Positional Encodings (LPEs).
    - Save processed `torch_geometric.data.Data` under `data/processed/<dataset>/train_k...`.

---

## Training and Evaluation

Run via SLURM scripts in `experiments/*/slurm` (recommended):

```bash
sbatch experiments/spatial/slurm/train_spatial.slurm
sbatch experiments/multiome/slurm/train_multiome.slurm
```

Label transfer (for cell types):
```bash
sbatch experiments/label_transfer/slurm/label_transfer_spatial.slurm
sbatch experiments/label_transfer/slurm/label_transfer_multiome.slurm
```

Baselines:
```bash
sbatch experiments/baselines/spatial/slurm/baseline_spatial_scvi_scanvi.slurm
sbatch experiments/baselines/multiome/slurm/baseline_multiome_multivi.slurm
```

### 3. Output

-   **Logs**: Training progress, including epoch-wise losses and learning rate, will be printed to the console (and captured in the log file).
-   **Checkpoints**: The trainer state (model weights, optimizer state, etc.) is saved periodically during training. Checkpoints are stored in the processed data directory (e.g., `data/pbmc3k_processed/train_k.../trainer_checkpoint_epoch_1500.pt`).
-   **Evaluation**: After training is complete, the script automatically:
    -   Loads the held-out test set.
    -   Generates synthetic data conditioned on the test set's cell types.
    -   Computes and prints evaluation metrics (MMD and 2-Wasserstein distance) comparing the real and generated data distributions.
<!-- -   **Qualitative Plots**: If `VIZ = True` is set in the script, UMAP plots comparing the real and generated data will be saved to a `qualitative_evaluation_plots_v2/` directory. -->

---

## Citation

If you use this code or our work in your research, please cite our paper:

```bibtex
@article{bini2025lapddpm,
      title={LapDDPM: A Conditional Graph Diffusion Model for scRNA-seq Generation with Spectral Adversarial Perturbations},
      author={Lorenzo Bini and Stéphane Marchand-Maillet},
      year={2025},
      eprint={2506.13344},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contact

For any questions or inquiries, please contact the authors by email or by opening an issue.
