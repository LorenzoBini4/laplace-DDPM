# LapDDPM: A Conditional Graph Diffusion Model for scRNA-seq Generation with Spectral Adversarial Perturbations

This repository contains the official PyTorch implementation for the paper: [**"LapDDPM: A Conditional Graph Diffusion Model for scRNA-seq Generation with Spectral Adversarial Perturbations"**](https://arxiv.org/abs/2506.13344), published at the ICML 2025 [GenBio Workshop: The Second Workshop on Generative AI and Biology, Vancouver](https://genbio-workshop.github.io/2025/).

## Abstract

Generating high-fidelity and biologically plausible synthetic single-cell RNA sequencing (scRNA-seq) data, especially with conditional control, is challenging due to its high dimensionality, sparsity, and complex biological variations. Existing generative models often struggle to capture these unique characteristics and ensure robustness to structural noise in cellular networks. We introduce LapDDPM, a novel conditional Graph Diffusion Probabilistic Model for robust and high-fidelity scRNA-seq generation. LapDDPM uniquely integrates graph-based representations with a score-based diffusion model, enhanced by a novel spectral adversarial perturbation mechanism on graph edge weights. Our contributions are threefold: we leverage Laplacian Positional Encodings (LPEs) to enrich the latent space with crucial cellular relationship information; we develop a conditional score-based diffusion model for effective learning and generation from complex scRNA-seq distributions; and we employ a unique spectral adversarial training scheme on graph edge weights, boosting robustness against structural variations. Extensive experiments on diverse scRNA-seq datasets demonstrate LapDDPM's superior performance, achieving high fidelity and generating biologically-plausible, cell-type-specific samples.

---

## Repository Structure

The repository is organized as follows:

```
.
├── data/                   # Directory for datasets
│   ├── pbmc3k_train.h5ad   # Example raw training data
│   ├── pbmc3k_test.h5ad    # Example raw test data
│   └── pbmc3k_processed/   # Directory for pre-processed data (auto-generated)
│       └── train_k.../
├── pbmc3k/                 # Python package for the PBMC3K dataset
│   ├── dataset.py          # Data loading and pre-processing logic
│   ├── models.py           # Model components (Encoder, Decoder, ScoreNet)
│   ├── utils.py            # Utility functions
│   └── ...
├── denta/                  # Python package for the Dentate Gyrus dataset
│   └── ...
├── results/                # Directory for storing results and checkpoints (e.g., metrics)
├── run_pbmc3k.py           # Main script to run experiments on PBMC3K
├── run_denta.py            # Main script to run experiments on Dentate Gyrus
├── main.sh                 # Example shell script for execution
├── main.slurm              # Example Slurm script for HPC execution
└── README.md               # This file
```

-   **`pbmc3k/`** and **`denta/`**: These are Python modules containing the core logic for their respective datasets, including data processing (`dataset.py`), model definitions (`models.py`), and helper functions (`utils.py`).
-   **`data/`**: This directory stores all datasets. Raw data (e.g., in `.h5ad` format) should be placed here. The `run_*.py` scripts will automatically process this data and save the results in corresponding `*_processed` subdirectories.
-   **`run_*.py`**: These are the main executable scripts for training and evaluating the LapDDPM model on a specific dataset. Hyperparameters are configured within these files.
-   **`main.sh` / `main.slurm`**: Example scripts demonstrating how to execute a run and redirect output to log files.

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

1.  Place your raw training and test datasets in the `data/` directory. The scripts are currently configured to read `.h5ad` files. For example:
    - `data/pbmc3k_train.h5ad`
    - `data/pbmc3k_test.h5ad`
    - `data/dentategyrus_train.h5ad`
    - `data/dentategyrus_test.h5ad`

2.  The `run_*.py` scripts will handle the rest. On the first run, the script will:
    -   Load the raw data.
    -   Filter genes based on the `GENE_THRESHOLD`.
    -   Construct a k-NN graph and compute Laplacian Positional Encodings (LPEs).
    -   Save the processed `torch_geometric.data.Data` object into a subdirectory within `data/<dataset_name>_processed/`. The subdirectory name is determined by the preprocessing hyperparameters (e.g., `k`, `pe_dim`, `pca_neighbors`). Subsequent runs with the same parameters will load the processed data directly, saving time.

---

## Training and Evaluation

The model can be trained and evaluated by running the corresponding script for your dataset.

### 1. Configure Hyperparameters

All hyperparameters for a run can be modified at the top of the relevant script (e.g., `run_pbmc3k.py`):

```python
# In run_pbmc3k.py
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 1500
HIDDEN_DIM = 1024
LATENT_DIM = 1024
PE_DIM = 50
K_NEIGHBORS = 50
# ... and so on
```

### 2. Run the Experiment

You can execute the script directly from the command line. The `main.sh` script provides a convenient way to do this and log the output:

```bash
bash main.sh
```

This will execute the `run_pbmc3k.py` script, and all standard output and errors will be saved to `scrna_ddpm.out` and `scrna_ddpm.err`, respectively.

To run the experiment for the Dentate Gyrus dataset, simply modify `main.sh` to call `run_denta.py`.

For execution on an HPC cluster using the Slurm workload manager, you can use the provided `main.slurm` script. Submit the job using `sbatch`:

```bash
sbatch main.slurm
```

This will execute the `run_pbmc3k.py` script, and all standard output and errors will be saved to `scrna_ddpm.out` and `scrna_ddpm.err`, respectively.

To run the experiment for the Dentate Gyrus dataset, simply modify `main.sh` to call `run_denta.py`.

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