#!/bin/sh
#SBATCH --job-name gen_lap_spatial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --error run_gen_spatial.e%j
#SBATCH --output run_gen_spatial.o%j
#SBATCH --time 0-00:59:59

nvidia-smi
REPO_ROOT="./"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/experiments:$PYTHONPATH"

conda run -n lrgb python3 experiments/spatial/main_spatial.py --eval_only \
    --checkpoint_path data/spatial_processed/train_k30/trainer_final_state.pt \
    --use_scvi_context --scvi_local_dir scvi_hub_cache/models--scvi-tools--tabula-sapiens-blood-scvi/snapshots/24aaac3d6a67620280c0438c046737c88423f4e7 \
    --hidden_dim 512 --latent_dim 256 \
    --knn 30 --pe_dim 50 --pca_dim 50 --gene_threshold 20 --timesteps_diffusion 1000 --viz True
