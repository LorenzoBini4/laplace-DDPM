#!/bin/sh
#SBATCH --job-name gen_lap_multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --error run_gen_multiome.e%j
#SBATCH --output run_gen_multiome.o%j
#SBATCH --time 0-00:59:59

nvidia-smi

conda run -n lrgb python3 main_multiome.py --eval_only \
    --checkpoint_path data/multiome_processed/train_k5/trainer_final_state.pt \
    --use_scvi_context --scvi_local_dir scvi_hub_cache/models--scvi-tools--tabula-sapiens-blood-scvi/snapshots/24aaac3d6a67620280c0438c046737c88423f4e7 \
    --hidden_dim 512 --latent_dim 256 \
    --knn 5 --pe_dim 50 --pca_dim 50 --gene_threshold 20 --timesteps_diffusion 1000 --viz True
