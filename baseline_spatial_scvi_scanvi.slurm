#!/bin/sh
#SBATCH --job-name baseline_spatial_scvi_scanvi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --error run_baseline_spatial.e%j
#SBATCH --output run_baseline_spatial.o%j
#SBATCH --time 0-01:59:59

nvidia-smi

python3 baseline_spatial_scvi_scanvi.py \
    --h5ad data/visium_lymph_node.h5ad \
    --seed 77 \
    --train_frac 0.8 \
    --label_key cell_type \
    --scvi_epochs 200 \
    --scanvi_epochs 200 \
    --batch_size 8 \
    --output_base qualitative_evaluation_plots_v2
