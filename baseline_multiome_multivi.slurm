#!/bin/sh
#SBATCH --job-name baseline_multiome_multivi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --error run_baseline_multiome.e%j
#SBATCH --output run_baseline_multiome.o%j
#SBATCH --time 0-07:59:59

nvidia-smi

python3 baseline_multiome_multivi.py \
    --h5 data/lymph_node_lymphoma_14k_raw_feature_bc_matrix.h5 \
    --seed 77 \
    --train_frac 0.8 \
    --epochs 200 \
    --batch_size 2 \
    --output_base qualitative_evaluation_plots_v2
