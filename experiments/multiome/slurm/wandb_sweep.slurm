#!/bin/sh
#SBATCH --job-name wandb_multiome
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --error wandb_sweep_multiome.e%j
#SBATCH --output wandb_sweep_multiome.o%j
#SBATCH --time 0-12:00:00

nvidia-smi
REPO_ROOT="/home/lbini/cmu/works/laplace-DDPM"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/experiments:$PYTHONPATH"
export SWEEP_EXPERIMENT=multiome

wandb agent lorenzo1729/laplace-DDPM/lxl4lg89
