#!/bin/sh
#SBATCH --job-name label_transfer_spatial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 12
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --error run_label_transfer_spatial.e%j
#SBATCH --output run_label_transfer_spatial.o%j
#SBATCH --time 0-03:59:59

nvidia-smi

REPO_ROOT="./"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/experiments:$PYTHONPATH"

python3 experiments/label_transfer/label_transfer.py \
  --reference_h5ad data/raw/hlca_core.h5ad \
  --query_path data/raw/visium_lymph_node.h5ad \
  --label_key cell_type \
  --output_csv data/labels/spatial_cell_types.tsv \
  --output_tsv \
  --reference_ncells 100000 \
  --n_latent 30 \
  --max_epochs_ref 50 \
  --max_epochs_scanvi 20 \
  --max_epochs_query 20 \
  --batch_size 1024 \
  --num_workers 8 \
  --pin_memory
