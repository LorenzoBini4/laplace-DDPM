#!/bin/sh
#SBATCH --job-name sweep_spatial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --error run_sweep_spatial.e%j
#SBATCH --output run_sweep_spatial.o%j
#SBATCH --time 0-23:59:59
#SBATCH --array=0-2

nvidia-smi

REPO_ROOT="./"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/experiments:$PYTHONPATH"

BASE_ARGS="--seed 77 --batch_size 8 --hidden_dim 512 --latent_dim 256 --lr 1e-4 \
  --epochs 2000 --warmup 200 --stage2_start_epoch 1200 \
  --input_masking_fraction 0.0 \
  --loss_weight_diff 0.5 --loss_weight_kl 0.05 --loss_weight_rec 3.0 --spectral_loss_weight 0.0 --mask_rec_weight 0.0 \
  --batch_adv_weight 0.0 --context_align_weight 0.0 --mask_strategy random \
  --gene_stats_weight 1.0 --zero_rate_weight 0.25 --cell_type_loss_weight 1.0 --dispersion_weight 0.1 \
  --pi_bias -0.2 --pi_blend 0.0 --diffusion_temperature 1.0 --diffusion_method sde \
  --libsize_weight 2.0 --gen_calibrate_means --gen_calibrate_zero \
  --gen_mean_scale_clip_low 0.1 \
  --gene_stats_anneal_steps 800 --zero_rate_anneal_steps 800 \
  --early_stop_gene_stats --early_stop_patience 200 --early_stop_min_delta 1e-4 \
  --kl_anneal_steps 800 \
  --encoder_type transformer --transformer_layers 4 --transformer_heads 8 --transformer_dropout 0.1 \
  --knn 30 --pe_dim 50 --pca_dim 50 --gene_threshold 20 --timesteps_diffusion 1000 --viz True \
  --max_time 1.0 --cell_type_path data/labels/spatial_cell_types.tsv"

TAGS=(
  "cellzero0.1_pitemp1.5_gzpc0.2_kfb0.5_clip10"
  "cellzero0.01_pitemp2.5_gzpc0.08_kfb1.5_clip20"
  "cellzero0.1_pitemp1.5_gzpc0.02_kfb2.5_clip30"
)
CELL_ZERO=(0.1 0.01 0.1)
PI_TEMP=(1.5 2.5 1.5)
GZPC=(0.2 0.08 0.02)
KFB=(0.5 1.5 2.5)
CLIP_HI=(10.0 20.0 30.0)

IDX=${SLURM_ARRAY_TASK_ID:-0}
TAG=${TAGS[$IDX]}

echo "Running spatial sweep: $TAG (index $IDX)"
python3 experiments/spatial/main_spatial.py ${BASE_ARGS} \
  --cell_zero_rate_weight ${CELL_ZERO[$IDX]} \
  --pi_temperature ${PI_TEMP[$IDX]} \
  --gen_match_zero_per_cell --gen_zero_per_cell_strength ${GZPC[$IDX]} \
  --kl_free_bits ${KFB[$IDX]} \
  --gen_mean_scale_clip_high ${CLIP_HI[$IDX]} \
  --plot_tag ${TAG}
