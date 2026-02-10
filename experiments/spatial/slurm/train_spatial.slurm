#!/bin/sh
#SBATCH --job-name inpaint_lap_spatial          
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4            
#SBATCH --mem-per-cpu=32G 
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --error run_train_spatial.e%j     
#SBATCH --output run_train_spatial.o%j      
#SBATCH --time 0-01:59:59

nvidia-smi
REPO_ROOT="./"
echo "REPO_ROOT=$REPO_ROOT"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/experiments:$PYTHONPATH"

python3 "$REPO_ROOT/experiments/spatial/main_spatial.py" --seed 77 --batch_size 8 --hidden_dim 512 --latent_dim 256 --lr 4e-4 \
    --epochs 10000 --warmup 1000 --force_reprocess --stage2_start_epoch 160 \
    --input_masking_fraction 0.0 \
    --loss_weight_diff 0.5 --loss_weight_kl 0.05 --loss_weight_rec 3.0 --spectral_loss_weight 0.0 --mask_rec_weight 0.0 \
    --batch_adv_weight 0.0 --context_align_weight 0.0 --mask_strategy random \
    --gene_stats_weight 1.5 --zero_rate_weight 0.25 --cell_zero_rate_weight 0.5 --cell_type_loss_weight 0.5 --dispersion_weight 0.2 \
    --pi_temperature 1.0 --pi_bias -0.2 --pi_blend 0.005 --diffusion_temperature 1.0 --diffusion_method sde \
    --libsize_weight 2.0 --gen_calibrate_means --gen_calibrate_zero --gen_match_zero_per_cell --gen_zero_per_cell_strength 0.8 \
    --gen_mean_scale_clip_low 0.1 --gen_mean_scale_clip_high 10.0 \
    --cell_type_path data/labels/spatial_cell_types.tsv \
    --gene_stats_anneal_steps 200 --zero_rate_anneal_steps 200 \
    --early_stop_gene_stats --early_stop_patience 2000 --early_stop_min_delta 1e-5 \
    --kl_anneal_steps 200 --kl_free_bits 0.5 \
    --encoder_type transformer --transformer_layers 4 --transformer_heads 8 --transformer_dropout 0.1 \
    --knn 30 --pe_dim 50 --pca_dim 50 --gene_threshold 20 --timesteps_diffusion 1000 --viz True \
    --max_time 1.0
