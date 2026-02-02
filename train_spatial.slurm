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
python3 main_spatial.py --seed 77 --batch_size 8 --hidden_dim 512 --latent_dim 256 --lr 4e-4 \
    --epochs 200 --warmup 10 \
    --dual_graph --input_masking_fraction 0.2 \
    --loss_weight_diff 0.01 --loss_weight_kl 0.0 --loss_weight_rec 1.0 --spectral_loss_weight 0.2 --mask_rec_weight 0.1 --disable_kl \
    --batch_adv_weight 0.5 --context_align_weight 0.5 --mask_strategy high_var \
    --gene_stats_weight 2.0 --zero_rate_weight 2.0 --cell_type_loss_weight 0.5 --dispersion_weight 0.5 \
    --pi_temperature 1.0 --pi_bias -0.3 --pi_blend 0.4 --diffusion_temperature 0.8 \
    --gene_stats_anneal_steps 2000 --zero_rate_anneal_steps 2000 \
    --early_stop_gene_stats --early_stop_patience 2000 --early_stop_min_delta 1e-4 \
    --kl_anneal_steps 2000 --kl_free_bits 0.2 \
    --encoder_type gnn \
    --use_scvi_context --scvi_local_dir scvi_hub_cache/models--scvi-tools--tabula-sapiens-blood-scvi/snapshots/24aaac3d6a67620280c0438c046737c88423f4e7 \
    --knn 30 --pe_dim 50 --pca_dim 50 --gene_threshold 20 --timesteps_diffusion 1000 --viz True \
    --max_time 1.0 --num_gnn_layers 3 --num_mlp_layers 3
