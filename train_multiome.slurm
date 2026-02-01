#!/bin/sh
#SBATCH --job-name inpaint_lap_multi          
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4            
#SBATCH --mem-per-cpu=64G 
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --error run_train_multiome.e%j     
#SBATCH --output run_train_multiome.o%j      
#SBATCH --time 0-00:59:59

nvidia-smi

python3 main_multiome.py --seed 77 --batch_size 8 --hidden_dim 512 --latent_dim 256 --lr 1e-4 \
    --epochs 2000 --warmup 100 \
    --dual_graph --input_masking_fraction 0.2 \
    --loss_weight_diff 1.0 --loss_weight_kl 0.01 --loss_weight_rec 1.0 --spectral_loss_weight 0.2 --mask_rec_weight 3.0 \
    --batch_adv_weight 0.1 --context_align_weight 0.1 --mask_strategy high_var \
    --kl_anneal_steps 2000 --kl_free_bits 0.05 \
    --encoder_type gnn \
    --use_scvi_context --scvi_local_dir scvi_hub_cache/models--scvi-tools--tabula-sapiens-blood-scvi/snapshots/24aaac3d6a67620280c0438c046737c88423f4e7 \
    --knn 5 --pe_dim 50 --pca_dim 50 --gene_threshold 20 --timesteps_diffusion 1000 --viz True \
    --max_time 1.0 --num_gnn_layers 3 --num_mlp_layers 3
