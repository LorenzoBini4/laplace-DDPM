#!/bin/sh
#SBATCH --job-name laplace_spatial          
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4            
#SBATCH --mem-per-cpu=32G 
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1 
#SBATCH --error run_train_spatial.e%j     
#SBATCH --output run_train_spatial.o%j      
#SBATCH --time 0-07:59:59

nvidia-smi

python3 main_spatial.py --seed 77 --batch_size 8 --hidden_dim 512 --latent_dim 512 --lr 1e-4 \
    --epochs 20 --warmup 1 \
    --loss_weight_diff 0.005 --loss_weight_kl 0.5 --loss_weight_rec 0.005 --input_masking_fraction 0.1 \
    --knn 30 --pe_dim 50 --pca_dim 50 --gene_threshold 20 --timesteps_diffusion 1000 --viz True \
    --num_gnn_layers 3 --num_mlp_layers 3
