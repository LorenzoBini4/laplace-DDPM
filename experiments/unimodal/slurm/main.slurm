#!/bin/sh
#SBATCH --job-name LapDDPM        
#SBATCH --ntasks 1                
#SBATCH --cpus-per-task 1
#SBATCH --partition=LocalQ
#SBATCH --error scrna_ddpm.%j.err     
#SBATCH --output scrna_ddpm.%j.out     
#SBATCH --gres=gpu:1
#SBATCH --time 7-04:14:59                  
#SBATCH --nodelist=kmer

## USED Python 3.8.18, lapddpm as env
source $HOME/.bashrc
nvidia-smi

REPO_ROOT="./"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/experiments:$PYTHONPATH"

python -u experiments/unimodal/run_pbmc3k.py
# python -u experiments/unimodal/run_denta.py
