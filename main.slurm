#!/bin/sh
#SBATCH --job-name LapDDPM         # this is a parameter to help you sort your job when listing it
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1
#SBATCH --partition=LocalQ
#SBATCH --error scrna_ddpm.%j.err     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output scrna_ddpm.%j.out      # optional. By default the error and output files are merged
#SBATCH --gres=gpu:1
#SBATCH --time 7-04:14:59                  # maximum run time.
#SBATCH --nodelist=kmer

## USED Python 3.8.18, lapddpm as env
source $HOME/.bashrc
nvidia-smi

# python -u run_pbmc3k.py
python -u run_denta.py
