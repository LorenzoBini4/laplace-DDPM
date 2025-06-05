#!/bin/sh
#SBATCH --job-name LapDDPM         # this is a parameter to help you sort your job when listing it
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1
#SBATCH --partition=LocalQ
#SBATCH --error run_ddpm_v3.%j.err     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output run_ddpm_v3.%j.out      # optional. By default the error and output files are merged
#SBATCH --gres=gpu:1
#SBATCH --time 0-04:14:59                  # maximum run time.
#SBATCH --nodelist=kmer

## USED Python 3.8.18
source $HOME/.bashrc
nvidia-smi

# python -u lap_ddpm.py
# python -u cond_lap_ddpm.py
# python -u adv_cond_lap_ddpm.py 
# python -u delme.py 
# python -u scrna_cond_ddpm.py 
python -u scrna_cond_ddpm_v3.py 
# python -u scrna_cond_ddpm_viz.py 
# python -u scrna_graphddpm.py 
# python -u scrna_cond_ddpm_v2_denta.py 