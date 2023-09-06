#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:a100:1                 # number of gpus
#SBATCH --time 7:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.out            # error file name (same to watch just one file)

module purge
source $HOME/.bashrc
export PATH=/scratch/yl9315/miniconda3/bin:$PATH
module load cuda/11.6.2
conda activate coati1

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_PORT=6000
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# to debug - add echo (it exits and prints what it would have launched)
python ./val_sft.py
