#!/bin/bash
#SBATCH --job-name=ppo_opt_1
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:a100:1                 # number of gpus
#SBATCH --time 15:00:00              # maximum execution time (HH:MM:SS)
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
# srun --cpus-per-task=8 --gres=gpu:1 --mem=64GB --time=2:00:00 
torchrun ./train_prompts_1.py --prompt_dataset ./ppo/data/ppo_dataset1.jsonl --pretrain_dataset ./reward_model/ppo_pretrain.jsonl\
    --model opt --pretrain "facebook/opt-2.7b" --checkpoint ./sft/model/output --rm_model opt --rm_path ./reward_model2/model/50epoch --save_path ./ppo/model/new/actor_checkpoint_prompts\
    --num_episode 40 --num_collect_steps 10 --lora_rank 8 --ptx_coef 0
# torchrun ./train_prompts_1.py --prompt_dataset ./ppo/data/ppo_dataset1.jsonl --pretrain_dataset ./reward_model/ppo_pretrain.jsonl\
#     --model opt --pretrain "facebook/opt-2.7b" --checkpoint ./sft/model/output --rm_model opt --rm_path ./reward_model2/model/50epoch --save_path ./ppo/model/new/actor_checkpoint_prompts\
#     --num_episode 40 --num_collect_steps 10 --ptx_coef 0