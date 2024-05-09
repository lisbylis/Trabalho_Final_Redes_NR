#!/bin/bash
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source ~/.bashrc
module load cuda/12.2
conda activate ilumpy

python -u trabalho_final_optuna_gpu.py > trabalho_final_optuna_gpu.out

