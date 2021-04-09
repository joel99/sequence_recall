#!/bin/bash
#SBATCH --job-name=seq_recall
#SBATCH --gres gpu:1
#SBATCH -p short,overcap
#SBATCH -A overcap
#SBATCH --output=/srv/flash1/jye72/projects/sequence_recall/slurm_logs/train-%j.out
#SBATCH --error=/srv/flash1/jye72/projects/sequence_recall/slurm_logs/train-%j.err
all_args=("$@")
echo ${all_args[@]}
python -u train.py ${all_args[@]}
