#!/bin/bash
#SBATCH --job-name=VAE_celeba
#SBATCH --error=VAE_celeba.err
#SBATCH --output=VAE_celeba.out
#SBATCH --time=5-23:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=p_ps848
#SBATCH --mem=50000
source ~/.bashrc
conda activate xiaoran
cd ..

python ~/EOT/main.py --config "./config/VAE_celeba.yaml"
