#!/bin/bash
#SBATCH --job-name=VAE_g
#SBATCH --error=VAE_g.err
#SBATCH --output=VAE_g.out
#SBATCH --time=5-23:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=p_ps848
#SBATCH --mem=50000
source ~/.bashrc
conda activate xiaoran
cd ..

python ~/EOT/main.py --config "./config/VAE.yaml"
