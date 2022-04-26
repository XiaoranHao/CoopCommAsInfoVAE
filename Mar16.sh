#!/bin/bash
#SBATCH --job-name=EOT
#SBATCH --error=EOT.err
#SBATCH --output=EOT.out
#SBATCH --time=1-23:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=p_ps848
#SBATCH --mem=40000
source ~/.bashrc
conda activate xiaoran

python ~/EOT/main.py --epochs 80 --seed 1 --batch_size 1024 --sample_size 6000 --save 'eotexp1'
python ~/EOT/main.py --epochs 80 --seed 1 --batch_size 1024 --sample_size 6000 --latent_dim 3 --save 'eotexp2'
python ~/EOT/main.py --epochs 80 --seed 1 --batch_size 1024 --sample_size 6000 --latent_dim 6 --save 'eotexp3'
python ~/EOT/main.py --epochs 80 --seed 1 --batch_size 1024 --sample_size 10000 --latent_dim 6 --save 'eotexp4'
