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
source activate xiaoran

python ~/EOT/main.py --epochs 80 --seed 1 --save 'eotexp1'
python ~/EOT/main.py --epochs 80 --seed 2 --save 'eotexp2'
python ~/EOT/main.py --epochs 80 --seed 1 --latent_dim 3 --save 'eotexp3'
python ~/EOT/main.py --epochs 80 --seed 2 --latent_dim 3 --save 'eotexp4'
python ~/EOT/main.py --epochs 80 --seed 1 --save 'eotexp5' --sample_size 1500
python ~/EOT/main.py --epochs 80 --seed 2 --save 'eotexp6' --sample_size 1500
python ~/EOT/main.py --epochs 80 --seed 1 --latent_dim 3 --save 'eotexp7' --sample_size 1500
python ~/EOT/main.py --epochs 80 --seed 2 --latent_dim 3 --save 'eotexp8' --sample_size 1500
