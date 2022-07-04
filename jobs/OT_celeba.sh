#!/bin/bash
#SBATCH --job-name=OT_celeba
#SBATCH --error=OT_celeba.err
#SBATCH --output=OT_celeba.out
#SBATCH --time=5-23:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=p_ps848
#SBATCH --mem=50000
#SBATCH --mail-user=xiaoranhao1012@gmail.com
#SBATCH --mail-type=ALL
source ~/.bashrc
conda activate xiaoran
cd ..

python ~/EOT/main2.py --config "./config/OT_celeba.yaml" --pretrain "./savedmodels/celeba64_decoder.pth"
