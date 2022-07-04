#!/bin/bash
#SBATCH --job-name=OT_g
#SBATCH --error=OT_g.err
#SBATCH --output=OT_g.out
#SBATCH --time=5-23:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=p_ps848
#SBATCH --mem=50000
source ~/.bashrc
conda activate xiaoran
cd ..

python ~/EOT/main2.py --config "./config/OT.yaml" --pretrain "./savedmodels/Gaussian_decoder.pth"
