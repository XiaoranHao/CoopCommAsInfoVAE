#!/bin/bash
#SBATCH --job-name=OT_mnist
#SBATCH --error=OT_mnist.err
#SBATCH --output=OT_mnist.out
#SBATCH --time=5-23:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=p_ps848
#SBATCH --mem=50000
source ~/.bashrc
conda activate xiaoran
cd ..

python ~/EOT/main2.py --config "./config/OT_mnist.yaml" --pretrain "./savedmodels/mnist_decoder.pth"
