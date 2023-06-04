#! /bin/bash

#SBATCH --job-name=aging
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-0
#SBATCH --partition batch_ugrad
#SBATCH --output=out.txt
#SBATCH --error=err.txt

python train.py

exit 0