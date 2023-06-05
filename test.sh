#! /bin/bash

#SBATCH --job-name=testing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-0
#SBATCH --partition batch_ugrad
#SBATCH --output=out1.txt
#SBATCH --error=err1.txt

python test.py --checkpoint ./logs/aging

exit 0