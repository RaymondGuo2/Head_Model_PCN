#!/bin/bash
#SBATCH --job-name=test_pcn_0408
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rqg23

/usr/bin/nvidia-smi
uptime

python3 test.py \
 --test_data=$HOME/../../vol/bitbucket/rqg23/project_data \
 --checkpoint=./checkpoint/train_0408_1056.pth \
 --batch_size=32\
 --device='cuda' \
 --step_ratio=2