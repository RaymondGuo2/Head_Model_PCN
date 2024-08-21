#!/bin/bash
#SBATCH --job-name=test_pcn_1108
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rqg23

/usr/bin/nvidia-smi
uptime

python3 test.py \
 --test_data=$HOME/../../vol/bitbucket/rqg23/project_data \
 --checkpoint=./checkpoint/train_1108_long.pth \
 --batch_size=32\
 --device='cuda:0' \
 --step_ratio=2