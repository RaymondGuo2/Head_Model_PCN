#!/bin/bash
#SBATCH --job-name=train_pcn_0408_morning
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rqg23

/usr/bin/nvidia-smi
uptime

python3 train.py \
 --train_data=$HOME/../../vol/bitbucket/rqg23/project_data \
 --val_data=$HOME/../../vol/bitbucket/rqg23/project_data \
 --checkpoint=./checkpoint \
 --epochs=1000 \
 --batch_size=32\
 --generator_learning_rate=1e-4 \
 --discriminator_learning_rate=1e-4 \
 --input_num_points=2048 \
 --gt_num_points=2048 \
 --device='cuda' \
 --rec_weight=200 \
 --step_ratio=2
