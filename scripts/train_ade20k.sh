#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --job-name=oasis
#SBATCH --cpus-per-task=48
#SBATCH --partition=ce-mri
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=256
#SBATCH --output=%j.out

srun python train.py  --name oasis_ade20k_small \
                      --dataset_mode ade20k
                      --gpu_ids 0,1,2,3 \
                      --dataroot ~/data/segmentation/ade20k/ \
                      --batch_size 32 \
                      --no_spectral_norm --channels_G 16 --z_dim 16
