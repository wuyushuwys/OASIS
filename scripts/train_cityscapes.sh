#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --job-name=oasis
#SBATCH --cpus-per-task=48
#SBATCH --partition=ce-mri
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=256
#SBATCH --output=%j.out

srun python -m torch.distributed.run --nproc_per_node 4 train.py  --name oasis_cityscapes_no_sn \
                      --dataset_mode cityscapes \
                      --gpu_ids 0,1,2,3 \
                      --dataroot ~/data/segmentation/cityscapes/data \
                      --batch_size 20  --no_spectral_norm --channels_G 64 --z_dim 64  --channels_D 64

