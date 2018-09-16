#!/bin/bash
#SBATCH --output=01.out
#SBATCH --error=01.err
#SBATCH --account=ece-gpu-high
#SBATCH -p ece-gpu-high --gres=gpu:1
#SBATCH -c 6
srun singularity exec --nv ~dec18/Containers/tfgpu.simg python 01_on_cluster.py
