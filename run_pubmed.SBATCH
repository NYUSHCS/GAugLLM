#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=Expert_PubMed
#SBATCH --output=Expert_PubMed.out

module purge

cd /vast/yf2722/LLM4GCL_update/LLMs

singularity \
    exec --nv \
    --overlay /scratch/yf2722/new_env/my_pytorch.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
python -u trainModel.py --dataset pubmed --LLM_type 'GIA_NEW' 
"