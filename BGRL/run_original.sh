#!/bin/bash
#SBATCH -p sfscai
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --gres=gpu:1


#module purge
#module load miniconda
#source activate yfhpc


for dataset in 'amazon-photo'; do  # 'amazon-computers' 'amazon-photo' 'amazon-history' 'pubmed' 'ogbn-arxiv'
   for feature_type in 'attention'; do
       python -u train_transductive_struct.py --dataset $dataset --device 0 > original_${dataset}.txt 2>&1
   done
done
