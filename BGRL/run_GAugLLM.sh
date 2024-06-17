#!/bin/bash
#SBATCH -p sfscai
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --gres=gpu:1


#module purge
#module load miniconda
#source activate yfhpc


for dataset in 'pubmed'; do  # 'amazon-computers' 'amazon-photo' 'amazon-history' 'pubmed' 'ogbn-arxiv'
   for feature_type in 'GIA_FT_E'; do
       python -u new_struct.py --dataset $dataset --feature_type $feature_type --device 0 > out/new_attention_struct_${dataset}.txt 2>&1
   done
done
