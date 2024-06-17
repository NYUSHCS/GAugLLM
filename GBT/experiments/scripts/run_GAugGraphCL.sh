#!/bin/bash
#SBATCH -p sfscai
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --gres=gpu:1


#module purge
#module load miniconda
#source activate yfhpc


for dataset in 'amazon-history'; do  # 'amazon-computers' 'amazon-photo' 'amazon-history' 'pubmed' 'ogbn-arxiv'
   for feature_type in 'GIA_text'; do
       python -u train_transductive_struct.py --dataset $dataset --feature_type $feature_type --loss_name 'GraphCL_loss' --device 0 > out/Basic_${dataset}_GraphCL.txt 2>&1
   done
done