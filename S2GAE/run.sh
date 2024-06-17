for dataset in   'amazon-photo' ; do #'Cora' 'Pubmed' 'arxiv'; do
    for feature_type in 'attention'; do 
        python -u s2gae_nc_acc.py --dataset $dataset --feature_type $feature_type --eval_multi_k 2>&1 | tee out/Basic_${dataset}_S2GAE.txt
    done
done