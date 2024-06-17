for dataset in   'amazon-history' ; do #'ogbn-arxiv' 'cora' 'pubmed' 
    for LLM_type in  'GIA'  ; do # 'MLM' 'GIA' 
        python trainModel.py --dataset $dataset  --LLM_type $LLM_type 2>&1 | tee out/${dataset}.out &
    done 
done    
wait

