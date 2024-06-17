for dataset in 'pubmed' ; do
    for feature_type  in 'attention'; do
        sh scripts/run_transductive.sh $dataset 0 $feature_type 2>&1 | tee out/Bacis_${dataset}.out
    done
done