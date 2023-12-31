python3 dense_ranking.py \
    --run_name 'CAsT2019' \ 
    --query_file_dir "CAsT/query_file/data2019_canon0_oriq_incontext.json" \
    --reformulator 'mrc' \
    --year '2019' \
    --device '0' \
    --reformulator_device '0' \
    --query_encoder 'castorini/tct_colbert-v2-hnp-msmarco' \
    --index_base_dir 'index/' \
    --search_in_qa_topk 5 #'castorini/tct_colbert-v2-hnp-msmarco'