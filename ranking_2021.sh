python3 dense_ranking.py \
    --run_name 'CAsT2021' \
    --query_file_dir "CAsT/query_file/query_file/data2021_canon1.json" \
    --reformulator 'mrc' \
    --year '2021' \
    --device '0' \
    --reformulator_device 0 \
    --query_encoder 'castorini/tct_colbert-v2-hnp-msmarco' \
    --index_base_dir 'index/' \
    --search_in_qa_topk 5