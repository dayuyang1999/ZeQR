python3 dense_ranking.py \
    --run_name 'CAsT2022' \
    --query_file_dir "CAsT/query_file/data2022_canon1_oriq_incontext.json" \
    --reformulator 'mrc' \
    --year '2022' \
    --device '1' \
    --reformulator_device 1 \
    --query_encoder 'castorini/tct_colbert-v2-hnp-msmarco' \
    --index_base_dir 'index/' \
    --search_in_qa_topk 5