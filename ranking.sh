#!/bin/bash
year=$1

if [ "$year" == "2019" ]; then
  canonical_psg_num=0
else
  canonical_psg_num=1
fi

python3 dense_ranking.py \
    --run_name CAsT$year \ 
    --query_file_dir CAsT/query_file/data${year}_canon${canonical_psg_num_oriq_incontext}.json \
    --reformulator 'mrc' \
    --year $year \
    --device 0 \
    --reformulator_device 0 \
    --query_encoder 'castorini/tct_colbert-v2-hnp-msmarco' \
    --index_base_dir 'index/' \
    --search_in_qa_topk 5 