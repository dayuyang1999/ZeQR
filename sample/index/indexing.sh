year=2020
method=dpr
for shard in $(seq 0 1 10) 

do
        echo "Now processing Shard {$shard}, year {$year}, method {$method}"
        mkdir your_directory_to_storage_index/index_$method/dense_$year/cast$year'_den'$shard
        python -m pyserini.encode input   --corpus your_directory_to_storage_jsonls/$year/jsonls \
                                        --fields text \
                                        --shard-id $shard \
                                        --shard-num 20 \
                                output  --embeddings your_directory_to_storage_index/index_$method/dense_$year/cast$year'_den'$shard \
                                        --to-faiss \
                                encoder --encoder facebook/dpr-reader_encoder-multiset-base \
                                        --device cuda:0 \
                                        --fields text \
                                        --batch 16 \
                                        --fp16

done