# # IC-DST (Default)
# python run_GPT35_test.py \
#       --train_fn data/mw21_5p_train_v2.json \
#       --test_fn data/mw24_100p_dev.json \
#       --retriever_dir retriever/expts/mw21_5p_v2 \
#       --output_file_name 5p_v2_dev_ex_num_3 \
#       --mwz_ver 2.4 \
#       --bracket \
#       --num_ex 3 \
#       --test_size -1

# cluster
# python run_GPT35_test_cluster.py \
#       --train_fn data/mw21_5p_train_v2.json \
#       --retriever_dir retriever/expts/mw21_5p_v2 \
#       --output_file_name 5p_v2_agglo_cluster_0.5 \
#       --mwz_ver 2.4 \
#       --bracket \
#       --num_cand 10 \
#       --num_ex 5 \
#       --test_size 3685


python run_GPT35_test_cluster.py \
      --train_fn data/mw21_5p_train_v2.json \
      --retriever_dir retriever/expts/mw21_5p_v2 \
      --output_file_name 5p_v2_domain_select \
      --mwz_ver 2.4 \
      --bracket \
      --domain_select \
      --num_select 10 \
      --test_size -1s

# # dynamic cluster
# python run_GPT35_test_cluster.py \
#       --train_fn data/mw21_5p_train_v2.json \
#       --retriever_dir retriever/expts/mw21_5p_v2 \
#       --output_file_name 5p_v2_agglo_cluster_0.5_dynamic_3_to_7 \
#       --mwz_ver 2.4 \
#       --bracket \
#       --num_cand 10 \
#       --num_ex 5 \
#       --num_max 7 \
#       --num_min 3 \
#       --test_size -1