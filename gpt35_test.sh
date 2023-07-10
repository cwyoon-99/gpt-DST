# # IC-DST (Default)
python run_GPT35_test.py \
      --train_fn data/mw21_5p_train_v2.json \
      --test_fn data/mw24_100p_dev.json \
      --retriever_dir retriever/expts/mw21_5p_v2 \
      --output_file_name 5p_v2_dev_ex_num_6 \
      --mwz_ver 2.4 \
      --bracket \
      --num_ex 6 \
      --test_size -1

python run_GPT35_test.py \
      --train_fn data/mw21_5p_train_v2.json \
      --test_fn data/mw24_100p_dev.json \
      --retriever_dir retriever/expts/mw21_5p_v2 \
      --output_file_name 5p_v2_dev_ex_num_5 \
      --mwz_ver 2.4 \
      --bracket \
      --num_ex 5 \
      --test_size -1

python run_GPT35_test.py \
      --train_fn data/mw21_5p_train_v2.json \
      --test_fn data/mw24_100p_dev.json \
      --retriever_dir retriever/expts/mw21_5p_v2 \
      --output_file_name 5p_v2_dev_ex_num_4 \
      --mwz_ver 2.4 \
      --bracket \
      --num_ex 4 \
      --test_size -1

python run_GPT35_test.py \
      --train_fn data/mw21_5p_train_v2.json \
      --test_fn data/mw24_100p_dev.json \
      --retriever_dir retriever/expts/mw21_5p_v2 \
      --output_file_name 5p_v2_dev_ex_num_3 \
      --mwz_ver 2.4 \
      --bracket \
      --num_ex 3 \
      --test_size -1

# python run_GPT35_test.py \
#       --train_fn data/mw21_5p_train_v2.json \
#       --retriever_dir retriever/expts/mw21_5p_v2 \
#       --output_file_name 5p_v2_ex_num_5 \
#       --mwz_ver 2.4 \
#       --bracket \
#       --num_ex 5 \
#       --test_size 3685

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

# dynamic cluster
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
#       --test_size 3685