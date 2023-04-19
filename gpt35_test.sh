python run_GPT35_test.py \
      --train_fn data/mw21_5p_train_v2.json \
      --retriever_dir retriever/expts/mw21_5p_v2 \
      --output_file_name gpt35_turbo_5p_v2_slot_classify_with_bracket  \
      --mwz_ver 2.4 \
      --test_size 193 \
      --slot_classify

# python run_GPT35_test_random_retrieve.py \
#       --train_fn data/mw21_5p_train_v2.json \
#       --retriever_dir retriever/expts/mw21_5p_v2 \
#       --output_dir expts/gpt35_turbo_5p_v2_ours_random_retreive  \
#       --mwz_ver 2.4

# python run_GPT35_test_ZeroCoT.py \
#       --train_fn data/mw21_5p_train_v2.json \
#       --retriever_dir retriever/expts/mw21_5p_v2 \
#       --output_file_name gpt35_turbo_5p_v2_ZeroCoT_slotlabel_provide  \
#       --save_interval 2 \
#       --mwz_ver 2.4

# python run_GPT35_test_two_stage_prediction.py \
#       --train_fn data/mw21_5p_train_v2.json \
#       --retriever_dir retriever/expts/mw21_5p_v2 \
#       --output_file_name gpt35_turbo_5p_v2_two_stage_prediction  \
#       --save_interval 2 \
#       --mwz_ver 2.4