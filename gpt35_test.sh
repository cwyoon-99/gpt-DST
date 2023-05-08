python run_GPT35_test.py \
      --train_fn data/mw21_5p_train_v2.json \
      --retriever_dir retriever/expts/mw21_5p_v2 \
      --output_file_name gpt35_turbo_5p_v2_baseline_no_blank_blanket  \
      --mwz_ver 2.4 \
      --bracket \

# python run_GPT35_follow_up_CoT.py \
#       --train_fn data/mw21_5p_train_v2.json \
#       --retriever_dir retriever/expts/mw21_5p_v2 \
#       --output_file_name gpt35_turbo_5p_v2_follow_up_CoT  \
#       --mwz_ver 2.4 \
#       --test_size 50 \
#       --bracket \
#       --load_result_dir expts/230419_0327-gpt35_turbo_5p_v2_custom_prompt_0to368 \