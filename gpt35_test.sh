# # IC-DST (Default)
# python run_GPT35_test.py \
#       --train_fn data/mw21_5p_train_v2.json \
#       --retriever_dir retriever/expts/5p_v2_ft_sep_entity_sim_0.1_9 \
#       --output_file_name gpt35_turbo_5p_v2_sep_entity_sim_0.1_9 \
#       --mwz_ver 2.4 \
#       --bracket \
#       --test_size 737

# IC-DST (with entity)
python run_GPT35_test.py \
      --train_fn ett_metadata/230606_1832--ner_best_f1_ett_meta/mw21_5p_train_v2_ner_best_f1_ett_meta.json \
      --test_fn ett_metadata/230606_1832--ner_best_f1_ett_meta/mw24_100p_test_ner_best_f1_ett_meta.json \
      --retriever_dir retriever/expts/mw21_5p_v2  \
      --output_file_name gpt35_turbo_5p_v2_with_ett \
      --mwz_ver 2.4 \
      --bracket \
      --test_size 100