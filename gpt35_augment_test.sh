python run_GPT35_test_augment.py \
      --train_fn data/mw21_5p_train_v2.json \
      --retriever_dir retriever/expts/mw21_5p_v2 \
      --output_file_name gpt35_turbo_5p_v2_retrieve_with_element2aug  \
      --mwz_ver 2.4 \
      --test_size 50 \
      --bracket \
      --ag_fn augments/230517_2222-augment_element_2/augment_log.json \
      --ag_search_index retriever/expts/fullshot_with_element2 \