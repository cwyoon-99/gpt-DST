# python run_GPT35_test_slot_predict.py \
#       --train_fn data/mw21_5p_train_v2.json \
#       --retriever_dir retriever/expts/mw21_5p_v2 \
#       --output_file_name gpt35_turbo_5p_v2_ner_onto_def2 \
#       --mwz_ver 2.4 \
#       --test_size 50 \
#       --bracket \
#       --ett_fn ett_replace/230601_2222-1.0-ner_onto_def/mw21_5p_train_v2_ner_onto_def.json \
#       --ett_retriever_dir retriever/expts/5p_v2_ft_ner_onto_def2 \
#       --test_ner_fn ner_data/230601_0558-bert-base-NER-ckpt4250/mw24_100p_test_ner.json \


python run_GPT35_test_slot_only_predict.py \
      --train_fn data/mw21_5p_train_v2.json \
      --retriever_dir retriever/expts/5p_v2_ft_sep_entity_sim_0.1_7 \
      --output_file_name gpt35_turbo_5p_v2_slot_only_predict \
      --mwz_ver 2.4 \
      --test_size 100 \