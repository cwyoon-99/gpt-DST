# python retriever_finetuning_custom.py \
#     --train_fn ../../data/mw21_5p_train_v2_ner_onto_def.json \
#     --save_name 5p_v2_ft_ner_onto_def \
#     --pretrained_index_dir 5p_v2_ner_onto_def \
#     --epoch 15 \
#     --topk 10 \
#     --toprange 200 \

python retriever_finetuning_custom.py \
    --train_fn ../../data/mw21_5p_train_v2_span_info_ett_meta_2.json \
    --save_name 5p_v2_ft_sep_entity_sim_0.1_12 \
    --epoch 15 \
    --topk 10 \
    --toprange 200 \
    --entity_lambda 0.1 \
    --dev_fn ../../data/mw24_100p_dev_span_info_ett_meta_2.json \
    --test_fn ../../data/mw24_100p_test_span_info_ett_meta_2.json

# 1% : topk=2, toprange=40
# 5% : topk=10, toprange=200
# 10% : topk=20, toprange=400
# 25% : topk=50, toprange=1000