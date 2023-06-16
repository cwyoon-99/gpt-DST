# # first step
# python ett_replace.py \
#         --output_file_name base \
#         --span_info_dir data/mwz21/origin_data.json \

# third step
# python ett_replace.py \
#         --output_file_name ner_preprocessed \
#         --specific_name ett \
#         --span_info_dir data/mwz21/origin_data.json \
#         --train_ner_fn ner_data/230530_0638-bert-large-NER-ckpt3000/mw21_5p_train_v2_ner.json \

# python ett_replace.py \
#         --output_file_name ner_best_f1 \
#         --specific_name ett \
#         --span_info_dir data/mwz21/origin_data.json \
#         --train_ner_fn ner_data/230604_1442-bert-large-NER-ckpt2750/mw21_5p_train_v2_ner.json \
#         --dev_ner_fn ner_data/230604_1442-bert-large-NER-ckpt2750/mw24_100p_dev_ner.json \
#         --test_ner_fn ner_data/230604_1442-bert-large-NER-ckpt2750/mw24_100p_test_ner.json

python ett_replace.py \
        --output_file_name ner_wo_booking_best_f1 \
        --specific_name ett \
        --span_info_dir data/mwz21/origin_data.json \
        --train_ner_fn ner_data/230614_1437-bert-large-NER-ckpt1650-wo-booking/mw21_5p_train_v2_ner.json \
        --dev_ner_fn ner_data/230614_1437-bert-large-NER-ckpt1650-wo-booking/mw24_100p_dev_ner.json \
        --test_ner_fn ner_data/230614_1437-bert-large-NER-ckpt1650-wo-booking/mw24_100p_test_ner.json