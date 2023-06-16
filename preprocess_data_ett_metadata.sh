# ett_replaced_fn in this command is predicted result from NER model
python preprocess_data_ett_metadata.py \
        --output_file_name ner_best_f1_wo_booking_ett_meta \
        --span_info_dir data/mwz21/origin_data.json \
        --train_ett_replaced_fn ett_replace/230615_0033-1.0-ner_wo_booking_best_f1/mw21_5p_train_v2_ner_wo_booking_best_f1.json \
        --dev_ett_replaced_fn ett_replace/230615_0033-1.0-ner_wo_booking_best_f1/mw24_100p_dev_ner_wo_booking_best_f1.json\
        --test_ett_replaced_fn ett_replace/230615_0033-1.0-ner_wo_booking_best_f1/mw24_100p_test_ner_wo_booking_best_f1.json

# # ett_replaced_fn in this command is span info
# python preprocess_data_ett_metadata.py \
#         --output_file_name span_info_ett_meta_2 \
#         --span_info_dir data/mwz21/origin_data.json \
#         --train_ett_replaced_fn ett_replace/230530_0109-1.0-base/data/mw21_5p_train_v2.json \
#         --dev_ett_replaced_fn ett_replace/230530_0109-1.0-base/data/mw24_100p_dev.json \
#         --test_ett_replaced_fn ett_replace/230530_0109-1.0-base/data/mw24_100p_test.json