python finetuned_embed_with_augmented.py \
    --train_fn ../../data/mw21_5p_train_v2.json \
    --aug_name element2 \
    --load_retriever_dir mw21_5p_v2 \
    --emb_save_dir ../expts/fullshot_with_element2 \