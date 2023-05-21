python retriever_finetuning.py \
    --train_fn ../../data/mw21_25p_train_v2.json \
    --save_name 01p_v2 \
    --epoch 15 \
    --topk 50 \
    --toprange 1000 \

# 1% : topk=2, toprange=40
# 5% : topk=10, toprange=200
# 10% : topk=20, toprange=400
# 25% : topk=50, toprange=1000