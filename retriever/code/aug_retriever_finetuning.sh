python aug_retriever_finetuning.py \
    --train_fn ../../para/230523_0150-aug_0.5p_v1/mw21_aug_0.5p_train_v1.json \
    --save_name aug_0.5p_v1 \
    --epoch 15 \
    --topk 2 \
    --toprange 20 \

# 1% : topk=2, toprange=40
# 5% : topk=10, toprange=200
# 10% : topk=20, toprange=400
# 25% : topk=50, toprange=1000