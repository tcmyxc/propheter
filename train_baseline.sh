#!/bin/bash


python3 pure_cls_train.py \
    --model_name resnet32 \
    --data_name cifar-10-lt-ir100 \
    --lr 0.01 \
    --epochs 200 \
    --loss_type bsl \
    --gpu_id 0 \
    > baseline_log/$(date "+%Y%m%d-%H%M%S").log
wait
