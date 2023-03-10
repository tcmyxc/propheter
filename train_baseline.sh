#!/bin/bash


python3 train_baseline.py \
    --model_name resnet32 \
    --data_name cifar-10-lt-ir100 \
    --lr 0.01 \
    --epochs 200 \
    --loss_type bsl \
    --gpu_id 0 \
    > logs/base/$(date "+%Y%m%d-%H%M%S").log
wait
