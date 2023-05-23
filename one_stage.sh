#!/bin/bash


python3 modify_acts.py \
    --data_name cifar-10-lt-ir100 \
    --model_name resnet32 \
    --lr 0.01 \
    --lr_scheduler cosine \
    --loss_type bsl \
    --epochs 200 \
    --gpu_id 1 \
    > $(date "+%Y%m%d-%H%M%S").log
  
wait

