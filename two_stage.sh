#!/bin/bash

cifar_10_lt_ir100_bsl_teacher_model_path='path/to/one_stage_weight'

python3 teach_me_noise.py \
    --teacher_model_path ${cifar_10_lt_ir100_bsl_teacher_model_path} \
    --gpu_id 0 \
    > $(date "+%Y%m%d-%H%M%S").log

wait

