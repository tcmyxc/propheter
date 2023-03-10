#!/bin/bash

cifar_10_lt_ir100_bsl_teacher_model_path='./work_dir/cifar-10-lt-ir100_resnet32/20230310/220957/best-model-acc0.8232.pth'

python3 teach_me_noise.py \
    --teacher_model_path ${cifar_10_lt_ir100_bsl_teacher_model_path} \
    --gpu_id 1 \
    > logs/two_stage/$(date "+%Y%m%d-%H%M%S").log

wait

