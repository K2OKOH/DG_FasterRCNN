#!/bin/bash
# 统一训练脚本

echo "training program start!"

CUDA_VISIBLE_DEVICES=2 python DiffCoder.py \
        --lr 2e-3 \
        --lr_decay_step 6 \
        --cuda \
        --epochs 10 \
        --bs 1 \
        --save_dir SaveFile/model/encoder/ \
        --task_name AE_Diff_1 \
