#!/bin/bash
# 统一训练脚本

echo "training program start!"

CUDA_VISIBLE_DEVICES=2 python FasterRCNN_DAD_simple.py \
        --dataset cityscape \
        --net vgg16 \
        --lr 2e-3 \
        --lr_decay_step 6 \
        --cuda \
        --epochs 10 \
        --bs 1 \
        --save_dir ./SaveFile/model/DG_gd_00 \
        --mode train_model \
        --log_flag 1 \
        # --mGPUs \
#       --r True \
#       --loadname s_cityscape_e6.pth \
#       --checkepoch  \

