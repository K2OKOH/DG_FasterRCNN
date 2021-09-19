#!/bin/bash
# 统一训练脚本

echo "training program start!"

CUDA_VISIBLE_DEVICES=0,2,3 python FasterRCNN_DAD_simple.py \
        --dataset cityscape \
        --net vgg16 \
        --lr 2e-3 \
        --lr_decay_step 6 \
        --cuda \
        --epochs 10 \
        --mode train_model \
        --bs 3 \
        --log_flag 1 \
        --save_dir ./SaveFile/model/DAD_simple_G2_3l_mg \
        --mGPUs \
#       --r True \
#       --loadname s_cityscape_e6.pth \
#       --checkepoch  \

