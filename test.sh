#!/bin/bash
# 统一测试脚本

CUDA_VISIBLE_DEVICES=3 python test.py \
        --dataset cityscape \
        --net vgg16 \
        --part test_t \
        --model_dir  ./SaveFile/model/da_baseline/vgg16/cityscape/cityscape_da_baseline_e3.pth \
        --cuda \
