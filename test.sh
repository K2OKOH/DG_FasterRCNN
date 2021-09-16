#!/bin/bash
# 统一测试脚本

CUDA_VISIBLE_DEVICES=3 python test.py \
        --dataset cityscape \
        --net vgg16 \
        --part test_t \
        --model_dir  "./SaveFile/model/DAD_simple_ λ/vgg16/cityscape/s_cityscape_e5.pth" \
        --cuda \
