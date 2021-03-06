#!/bin/bash
# 统一测试脚本

CUDA_VISIBLE_DEVICES=1 python test.py \
        --dataset cityscape \
        --net vgg16 \
        --part test_t \
        --model_dir  "SaveFile/model/source+target/vgg16/cityscape/cityscape_e10.pth" \
        --cuda \
