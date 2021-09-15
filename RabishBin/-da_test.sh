# Test model in target domain 
CUDA_VISIBLE_DEVICES=1 python da_test_net.py \
            --dataset cityscape \
            --part test_t \
            --model_dir ./SaveFile/model/DAD_simple/vgg16/cityscape/s_cityscape_e10.pth \
            --cuda
