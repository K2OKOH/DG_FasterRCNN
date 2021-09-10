# CUDA_VISIBLE_DEVICES=1 python FasterRCNN.py \
#        --dataset cityscape \
#        --net vgg16 \
#        --lr 2e-3 \
#        --lr_decay_step 6 \
#        --cuda \
#        --epochs 10 \
#        --mode train_model \
#        --bs 1 \
#        --save_dir ./SaveFile/model/init

CUDA_VISIBLE_DEVICES=0,1,2,3 python FasterRCNN_6DataSet.py \
        --dataset cityscape \
        --net vgg16 \
        --lr 2e-3 \
        --lr_decay_step 6 \
        --cuda \
        --epochs 10 \
        --mode train_model \
        --bs 4 \
        --mGPUs \
#        --r True \
#        --loadname s_cityscape_e6.pth \
#        --checkepoch 6 \
