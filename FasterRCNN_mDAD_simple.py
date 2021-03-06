# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from genericpath import exists
from operator import truediv
from numpy.core.arrayprint import dtype_is_implied

from torch import DoubleStorage, tensor

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import copy

from lib.roi_dg_data_layer.roidb_DG import combined_roidb
from lib.roi_dg_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.utils.net_utils import weights_normal_init, save_net, load_net, \
        adjust_learning_rate, save_checkpoint, clip_gradient

from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.nms.nms_wrapper import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.utils.net_utils import save_net, load_net, vis_detections

from lib.model.faster_rcnn_DAD_simple.vgg16 import vgg16
from lib.model.faster_rcnn_DAD_simple.resnet import resnet
import torchvision.utils as vutils

from test import test_model
import setproctitle

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                      help='vgg16, res101',
                      default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)
    parser.add_argument('--mode', dest="mode", help='set program mode(train_model/test_model/train_patch/test_patch)', 
                        default='test_model', type=str)
    parser.add_argument('--model_dir', dest='model_dir',
                        help='directory to load models', default="models.pth",
                        type=str)
    parser.add_argument('--part', dest='part',
                        help='test_s or test_t or test_all', default="test_t",
                        type=str)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="./SaveFile/model/DAD_simple_G2",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')                      
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    parser.add_argument('--loadname', dest='loadname',
                        help='input loadname',
                        default='s_cityscape.pth')
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--log_flag', dest='log_flag', # add by xmj
                        help='1:batch_loss, 2:epoch_test',
                        default=0, type=int)

    args = parser.parse_args()
    return args

class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

if __name__ == '__main__':    #?????????????????????    

    setproctitle.setproctitle("< xmj_DG_31 >")

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == "pascal_voc":
        print('loading our dataset...........')
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "cityscape":
        print('loading our dataset...........')
        args.s_imdb_name = "cityscape_2007_train_s"
        args.t_imdb_name = "cityscape_2007_train_t"
        args.D1_imdb_name = "domain1_2007_train_s"
        args.D2_imdb_name = "domain2_2007_train_s"
        args.Dqs1_imdb_name = "domain_qs1_2007_train_s"
        args.DRB_imdb_name = "domain_RB_2007_train_s"
        args.DGY_imdb_name = "domain_GY_2007_train_s"
        args.DRB2_imdb_name = "domain_RB2_2007_train_s"
        args.DGY2_imdb_name = "domain_GY2_2007_train_s"
        args.s_imdbtest_name="cityscape_2007_test_s"
        args.t_imdbtest_name="cityscape_2007_test_t"
        args.all_imdb_name="cityscape_2007_train_all"
        args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    #torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # ??????????????????
    # ??????????????????
    M_D = time.strftime("(%b-%d[%H])", time.localtime())

    def train_model():
        print(">>train model start")
        # -- Note: Use validation set and disable the flipped to enable faster loading.
        cfg.TRAIN.USE_FLIPPED = True
        cfg.USE_GPU_NMS = args.cuda
        '''
        s_imdb        -> ????????????????????????       !! ?????? imdb = cityscape(train_s, 2007)
        s_roidb       -> ?????????????????????????????????  !! ?????? [{ ???????????????????????? },{ ???????????????????????? },{...}]
        s_ratio_list  -> ???????????????????????????
        s_ratio_index -> ??????????????????
        '''
        # ????????????????????????
        imdb_s, roidb_s, ratio_list_s, ratio_index_s = combined_roidb(args.s_imdb_name)
        # imdb_d1, roidb_d1, ratio_list_d1, ratio_index_d1 = combined_roidb(args.t_imdb_name)
        imdb_d1, roidb_d1, ratio_list_d1, ratio_index_d1 = combined_roidb(args.Dqs1_imdb_name)
        # imdb_d2, roidb_d2, ratio_list_d2, ratio_index_d2 = combined_roidb(args.DGY2_imdb_name)
        
        train_size_s = len(roidb_s)   # add flipped         image_index*2
        train_size_d1 = len(roidb_d1)   # add flipped         image_index*2
        # train_size_d2 = len(roidb_d2)   # add flipped         image_index*2

        print('{:d} source roidb entries'.format(len(roidb_s)))
        print('{:d} domain1 roidb entries'.format(len(roidb_d1)))
        # print('{:d} domain2 roidb entries'.format(len(roidb_d2)))

        output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # ?????? log ??????
        if args.log_flag:
            log_dir = output_dir + '/log'
            loss_log_dir = log_dir + '/loss_log.txt'
            epoch_test_log_dir = log_dir + '/epoch_test_log.txt'
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            with open(loss_log_dir,"w") as f: 
                f.write("[Date: %s]\r" %M_D)   #???????????????????????????????????????????????????f.close( )
            with open(epoch_test_log_dir,"w") as f: 
                f.write("[Date: %s]\r" %M_D)   #???????????????????????????????????????????????????f.close( )

        sampler_batch_s = sampler(train_size_s, args.batch_size)
        dataset_s = roibatchLoader(roidb_s, ratio_list_s, ratio_index_s, args.batch_size, \
                                imdb_s.num_classes, training=True)
        dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                    sampler=sampler_batch_s, num_workers=args.num_workers)
        
        sampler_batch_d1 = sampler(train_size_d1, args.batch_size)
        dataset_d1 = roibatchLoader(roidb_d1, ratio_list_d1, ratio_index_d1, args.batch_size, \
                                imdb_d1.num_classes, training=True)
        dataloader_d1 = torch.utils.data.DataLoader(dataset_d1, batch_size=args.batch_size,
                                    sampler=sampler_batch_d1, num_workers=args.num_workers)
        '''
        sampler_batch_d2 = sampler(train_size_d2, args.batch_size)
        dataset_d2 = roibatchLoader(roidb_d2, ratio_list_d2, ratio_index_d2, args.batch_size, \
                                imdb_d2.num_classes, training=True)
        dataloader_d2 = torch.utils.data.DataLoader(dataset_d2, batch_size=args.batch_size,
                                    sampler=sampler_batch_d2, num_workers=args.num_workers)
        '''
        # initilize the tensor holder here.
        im_data_s = torch.FloatTensor(1)
        im_info_s = torch.FloatTensor(1)
        num_boxes_s = torch.LongTensor(1)
        gt_boxes_s = torch.FloatTensor(1)

        im_data_d1 = torch.FloatTensor(1)
        im_info_d1 = torch.FloatTensor(1)
        num_boxes_d1 = torch.LongTensor(1)
        gt_boxes_d1 = torch.FloatTensor(1)
        
        # im_data_d2 = torch.FloatTensor(1)
        # im_info_d2 = torch.FloatTensor(1)
        # num_boxes_d2 = torch.LongTensor(1)
        # gt_boxes_d2 = torch.FloatTensor(1)

        # ship to cuda
        if args.cuda:
            im_data_s = im_data_s.cuda()
            im_info_s = im_info_s.cuda()
            num_boxes_s = num_boxes_s.cuda()
            gt_boxes_s = gt_boxes_s.cuda()

            im_data_d1 = im_data_d1.cuda()
            im_info_d1 = im_info_d1.cuda()
            num_boxes_d1 = num_boxes_d1.cuda()
            gt_boxes_d1 = gt_boxes_d1.cuda()

            # im_data_d2 = im_data_d2.cuda()
            # im_info_d2 = im_info_d2.cuda()
            # num_boxes_d2 = num_boxes_d2.cuda()
            # gt_boxes_d2 = gt_boxes_d2.cuda()


        # make variable
        im_data_s = Variable(im_data_s)
        im_info_s = Variable(im_info_s)
        num_boxes_s = Variable(num_boxes_s)
        gt_boxes_s = Variable(gt_boxes_s)

        im_data_d1 = Variable(im_data_d1)
        im_info_d1 = Variable(im_info_d1)
        num_boxes_d1 = Variable(num_boxes_d1)
        gt_boxes_d1 = Variable(gt_boxes_d1)

        # im_data_d2 = Variable(im_data_d2)
        # im_info_d2 = Variable(im_info_d2)
        # num_boxes_d2 = Variable(num_boxes_d2)
        # gt_boxes_d2 = Variable(gt_boxes_d2)

        if args.cuda:
            cfg.CUDA = True

        # initilize the network here.
        # ?????????????????????????????????????????????????????????????????????????????????????????????
        if args.net == 'vgg16':
            fasterRCNN_DAD = vgg16(imdb_s.classes, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            fasterRCNN_DAD = resnet(imdb_s.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            fasterRCNN_DAD = resnet(imdb_s.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            fasterRCNN_DAD = resnet(imdb_s.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()

        fasterRCNN_DAD.create_architecture()

        lr = cfg.TRAIN.LEARNING_RATE
        lr = args.lr
        #tr_momentum = cfg.TRAIN.MOMENTUM
        #tr_momentum = args.momentum

        params = []
        for key, value in dict(fasterRCNN_DAD.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

        if args.optimizer == "adam":
            lr = lr * 0.1
            optimizer = torch.optim.Adam(params)
            # optimizer_dg = torch.optim.Adam(im_data_s)

        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

        if args.resume:
            # load_name = os.path.join(output_dir,
            # 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
            load_name = os.path.join(output_dir, args.loadname)
            print("loading checkpoint %s" % (load_name))
            checkpoint = torch.load(load_name)
            args.session = checkpoint['session']
            args.start_epoch = checkpoint['epoch']
            fasterRCNN_DAD.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
            if 'pooling_mode' in checkpoint.keys():
                cfg.POOLING_MODE = checkpoint['pooling_mode']
                print("loaded checkpoint %s" % (load_name))

        if args.mGPUs:
            fasterRCNN_DAD = nn.DataParallel(fasterRCNN_DAD)

        if args.cuda:
            fasterRCNN_DAD.cuda()

        # ??????epoch???????????????
        iters_per_epoch = int(train_size_s / args.batch_size)

        for epoch in range(args.start_epoch, args.max_epochs + 1):
            # setting to train mode

            if args.log_flag:
                with open(loss_log_dir,"a") as f: 
                    f.write("epoch: %d\r" % (epoch))   #???????????????????????????????????????????????????f.close( )

            fasterRCNN_DAD.train()
            loss_temp = 0
            start = time.time()

            if epoch % (args.lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)
                lr *= args.lr_decay_gamma

            # ??????3??????????????????
            data_iter_s = iter(dataloader_s)
            data_iter_d1 = iter(dataloader_d1)
            # data_iter_d2 = iter(dataloader_d2)

            # ??????batch??????
            for step in range(iters_per_epoch):
                data_s = next(data_iter_s)
                data_d1 = next(data_iter_d1)
                # data_d2 = next(data_iter_d2)

                # ?????? D_0 ????????? #########################
                im_data_s.data.resize_(data_s[0].size()).copy_(data_s[0])   #change holder size
                im_info_s.data.resize_(data_s[1].size()).copy_(data_s[1])
                gt_boxes_s.data.resize_(data_s[2].size()).copy_(data_s[2])
                num_boxes_s.data.resize_(data_s[3].size()).copy_(data_s[3])

                # ?????? D_1 ????????? ######################################
                im_data_d1.data.resize_(data_d1[0].size()).copy_(data_d1[0])   #change holder size
                im_info_d1.data.resize_(data_d1[1].size()).copy_(data_d1[1])
                gt_boxes_d1.data.resize_(data_d1[2].size()).copy_(data_d1[2])
                num_boxes_d1.data.resize_(data_d1[3].size()).copy_(data_d1[3])

                # ?????? D_2 ????????? #########################
                # im_data_d2.data.resize_(data_d2[0].size()).copy_(data_d2[0])   #change holder size
                # im_info_d2.data.resize_(data_d2[1].size()).copy_(data_d2[1])
                # gt_boxes_d2.data.resize_(data_d2[2].size()).copy_(data_d2[2])
                # num_boxes_d2.data.resize_(data_d2[3].size()).copy_(data_d2[3])

                # ??????????????????
                im_data = torch.cat((im_data_s, im_data_d1),0)
                im_info = torch.cat((im_info_s, im_info_d1),0)
                gt_boxes = torch.cat((gt_boxes_s, gt_boxes_d1),0)
                num_boxes = torch.cat((num_boxes_s, num_boxes_d1),0)
                # im_data = torch.cat((im_data_s, im_data_d1, im_data_d2),0)
                # im_info = torch.cat((im_info_s, im_info_d1, im_info_d2),0)
                # gt_boxes = torch.cat((gt_boxes_s, gt_boxes_d1, gt_boxes_d2),0)
                # num_boxes = torch.cat((num_boxes_s, num_boxes_d1, num_boxes_d2),0)

                fasterRCNN_DAD.zero_grad()

                # ??????size [3, 3, 600, 1200]
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, \
                d0_d01_img_loss_cls, d0_d01_ins_loss_cls, d0_d01_cst_loss, \
                d1_d01_img_loss_cls, d1_d01_ins_loss_cls, d1_d01_cst_loss, \
                        = fasterRCNN_DAD(im_data, im_info, gt_boxes, num_boxes)
                # d0_d02_img_loss_cls, d0_d02_ins_loss_cls, d0_d02_cst_loss, \
                # d2_d02_img_loss_cls, d2_d02_ins_loss_cls, d2_d02_cst_loss, \
                # d1_d12_img_loss_cls, d1_d12_ins_loss_cls, d1_d12_cst_loss, \
                # d2_d12_img_loss_cls, d2_d12_ins_loss_cls, d2_d12_cst_loss \
                #         = fasterRCNN_DAD(im_data, im_info, gt_boxes, num_boxes)

                da_loss =0.1*(d0_d01_img_loss_cls.mean() + d0_d01_ins_loss_cls.mean() + d0_d01_cst_loss.mean() + \
                            d1_d01_img_loss_cls.mean() + d1_d01_ins_loss_cls.mean() + d1_d01_cst_loss.mean())# + \
                            # d0_d02_img_loss_cls.mean() + d0_d02_ins_loss_cls.mean() + d0_d02_cst_loss.mean() + \
                            # d2_d02_img_loss_cls.mean() + d2_d02_ins_loss_cls.mean() + d2_d02_cst_loss.mean() + \
                            # d1_d12_img_loss_cls.mean() + d1_d12_ins_loss_cls.mean() + d1_d12_cst_loss.mean() + \
                            # d2_d12_img_loss_cls.mean() + d2_d12_ins_loss_cls.mean() + d2_d12_cst_loss.mean()) 

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                    + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
                    + da_loss.mean()
                loss_temp += loss.item()

                # backward
                optimizer.zero_grad()
                loss.backward()
                if args.net == "vgg16":
                    clip_gradient(fasterRCNN_DAD, 10.)
                optimizer.step()

                if step % args.disp_interval == 0:
                    end = time.time()
                    if step > 0:
                        loss_temp /= (args.disp_interval + 1)

                    if args.mGPUs:
                        loss_rpn_cls = rpn_loss_cls.mean().item()
                        loss_rpn_box = rpn_loss_box.mean().item()
                        loss_rcnn_cls = RCNN_loss_cls.mean().item()
                        loss_rcnn_box = RCNN_loss_bbox.mean().item()
                        loss_da = da_loss.mean().item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt
                    else:
                        loss_rpn_cls = rpn_loss_cls.item()
                        loss_rpn_box = rpn_loss_box.item()
                        loss_rcnn_cls = RCNN_loss_cls.item()
                        loss_rcnn_box = RCNN_loss_bbox.item()
                        loss_da = da_loss.mean().item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt

                    print(args.save_dir)
                    print("[session %d][epoch %2d][iter %4d/%4d] \t\033[0;31m loss: %.4f\033[0m, lr: %.2e" \
                                            % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                    print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                    print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, loss_da %.4f" \
                                % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_da))

                    # loss write
                    if args.log_flag:
                        with open(loss_log_dir,"a") as f: 
                            f.write("  count: %s  \tloss: %f\r" % (step, loss_temp))   #???????????????????????????????????????????????????f.close( )

                    loss_temp = 0
                    start = time.time()

            if epoch <= args.max_epochs: # == args.max_epochs:
                save_name = os.path.join(output_dir, 's_cityscape_e%s.pth' % epoch)
                save_checkpoint({
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': fasterRCNN_DAD.module.state_dict() if args.mGPUs else fasterRCNN_DAD.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
                print('save model: {}'.format(save_name))
                map = test_model(save_name, args)
                # MAP write
                if args.log_flag:
                    with open(epoch_test_log_dir,"a") as f: 
                        f.write("  epoch: %s  \tmap: %f\r" % (epoch, map))   #???????????????????????????????????????????????????f.close( )
    
    # train set
    if args.mode == "train_model":
        train_model()
    elif args.mode == "test_model":
        map = test_model(args.model_dir, args)
        print("\n\tMAP: %s" % map)