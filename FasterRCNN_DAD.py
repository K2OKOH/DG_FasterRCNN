# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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

from lib.roi_data_layer.roidb_DG import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.nms.nms_wrapper import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.utils.net_utils import save_net, load_net, vis_detections

from lib.model.faster_rcnn_DAD.vgg16 import vgg16
from lib.model.faster_rcnn_DAD.resnet import resnet
import torchvision.utils as vutils

from lib.patch.utils import *
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
                        help='directory to save models', default="./SaveFile/model/bg_InAllLoss",
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

    # patch info
    parser.add_argument('--PatchSize', dest='PatchSize',
                        help='the side length of patches',
                        default=100, type=int) 
    parser.add_argument('--PatchTarget', dest='PatchTarget',
                        help='the target of patches',
                        default=0, type=int)  
    args = parser.parse_args()
    return args

def draw_box(draw_box,image,color,width=4):
    image = image.data.cpu().numpy()

    if len(draw_box.size()) == 3:
        draw_box = draw_box[0]
    elif len(draw_box.size()) == 2:
        draw_box = draw_box

    if color == "RED":
        box_RGB = [1,0,0]
        print("\033[5;31mGT:\033[0m")
    elif color == "GREEN":
        box_RGB = [0,1,0]
        print("\033[5;32mATTACK:\033[0m")
    elif color == "BLUE":
        box_RGB = [0,0,1]
        print("\033[5;34mPATCH:\033[0m")
    elif color == "YELLOW":
        box_RGB = [1,1,0]
        print("\033[5;33mPRE:\033[0m")
        
    for i in range(draw_box.size(0)):
        box = draw_box[i]
        box = box.cpu().numpy()

        w_s = int(box[0])
        h_s = int(box[1])
        w_e = int(box[2])
        h_e = int(box[3])
        
        # R
        image[0, 0, h_s:h_e, w_s:w_s+width] = box_RGB[0]
        image[0, 0, h_s:h_e, w_e-width:w_e] = box_RGB[0]
        image[0, 0, h_s:h_s+width, w_s:w_e] = box_RGB[0]
        image[0, 0, h_e-width:h_e, w_s:w_e] = box_RGB[0]
        # G
        image[0, 1, h_s:h_e, w_s:w_s+width] = box_RGB[1]
        image[0, 1, h_s:h_e, w_e-width:w_e] = box_RGB[1]
        image[0, 1, h_s:h_s+width, w_s:w_e] = box_RGB[1]
        image[0, 1, h_e-width:h_e, w_s:w_e] = box_RGB[1]
        # B
        image[0, 2, h_s:h_e, w_s:w_s+width] = box_RGB[2]
        image[0, 2, h_s:h_e, w_e-width:w_e] = box_RGB[2]
        image[0, 2, h_s:h_s+width, w_s:w_e] = box_RGB[2]
        image[0, 2, h_e-width:h_e, w_s:w_e] = box_RGB[2]

        print("No.%d\ts: (%s,%s)\te: (%s,%s)" % (i,w_s,h_s,w_e,h_e))

    return torch.from_numpy(image)

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

if __name__ == '__main__':    #仅作为脚本运行    

    setproctitle.setproctitle("xmj XD")

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
        args.D3_imdb_name = "domain3_2007_train_s"
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

    # 加载网络模型
    # 加载当前日期
    M_D = time.strftime("(%b-%d[%H])", time.localtime())

    # train set
    if args.mode == "train_model":
        print(">>train model start")
        # -- Note: Use validation set and disable the flipped to enable faster loading.
        cfg.TRAIN.USE_FLIPPED = True
        cfg.USE_GPU_NMS = args.cuda

        # 读取三个域的数据
        imdb_s, roidb_s, ratio_list_s, ratio_index_s = combined_roidb(args.s_imdb_name)
        train_size_s = len(roidb_s)   # add flipped         image_index*2

        print('{:d} source roidb entries'.format(len(roidb_s)))

        output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sampler_batch_s = sampler(train_size_s, args.batch_size)
        dataset_s = roibatchLoader(roidb_s, ratio_list_s, ratio_index_s, args.batch_size, \
                                imdb_s.num_classes, training=True)
        dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                    sampler=sampler_batch_s, num_workers=args.num_workers)
        
        # initilize the tensor holder here.
        im_data_s = torch.FloatTensor(1)
        im_info_s = torch.FloatTensor(1)
        num_boxes_s = torch.LongTensor(1)
        gt_boxes_s = torch.FloatTensor(1)

        # ship to cuda
        if args.cuda:
            im_data_s = im_data_s.cuda()
            im_info_s = im_info_s.cuda()
            num_boxes_s = num_boxes_s.cuda()
            gt_boxes_s = gt_boxes_s.cuda()

        # make variable
        im_data_s = Variable(im_data_s)
        im_info_s = Variable(im_info_s)
        num_boxes_s = Variable(num_boxes_s)
        gt_boxes_s = Variable(gt_boxes_s)

        if args.cuda:
            cfg.CUDA = True

        # initilize the network here.
        # 暂时所有的领域的类别都是一样的，所以就拿源域的类别来建立网络了
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

        # 计算epoch的迭代次数
        iters_per_epoch = int(train_size_s / args.batch_size)

        for epoch in range(args.start_epoch, args.max_epochs + 1):
            # setting to train mode

            fasterRCNN_DAD.train()
            loss_temp = 0
            start = time.time()

            if epoch % (args.lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)
                lr *= args.lr_decay_gamma

            # 准备3个域的迭代器
            data_iter_s = iter(dataloader_s)

            # 进行batch迭代
            for step in range(iters_per_epoch):
                data_s = next(data_iter_s)

                # 针对source的训练 #########################
                im_data_s.data.resize_(data_s[0].size()).copy_(data_s[0])   #change holder size
                im_info_s.data.resize_(data_s[1].size()).copy_(data_s[1])
                gt_boxes_s.data.resize_(data_s[2].size()).copy_(data_s[2])
                num_boxes_s.data.resize_(data_s[3].size()).copy_(data_s[3])

                fasterRCNN_DAD.zero_grad()
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, \
                fg_sim, bg_sim = fasterRCNN_DAD(im_data_s, im_info_s, gt_boxes_s, num_boxes_s)

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                    + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
                    - fg_sim.mean() + bg_sim.mean()
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
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt
                    else:
                        loss_rpn_cls = rpn_loss_cls.item()
                        loss_rpn_box = rpn_loss_box.item()
                        loss_rcnn_cls = RCNN_loss_cls.item()
                        loss_rcnn_box = RCNN_loss_bbox.item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt

                    print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                            % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                    print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                    print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                                % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

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

    elif args.mode == "test_model":
        print(">>test model start")

        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name, False)

        imdb.competition_mode(on=True)

        print('{:d} roidb entries'.format(len(roidb)))

        load_name=args.model_dir

        # initilize the network here.
        if args.net == 'vgg16':
            fasterRCNN_DAD = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            fasterRCNN_DAD = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            fasterRCNN_DAD = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            fasterRCNN_DAD = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()

        fasterRCNN_DAD.create_architecture()

        print("load checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        fasterRCNN_DAD.load_state_dict({k: v for k, v in checkpoint['model'].items() if k in fasterRCNN_DAD.state_dict()})
        #fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']


        print('load model successfully!')
        # initilize the tensor holder here.
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if args.cuda:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_boxes = num_boxes.cuda()
            gt_boxes = gt_boxes.cuda()

        # make variable
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

        if args.cuda:
            cfg.CUDA = True

        if args.cuda:
            fasterRCNN_DAD.cuda()

        start = time.time()
        max_per_image = 100

        vis = args.vis

        if vis:
            thresh = 0.05
        else:
            thresh = 0.0

        save_name = args.part+'_faster_rcnn_10'
        num_images = len(imdb.image_index)
        all_boxes = [[[] for _ in range(num_images)]
                    for _ in range(imdb.num_classes)]

        output_dir = get_output_dir(imdb, save_name)
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                                imdb.num_classes, training=False, normalize = False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                    shuffle=False, num_workers=0,
                                    pin_memory=True)

        data_iter = iter(dataloader)

        _t = {'im_detect': time.time(), 'misc': time.time()}
        det_file = os.path.join(output_dir, 'detections.pkl')

        fasterRCNN_DAD.eval()
        empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
        for i in range(num_images):

            data = next(data_iter)
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            det_tic = time.time()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, \
            fg_sim, bg_sim = fasterRCNN_DAD(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= data[1][0][2].item()

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()
            if vis:
                im = cv2.imread(imdb.image_path_at(i))
                im2show = np.copy(im)
            for j in range(1, imdb.num_classes):
                inds = torch.nonzero(scores[:,j]>thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if vis:
                        im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                            for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                .format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

            if vis:
                cv2.imwrite('result.png', im2show)
                pdb.set_trace()
                #cv2.imshow('test', im2show)
                #cv2.waitKey(0)

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        imdb.evaluate_detections(all_boxes, output_dir)

        end = time.time()
        print("test time: %0.4fs" % (end - start))
