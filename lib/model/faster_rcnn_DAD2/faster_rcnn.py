import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.conv import Conv2d
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from lib.model.utils.config import cfg
from lib.model.rpn.rpn_DAD import _RPN
from lib.model.roi_pooling.modules.roi_pool import _RoIPooling
from lib.model.roi_crop.modules.roi_crop import _RoICrop
from lib.model.roi_align.modules.roi_align import RoIAlignAvg
from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from lib.model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
        
        self.DG_layer_1 = nn.Sequential(Conv2d(3,3,3,1,1))
        self.DG_layer_2 = nn.Sequential(Conv2d(3,3,3,1,1))
        # debug use
        
        self.DB_flag = False

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data     #(size1,size2, image ratio(new image / source image) )
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # 先把 image 送入域变化网络中
        im_data_d1 = self.DG_layer_1(im_data)
        im_data_d2 = self.DG_layer_2(im_data)

        # 希望变化之后的domain之间的差距变大
        # 应为维度问题暂时搁置
        # F.cosine_similarity(im_data_d1, im_data_d2, 1)

        # feed image data to base model to obtain base feature map
        # 把 经过域变化后的 image 送入 同一个 特征提取网络
        base_feat_0 = self.RCNN_base(im_data)
        base_feat_1 = self.RCNN_base(im_data_d1)
        base_feat_2 = self.RCNN_base(im_data_d2)

        # feed base feature map tp RPN to obtain rois
        # 用原始图片提取出的特征计算 roi
        rois, bg_rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_0, im_info, gt_boxes, num_boxes)

        if self.DB_flag:
            print(">> RPN finish")
            print("\tfg_roi sizes: ", rois.size())
            print("\tbg_roi sizes: ", rois.size())

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        bg_rois = Variable(bg_rois)

        if self.DB_flag:
            print("roi: ",rois.size())
            print("bg_roi: ",bg_rois.size())
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat_0.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat_0, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            # 默认采用align方法
            # 三个域 全部使用 一个 roi
            # raw domain
            pooled_feat = self.RCNN_roi_align(base_feat_0, rois.view(-1, 5))
            bg_pooled_feat = self.RCNN_roi_align(base_feat_0, bg_rois.view(-1, 5))
            # F1 domain
            pooled_feat_1 = self.RCNN_roi_align(base_feat_1, rois.view(-1, 5))
            bg_pooled_feat_1 = self.RCNN_roi_align(base_feat_1, bg_rois.view(-1, 5))
            # F2 domain
            pooled_feat_2 = self.RCNN_roi_align(base_feat_2, rois.view(-1, 5))
            bg_pooled_feat_2 = self.RCNN_roi_align(base_feat_2, bg_rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat_0, rois.view(-1,5))
        
        if self.DB_flag:
            print("feature_map: ",base_feat_0.size())
            print("s_pooled_feature: ",pooled_feat.size())
            print("s_bg_pooled_feature: ",bg_pooled_feat.size())
        
        # feed pooled features to top model
        # 输入头部网络
        pooled_feat = self._head_to_tail(pooled_feat)
        bg_pooled_feat = self._head_to_tail(bg_pooled_feat)
        # F1
        pooled_feat_1 = self._head_to_tail(pooled_feat_1)
        bg_pooled_feat_1 = self._head_to_tail(bg_pooled_feat_1)
        # F2
        pooled_feat_2 = self._head_to_tail(pooled_feat_2)
        bg_pooled_feat_2 = self._head_to_tail(bg_pooled_feat_2)

        fg_sim_s2 = F.cosine_similarity(pooled_feat, pooled_feat_2, 1, 1e-5).mean()
        bg_sim_s2 = F.cosine_similarity(bg_pooled_feat, bg_pooled_feat_2, 1, 1e-5).mean()
        fg_sim_12 = F.cosine_similarity(pooled_feat_1, pooled_feat_2, 1, 1e-5).mean()
        bg_sim_12 = F.cosine_similarity(bg_pooled_feat_1, bg_pooled_feat_2, 1, 1e-5).mean()
        fg_sim_s1 = F.cosine_similarity(pooled_feat, pooled_feat_1, 1, 1e-5).mean()
        bg_sim_s1 = F.cosine_similarity(bg_pooled_feat, bg_pooled_feat_1, 1, 1e-5).mean()

        # 计算前景之间，背景之间的相似性（cos），想要前景相似背景不相似。
        # 余弦相似度 越大 越相似
        fg_sim = (fg_sim_s1, fg_sim_12, fg_sim_s2).mean()
        bg_sim = (bg_sim_s1, bg_sim_12, bg_sim_s2).mean()

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        if self.training:
            rpn_loss_cls = torch.unsqueeze(rpn_loss_cls, 0)
            rpn_loss_bbox = torch.unsqueeze(rpn_loss_bbox, 0)
            RCNN_loss_cls = torch.unsqueeze(RCNN_loss_cls, 0)
            RCNN_loss_bbox = torch.unsqueeze(RCNN_loss_bbox, 0)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, fg_sim, bg_sim

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()