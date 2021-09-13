import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from lib.model.utils.config import cfg
from lib.model.rpn.rpn import _RPN
from lib.model.roi_pooling.modules.roi_pool import _RoIPooling
from lib.model.roi_crop.modules.roi_crop import _RoICrop
from lib.model.roi_align.modules.roi_align import RoIAlignAvg
from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer

from lib.model.faster_rcnn_DAD_simple.DG import _ImageDG
from lib.model.faster_rcnn_DAD_simple.DG import _InstanceDG
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

        self.RCNN_imageDG_d01 = _ImageDG(self.dout_base_model)
        self.RCNN_instanceDG_d01 = _InstanceDG()
        self.RCNN_imageDG_dl2 = _ImageDG(self.dout_base_model)
        self.RCNN_instanceDG_d12 = _InstanceDG()
        self.RCNN_imageDG_d02 = _ImageDG(self.dout_base_model)
        self.RCNN_instanceDG_d02 = _InstanceDG()
        self.consistency_loss = torch.nn.MSELoss(size_average=False)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):

        # 输入三个域的图片
        im_data_d0 = im_data[0].unsqueeze(0)
        im_data_d1 = im_data[1].unsqueeze(0)
        im_data_d2 = im_data[2].unsqueeze(0)

        im_info_d0 = im_info[0].unsqueeze(0)
        im_info_d1 = im_info[1].unsqueeze(0)
        im_info_d2 = im_info[2].unsqueeze(0)

        gt_boxes_d0 = gt_boxes[0].unsqueeze(0)
        gt_boxes_d1 = gt_boxes[1].unsqueeze(0)
        gt_boxes_d2 = gt_boxes[2].unsqueeze(0)

        num_boxes_d0 = num_boxes[0].unsqueeze(0)
        num_boxes_d1 = num_boxes[1].unsqueeze(0)
        num_boxes_d2 = num_boxes[2].unsqueeze(0)

        batch_size = im_data_d0.size(0)
        
        im_info_d0 = im_info_d0.data     #(size1,size2, image ratio(new image / source image) )
        gt_boxes_d0 = gt_boxes_d0.data
        num_boxes_d0 = num_boxes_d0.data

        im_info_d1 = im_info_d1.data     #(size1,size2, image ratio(new image / source image) )
        gt_boxes_d1 = gt_boxes_d1.data
        num_boxes_d1 = num_boxes_d1.data

        im_info_d2 = im_info_d2.data     #(size1,size2, image ratio(new image / source image) )
        gt_boxes_d2 = gt_boxes_d2.data
        num_boxes_d2 = num_boxes_d2.data

        # feed image data to base model to obtain base feature map
        base_feat_d0 = self.RCNN_base(im_data_d0)
        base_feat_d1 = self.RCNN_base(im_data_d1)
        base_feat_d2 = self.RCNN_base(im_data_d2)

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.train()
        # 提取三张图的FeatureMap
        rois_d0, rpn_loss_cls_d0, rpn_loss_bbox_d0 = self.RCNN_rpn(base_feat_d0, im_info_d0, gt_boxes_d0, num_boxes_d0)
        rois_d1, rpn_loss_cls_d1, rpn_loss_bbox_d1 = self.RCNN_rpn(base_feat_d1, im_info_d1, gt_boxes_d1, num_boxes_d1)
        rois_d2, rpn_loss_cls_d2, rpn_loss_bbox_d2 = self.RCNN_rpn(base_feat_d2, im_info_d2, gt_boxes_d2, num_boxes_d2)

        # if it is training phrase, then use ground trubut bboxes for refining
        # 训练过程要控制roi的数量，因为是DG任务，每个domain都是有标签的
        if self.training:
            roi_data = self.RCNN_proposal_target(rois_d0, gt_boxes_d0, num_boxes_d0)
            rois_d0, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label_d0 = Variable(rois_label.view(-1).long())
            rois_target_d0 = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws_d0 = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws_d0 = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            
            roi_data = self.RCNN_proposal_target(rois_d0, gt_boxes_d0, num_boxes_d0)
            rois_d1, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label_d1 = Variable(rois_label.view(-1).long())
            rois_target_d1 = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws_d1 = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws_d1 = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

            roi_data = self.RCNN_proposal_target(rois_d0, gt_boxes_d0, num_boxes_d0)
            rois_d2, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label_d2 = Variable(rois_label.view(-1).long())
            rois_target_d2 = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws_d2 = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws_d2 = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois_d0 = Variable(rois_d0)
        rois_d1 = Variable(rois_d1)
        rois_d2 = Variable(rois_d2)
        # print("roi: ",rois.size())
        # do roi pooling based on predicted rois

        # roi_pooling for d0, d1, d2
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat_d0, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois_d0.view(-1, 5), base_feat_d0.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat_d0 = self.RCNN_roi_crop(base_feat_d0, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat_d0 = F.max_pool2d(pooled_feat_d0, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat_d0 = self.RCNN_roi_align(base_feat_d0, rois_d0.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_d0 = self.RCNN_roi_pool(base_feat_d0, rois_d0.view(-1,5))
            
        pooled_feat_d1 = self.RCNN_roi_align(base_feat_d1, rois_d1.view(-1, 5))
        pooled_feat_d2 = self.RCNN_roi_align(base_feat_d2, rois_d2.view(-1, 5))
        
        # feed pooled features to top model
        pooled_feat_d0 = self._head_to_tail(pooled_feat_d0)
        pooled_feat_d1 = self._head_to_tail(pooled_feat_d1)
        pooled_feat_d2 = self._head_to_tail(pooled_feat_d2)

        # ========================= reg & cls for d0 =====================================
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat_d0)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label_d0.view(rois_label_d0.size(0), 1, 1).expand(rois_label_d0.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat_d0)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label_d0)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target_d0, rois_inside_ws_d0, rois_outside_ws_d0)

        cls_prob = cls_prob.view(batch_size, rois_d0.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois_d0.size(1), -1)
        
        #=========================开始进行域泛化============================#
        
        #========= d01 classifier: img & ins & cons ==========#

        ## d0
        d0_d01_img_loss_cls = 0
        d0_d01_ins_loss_cls = 0

        # Image-level
        base_score, base_label = self.RCNN_imageDG_d01(base_feat_d0, torch.tensor([1.]).cuda())
        base_prob = F.log_softmax(base_score, dim=1)
        d0_d01_img_loss_cls = F.nll_loss(base_prob, base_label)
        
        # Instance-level
        instance_sigmoid, same_size_label = self.RCNN_instanceDG_d01(pooled_feat_d0, torch.tensor([1.]).cuda())
        instance_loss = nn.BCELoss()
        d0_d01_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

        # consistency loss
        consistency_prob = F.softmax(base_score, dim=1)[:,1,:,:]
        consistency_prob=torch.mean(consistency_prob)
        consistency_prob=consistency_prob.repeat(instance_sigmoid.size())
        d0_d01_cst_loss=self.consistency_loss(instance_sigmoid,consistency_prob.detach())

        #d1
        d1_d01_img_loss_cls = 0
        d1_d01_ins_loss_cls = 0

        # Image-level
        base_score, base_label = self.RCNN_imageDG_d01(base_feat_d1, torch.tensor([0.]).cuda())
        base_prob = F.log_softmax(base_score, dim=1)
        d1_d01_img_loss_cls = F.nll_loss(base_prob, base_label)
        
        # Instance-level
        instance_sigmoid, same_size_label = self.RCNN_instanceDG_d01(pooled_feat_d1, torch.tensor([0.]).cuda())
        instance_loss = nn.BCELoss()
        d1_d01_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

        # consistency loss
        consistency_prob = F.softmax(base_score, dim=1)[:,1,:,:]
        consistency_prob=torch.mean(consistency_prob)
        consistency_prob=consistency_prob.repeat(instance_sigmoid.size())
        d1_d01_cst_loss=self.consistency_loss(instance_sigmoid,consistency_prob.detach())

        #========= d12 classifier: img & ins & cons ==========#

        ## d2
        d2_d12_img_loss_cls = 0
        d2_d12_ins_loss_cls = 0

        # Image-level

        base_score, base_label = self.RCNN_imageDG_dl2(base_feat_d2, torch.tensor([1.]).cuda())
        base_prob = F.log_softmax(base_score, dim=1)
        d2_d12_img_loss_cls = F.nll_loss(base_prob, base_label)
        
        # Instance-level
        instance_sigmoid, same_size_label = self.RCNN_instanceDG_d12(pooled_feat_d2, torch.tensor([1.]).cuda())
        instance_loss = nn.BCELoss()
        d2_d12_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

        # consistency loss
        consistency_prob = F.softmax(base_score, dim=1)[:,1,:,:]
        consistency_prob=torch.mean(consistency_prob)
        consistency_prob=consistency_prob.repeat(instance_sigmoid.size())
        d2_d12_cst_loss=self.consistency_loss(instance_sigmoid,consistency_prob.detach())

        #d1
        d1_d12_img_loss_cls = 0
        d1_d12_ins_loss_cls = 0

        # Image-level
        base_score, base_label = self.RCNN_imageDG_dl2(base_feat_d1, torch.tensor([0.]).cuda())
        base_prob = F.log_softmax(base_score, dim=1)
        d1_d12_img_loss_cls = F.nll_loss(base_prob, base_label)
        
        # Instance-level
        instance_sigmoid, same_size_label = self.RCNN_instanceDG_d12(pooled_feat_d1, torch.tensor([0.]).cuda())
        d1_d12_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

        # consistency loss
        consistency_prob = F.softmax(base_score, dim=1)[:,1,:,:]
        consistency_prob=torch.mean(consistency_prob)
        consistency_prob=consistency_prob.repeat(instance_sigmoid.size())
        d1_d12_cst_loss=self.consistency_loss(instance_sigmoid,consistency_prob.detach())

        #========= d02 classifier: img & ins & cons ==========#

        ## d0
        d0_d02_img_loss_cls = 0
        d0_d02_ins_loss_cls = 0

        # Image-level
        base_score, base_label = self.RCNN_imageDG_d02(base_feat_d0, torch.tensor([1.]).cuda())
        base_prob = F.log_softmax(base_score, dim=1)
        d0_d02_img_loss_cls = F.nll_loss(base_prob, base_label)
        
        # Instance-level
        instance_sigmoid, same_size_label = self.RCNN_instanceDG_d02(pooled_feat_d0, torch.tensor([1.]).cuda())
        instance_loss = nn.BCELoss()
        d0_d02_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

        # consistency loss
        consistency_prob = F.softmax(base_score, dim=1)[:,1,:,:]
        consistency_prob=torch.mean(consistency_prob)
        consistency_prob=consistency_prob.repeat(instance_sigmoid.size())
        d0_d02_cst_loss=self.consistency_loss(instance_sigmoid,consistency_prob.detach())

        #d2
        d2_d02_img_loss_cls = 0
        d2_d02_ins_loss_cls = 0

        # Image-level
        base_score, base_label = self.RCNN_imageDG_d02(base_feat_d2, torch.tensor([0.]).cuda())
        base_prob = F.log_softmax(base_score, dim=1)
        d2_d02_img_loss_cls = F.nll_loss(base_prob, base_label)
        
        # Instance-level
        instance_sigmoid, same_size_label = self.RCNN_instanceDG_d02(pooled_feat_d2, torch.tensor([0.]).cuda())
        instance_loss = nn.BCELoss()
        d2_d02_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

        # consistency loss
        consistency_prob = F.softmax(base_score, dim=1)[:,1,:,:]
        consistency_prob=torch.mean(consistency_prob)
        consistency_prob=consistency_prob.repeat(instance_sigmoid.size())
        d2_d02_cst_loss=self.consistency_loss(instance_sigmoid,consistency_prob.detach())

        """  ************** taget loss ****************  """

        if self.training:
            rpn_loss_cls = torch.unsqueeze(rpn_loss_cls_d0, 0)
            rpn_loss_bbox = torch.unsqueeze(rpn_loss_bbox_d0, 0)
            RCNN_loss_cls = torch.unsqueeze(RCNN_loss_cls, 0)
            RCNN_loss_bbox = torch.unsqueeze(RCNN_loss_bbox, 0)

        return  rois_d0, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
                RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
                d0_d01_img_loss_cls, d0_d01_ins_loss_cls, d0_d01_cst_loss, \
                d1_d01_img_loss_cls, d1_d01_ins_loss_cls, d1_d01_cst_loss, \
                d0_d02_img_loss_cls, d0_d02_ins_loss_cls, d0_d02_cst_loss, \
                d2_d02_img_loss_cls, d2_d02_ins_loss_cls, d2_d02_cst_loss, \
                d1_d12_img_loss_cls, d1_d12_ins_loss_cls, d1_d12_cst_loss, \
                d2_d12_img_loss_cls, d2_d12_ins_loss_cls, d2_d12_cst_loss

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
