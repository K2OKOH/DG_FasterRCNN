"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lib.datasets
import numpy as np
from lib.model.utils.config import cfg
from lib.datasets.factory_DG import get_imdb
import PIL
import pdb

def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    通过添加一些对训练有用的派生量来丰富imdb的roidd。
    此函数预先计算每个ROI和每个GT框之间在地面真值框上获得的最大重叠量。
    还记录了具有最大重叠的类。
    增加height...标签
    """
    # roidb 是 cityscape.py中的gt_roidb,标注信息,从annotation文件中读取,是列表,表中是字典字典结构如下

    # 'boxes'       : boxes,
    # 'gt_classes'  : gt_classes,
    # 'gt_ishard'   : ishards,
    # 'gt_overlaps' : overlaps,
    # 'flipped'     : False,   # 不翻转
    # 'seg_areas'   : seg_areas

    roidb = imdb.roidb
    if not (imdb.name.startswith('coco')):
        sizes = [PIL.Image.open(imdb.image_path_at(i)).size
            for i in range(imdb.num_images)]

    for i in range(len(imdb.image_index)):
        roidb[i]['img_id'] = imdb.image_id_at(i)
        roidb[i]['image'] = imdb.image_path_at(i)
        if not (imdb.name.startswith('coco')):
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()

        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.    
    
    ratio_list = []
    for i in range(len(roidb)):
        width = roidb[i]['width']
        height = roidb[i]['height']
        ratio = width / float(height)

        if ratio > ratio_large:
            roidb[i]['need_crop'] = 1
            ratio = ratio_large
        elif ratio < ratio_small:
            roidb[i]['need_crop'] = 1
            ratio = ratio_small        
        else:
            roidb[i]['need_crop'] = 0

        ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
        if len(roidb[i]['boxes']) == 0:
            del roidb[i]
            i -= 1
        i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb

def combined_roidb(imdb_names, training=True):   #dataset name
    """
    Combine multiple roidbs
    """
    def get_training_roidb(imdb):
        """Returns a roidb (Region of Interest database) for use in training."""
        # 如果使用翻转,数据增广 数据翻倍
        if cfg.TRAIN.USE_FLIPPED:
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_images()    #  data augment
            # imdb.append_loop_images(14962)
            print('done')

        print('Preparing training data...')

        prepare_roidb(imdb)
        #ratio_index = rank_roidb_ratio(imdb)
        print('done')

        return imdb.roidb

    # 如imdb_name="cityscape_2007_train_s"
    def get_roidb(imdb_name):
        # get_imdb 在 factory.py中定义,通过名称获取imdb(image database)
        # imdb 是数据集标注的实例化对象( !! 例如 imdb = cityscape(train_s, 2007))
        imdb = get_imdb(imdb_name)     # return a pascal_voc dataset object     get_imdb is from factory which contain all legal dataset object

        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        # 由imdb变为roidb
        roidb = get_training_roidb(imdb)
        return roidb

    # 对字符串进行分割，有的数据集中是多个数据集名用‘+’相连，先分开处理。
    # 最终返回GT的roidbs,形式[ 第一种数据集->[{ 第一张图片的字典 },{ 第二张图片的字典 },{...}],第二种数据集-> [{},...],[...]]
    roidbs = [get_roidb(s) for s in imdb_names.split('+')]

    roidb = roidbs[0]   # 这里因为只有一个数据集, 直接取0, 即cityscapes

    # 如果数据集的个数 > 1
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)

    # 如果是在训练过程
    if training:
        # 过滤没有目标框的目标    !!对cityscape : 5950张 -> 5932张
        roidb = filter_roidb(roidb)    # filter samples without bbox

    ratio_list, ratio_index = rank_roidb_ratio(roidb)

    return imdb, roidb, ratio_list, ratio_index  # dataset, roidb dict,ratio_list(0.5,0.5,0.5......2,2,2,), ratio_increase_index(4518,6421,.....)
