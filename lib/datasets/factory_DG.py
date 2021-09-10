# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.cityscape_DG import cityscape
from datasets.cckw import cckw
from datasets.imagenet import imagenet

import numpy as np


for domin in ['cityscape', 'domain1', 'domain2', 'domain3']:
  for split in ['train_s', 'train_t', 'train_all', 'test_s', 'test_t','test_all']:
    name = '{}_{}_{}'.format(domin, '2007', split)
    __sets[name] = (lambda split=split, domin=domin: cityscape(split, '2007', domin))

for domin in ['watercolor', 'comic', 'clipart', 'kitti']:
  for split in ['train', 'test']:
    name = '{}_{}'.format(domin, split)
    __sets[name] = (lambda split=split, domin=domin: cckw(split, domin))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())