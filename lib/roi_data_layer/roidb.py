"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import numpy as np
from model.utils.config import cfg
from datasets.factory import get_imdb
import PIL
import pdb

# 函数输入：imdb，一个pascal_voc数据类
# 函数输出：无输出。
# 函数执行操作：
'''  通过添加一些对训练有用的派生量来丰富imdb的roidb。 此函数会预先计算每个ROI和每个
  地面真实框之间的地面真实框的最大重叠量。 具有最大重叠的类别也会被记录。
'''
def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """

  # A roidb is a list of dictionaries, each with the following keys:
  #   boxes
  #   gt_overlaps
  #   gt_classes
  #   flipped
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

# 函数输入：一个数据类的roidb字典列表
# 函数输出：ratio_list和ratio_index。ratio_list是roidb中所有roi的宽高比数组（按从小到大排列），ratio_index是按宽高比从小到大排列的原roidb索引序列
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

# 函数输入：一个数据类的roidb字典列表
# 函数输出：经过过滤的roidb
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

# 函数输入：imdb_names，数据集名称，如'voc_2007_trainval'
# 函数输出：imdb, roidb, ratio_list, ratio_index
def combined_roidb(imdb_names, training=True):
  """
  Combine multiple roidbs
  """
  # 函数输入：imdb，一个pascal_voc数据类
  # 函数输出：imdb数据类的roidb
  # A roidb is a list of dictionaries, each with the following keys:
  #   boxes
  #   gt_overlaps
  #   gt_classes
  #   flipped
  def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
      print('Appending horizontally-flipped training examples...')
      imdb.append_flipped_images() # 调用数据母类imdb的方法append_flipped_images()，对图片进行翻转
      print('done')

    print('Preparing training data...')

    prepare_roidb(imdb) # prepare_roidb函数定义在上面几行。函数对imdb数据类的roidb进行丰富
    #ratio_index = rank_roidb_ratio(imdb)
    print('done')

    return imdb.roidb

  # 函数输入：imdb_name，数据集名称，如'voc_2007_trainval'
  # 函数输出：数据集名称为imdb_name的pascal_voc数据类的roidb
  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name) # get_imdb函数定义在datasets.factory中，输入数据集名称，返回初始化好的数据类pascal_voc(trainval, 2007)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD) # 设置数据集处理方法，不太理解，但是cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb) # get_training_roidb函数定义在上面几行。输入为imdb这个pascal_voc类，输出为imdb数据类的roidb
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]

  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names) # get_imdb函数定义在datasets.factory中，输入数据集名称，返回初始化好的数据类pascal_voc(trainval, 2007)

  if training:
    roidb = filter_roidb(roidb) # 函数filter_roidb定义在上面几行。函数输入一个数据类的roidb字典列表，输出经过过滤的roidb

  ratio_list, ratio_index = rank_roidb_ratio(roidb) # 函数rank_roidb_ratio定义在上面几行。函数输入一个数据类的roidb，输出ratio_list和
  # ratio_index。ratio_list是roidb中所有roi的宽高比数组（按从小到大排列），ratio_index是按宽高比从小到大排列的原roidb索引序列

  return imdb, roidb, ratio_list, ratio_index
