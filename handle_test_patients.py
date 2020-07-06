import os
import xml.etree.ElementTree as ET
import shutil

txt_name = 'cvTest2.txt'
txt_path = '/home/cjx/chenjixin/faster-rcnn.pytorch-pytorch-1.0/data/VOCdevkit2007/VOC2007/ImageSets/Main/图片文档库/5倍交叉验证/'
jpg_path = '/home/cjx/chenjixin/faster-rcnn.pytorch-pytorch-1.0/data/VOCdevkit2007/VOC2007/JPEGImages'

image_index = []
txtname = os.path.join(txt_path, txt_name)
with open(txtname, 'r') as f:
    image_index = [x.strip() for x in f.readlines()]

patients_slices = []
a_patient_name = ''
a_patient_slices = []

# 下面选出每个病人包含的切片名
for ind in image_index:
    ind_name = ind[1:4]
    if ind_name != a_patient_name:
        if a_patient_name != '':
            patients_slices.append(a_patient_slices)
        a_patient_slices = []
        a_patient_name = ind_name
        a_patient_slices.append(ind)
    else:
        a_patient_slices.append(ind)

import torch
a = torch.tensor([1,5,62,54])
m = torch.max(a)
print(m)