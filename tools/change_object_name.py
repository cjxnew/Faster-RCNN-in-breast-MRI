import os
# import yolo.config as cfg
import xml.etree.ElementTree as ET

#@ ------------这个脚本修改原先的benign-malignant二类.xml标签文件修改为单类mass标签文件 -----------------------------------------

# #@ 以前用来检查/修正错误标签名的部分
# image_index = []
# txtname = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit', 'VOC2007','ImageSets', 'Main', 'total_sample.txt')
# #@@@ image_index是保存了test.txt里的所有病人编号的列表
# with open(txtname, 'r') as f:
#     image_index = [x.strip() for x in f.readlines()]
#
# abnormal_num = 0
# abnormal_dict = {}
# abnormal_list = []
# for index in image_index:
#     # @@@ 图片编号对应的xml文件
#     filename = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit', 'VOC2007', 'Annotations', index + '.xml')
#     # @@@ ET.parse函数把xml文件解析为元素树，返回一个元素树实例
#     tree = ET.parse(filename)
#     # @@@ 找到元素树里所有的'object'内容
#     objs = tree.findall('object')
#     # @@@ 对xml文件元素树里的每一个'object'
#     for obj in objs:
#         obj_name = obj.find('name').text.lower().strip()
#         if (obj_name != 'benign') and (obj_name != 'malignant'):
#             abnormal_num += 1
#             abnormal_list.append(index)
#             if obj_name in abnormal_dict.keys():
#                 abnormal_dict[obj_name] += 1
#             else:
#                 abnormal_dict[obj_name] = 0
#
# print ('异常标签名有{}个'.format(abnormal_num))
# print ('异常标签为：')
# print (abnormal_dict)
#
#
# # ----------------- 单独改一改 ---------------------------
# ind = '107701'
# filename = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit', 'VOC2007', 'Annotations', ind + '.xml')
# # @@@ ET.parse函数把xml文件解析为元素树，返回一个元素树实例
# tree = ET.parse(filename)
# # @@@ 找到元素树里所有的'object'内容
# objs = tree.findall('object')
# for obj in objs:
#     obj_name = obj.find('name').text.lower().strip()
#     if (obj_name != 'benign') and (obj_name != 'malignant'):
#         name = obj.find('name')
#         name.text = 'malignant'
#         tree.write(filename)

# ------------- 对所有的.xml的所有object名字进行修改，改为'mass' ----------------------------------
image_index = []
# txtname = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit', 'VOC2007','ImageSets', 'Main', 'total_sample.txt')
txtname = '/home/cjx/chenjixin/faster-rcnn.pytorch-pytorch-1.0/data/VOCdevkit2007/VOC2007/ImageSets/Main/图片文档库/total_sample.txt'
#@@@ image_index是保存了test.txt里的所有病人编号的列表
with open(txtname, 'r') as f:
    image_index = [x.strip() for x in f.readlines()]

xml_path = '/home/cjx/chenjixin/faster-rcnn.pytorch-pytorch-1.0/data/VOCdevkit2007/VOC2007'
for ind in image_index:
    filename = os.path.join(xml_path, 'Annotations_only_mass', ind + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    for obj in objs:
        # obj_name = obj.find('name').text.lower().strip()
        # if (obj_name != 'benign') and (obj_name != 'malignant'):
        #     name = obj.find('name')
        #     name.text = 'malignant'
        #     tree.write(filename)
        name = obj.find('name')
        name.text = 'mass'
        tree.write(filename)
