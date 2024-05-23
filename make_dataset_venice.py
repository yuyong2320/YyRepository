# -*- coding: utf-8 -*-
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import importlib
importlib.reload(sys)

import  h5py
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import json
from image import *


import os

root = '/opt/data/private/Venice'
import glob
import h5py
import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
import scipy
import scipy.spatial
import scipy.ndimage
import PIL.Image as Image

# now generate the ShanghaiA's ground truth
# part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
# part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'venice/train_data', 'images')
part_B_test = os.path.join(root, 'venice/test_data', 'images')
path_sets = [part_B_train]  # 将训练集和测试集放在一起
# print(path_sets)
img_paths = []  # 这是所有图片的路径
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print("-----------------------------------------------------------------------------------------------")


def gaussian_filter_density(gt):  # 这个是高斯核函数
    #print(gt.shape)

    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

    # 构造KDTree寻找相邻的人头位置
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)
    distances, locations = tree.query(pts, k=4)

    print('generate density...')

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            # 相邻三个人头的平均距离，其中beta=0.3
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density



for img_path in img_paths:  # 读取图片并生成密度图
    print(img_path)
    # 获取每张图片对应的mat标记文件
    mat = io.loadmat(img_path.replace('images', 'ground_truth').replace('.jpg', '.mat'))
    #print(mat)
    img = plt.imread(img_path)  # 用于读取一张图片，将图像数据变成数组array
    print(img.shape)

    # # print(img.shape[0])
    # plt.imshow(img)
    # plt.show()
    # 生成密度图
    gt_density_map = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["annotation"]
    #print(gt)
    print(gt.shape)
    for i in range(0, len(gt)):
        # print(gt[i][1])
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:  # 保证这个标注的人头点在图里面
            gt_density_map[int(gt[i][1]), int(gt[i][0])] = 1

    gt_density_map = gaussian_filter_density(gt_density_map)  # 再和高斯核结合一下
    # 保存生成的密度图
    print(gt_density_map.shape)
    with h5py.File(img_path.replace('images', 'ground_truth').replace('.jpg', '.h5'), 'w') as hf:
        hf['density'] = gt_density_map
        hf.close()
    # plt.imshow(Image.open(img_paths[0]))
    # 测试
    print('总数量=', len(gt))
    print('密度图=', gt_density_map.sum())
    #img = plt.imread(img_path)
    #plt.imshow(img)
    #plt.show()
    #img_density = plt.imshow(gt_density_map)
    #plt.show()

print('图片数量：', len(img_paths))

#
