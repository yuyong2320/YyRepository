import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM
import json
from os.path import join
import os
import random

import torch
#%matplotlib inline


def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print ('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density




train_folders ='D:/视频人群计数数据集/venice_414/train_data/ground_truth'
train_all_gruth_lists = []
for root, dirs, files in os.walk(train_folders):
    for file_name in files:
        if file_name.endswith('.mat'):
            train_all_gruth_lists.append(os.path.join(root, file_name))
# print(train_all_gruth_lists)


for gruth_path in train_all_gruth_lists:  # 读取图片并生成密度图
    # print(gruth_path)
    # 获取每张图片对应的mat标记文件
    mat = io.loadmat(gruth_path)
    #print(mat)
    #img = plt.imread(img_path)  # 用于读取一张图片，将图像数据变成数组array
    #print(img.shape)

    # # print(img.shape[0])
    # plt.imshow(img)
    # plt.show()
    # 生成密度图
    gt_density_map = np.zeros((1280,720))
    gt = mat["annotation"]
    #print(gt)
    #print(gt.shape)
    for i in range(0, len(gt)):
        # print(gt[i][1])
        #if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:  # 保证这个标注的人头点在图里面
        gt_density_map[int(gt[i][0]), int(gt[i][1])] = 1

    gt_density_map = gaussian_filter_density(gt_density_map)  # 再和高斯核结合一下
    # 保存生成的密度图
    #print(gt_density_map.shape)
    with h5py.File(gruth_path.replace('.mat', '.h5'), 'w') as hf:
        hf['density'] = gt_density_map
        hf.close()
    # plt.imshow(Image.open(img_paths[0]))
    # 测试
    # print('总数量=', len(gt))
    # print('密度图=', gt_density_map.sum())
    #img = plt.imread(img_path)
    #plt.imshow(img)
    #plt.show()
    #img_density = plt.imshow(gt_density_map)
    #plt.show()

# print('图片数量：', len(img_paths))
#
#



# mat = io.loadmat(root)
#     for i in range(1200,2000):
#         k = np.zeros((640,480))
#         gt = mat['frame'][0,i][0,0][0]
#         for j in range(0,len(gt)):
#             k[int(gt[j,0]),int(gt[j,1])]=1
#         k = gaussian_filter_density(k)
#         with h5py.File(os.path.join('/opt/data/private/FDSTflows/mall_traindata',str(i)+'.h5'),'w') as hf:
#                 hf['density'] = k
#                 hf.close()