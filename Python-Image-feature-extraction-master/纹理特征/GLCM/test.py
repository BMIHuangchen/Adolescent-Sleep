import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import cv2
import os
import math

import fast_glcm
from PIL import Image

# 图像灰度延展、直方图均衡化

file_root = r'C:/Users/Administrator/Desktop/Python-Image-feature-extraction-master/eeg_picture/'  # 当前文件夹下的所有图片
file_list = os.listdir(file_root)
save_out = r'C:/Users/Administrator/Desktop/Python-Image-feature-extraction-master/img/'  # 保存图片的文件夹名称
for img_name in file_list:
    img_path = file_root + img_name

    img = r'C:/Users/Administrator/Desktop/Python-Image-feature-extraction-master/eeg_picture/' + img_name
    # img = cv.imread(img_path, -1)
    # out_min = 0
    # out_max = 255

    # in_min = np.min(img)
    # in_max = np.max(img)

    # a = float(out_max - out_min) / (in_max - in_min)
    # b = out_min - a * in_min
    # img_norm = img * a + b
    # img_norm = img_norm.astype(np.uint8)

    out_name = img_name.split('.')[0]
    save_path = save_out + out_name + '.png'

    img = np.array(Image.open(img).convert('L'))

    mean = fast_glcm.fast_glcm_mean(img)

    # plt.figure(figsize=(8,4.5))
    fs = 15
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(mean)
    plt.savefig(save_path,bbox_inches='tight')
    # cv2.imwrite(save_path, img_norm)




# 伽马变换

# file_root = 'E:/program/sxdevice2/SRCNN_en/data/train/contrast/'  #当前文件夹下的所有图片
# file_list = os.listdir(file_root)
# save_out = "E:/program/sxdevice2/SRCNN_en/data/train/contrast/" #保存图片的文件夹名称
# for img_name in file_list:
#     img_path = file_root + img_name
#
#     img = cv.imread(img_path, -1)
#
#     img_norm = img / 255.0  # 注意255.0得采用浮点数
#     img_gamma = np.power(img_norm, 0.4) * 255.0
#     img_gamma = img_gamma.astype(np.uint8)
#
#     out_name = img_name.split('.')[0]
#     save_path = save_out + out_name + '.png'
#     cv2.imwrite(save_path, img_gamma)
