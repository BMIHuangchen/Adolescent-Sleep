from numpy import *
from scipy.io import loadmat

eeg = loadmat("E:\\文件\\毕业\\论文\\2022\\论文\\2022论文\\论文资料\\数据集\\SEED_IV\\eeg_feature_smooth\\1\\1_20160518.mat")

arr=eeg['de_LDS1'][:,:,0]

arr=[]

# for i in file.readlines():
#
#     temp=[]
#
#     for j in i.strip().split('\t'):
#
#         temp.append(float(j))
#
#     arr.append(temp)

import random

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d  import Axes3D

mpl.rcParams['font.size']=10

fig=plt.figure()

ax=fig.add_subplot(111,projection='3d')

xs=range(len(arr))

ys=range(len(arr[0]))

for z in range(len(arr)):

    xs=range(len(arr))

    ys=arr[z]

    color=plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))

    ax.bar(xs,ys,zs=z,zdir='y',color=color,alpha=0.5)

    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xs))

    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ys))

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('copies')

plt.show()