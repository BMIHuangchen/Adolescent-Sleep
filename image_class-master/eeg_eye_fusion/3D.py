import numpy as np
from mayavi import mlab
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tvtk.api import tvtk

# eeg = loadmat("E:\\文件\\毕业\\论文\\2022\\论文\\2022论文\\论文资料\\数据集\\SEED_IV\\eeg_feature_smooth\\1\\1_20160518.mat")
# eye = loadmat("E:\\文件\\毕业\\论文\\2022\\论文\\2022论文\\论文资料\\数据集\\SEED_IV\\eye_feature_smooth\\1\\1_20160518.mat")


figure = plt.figure()

ax = Axes3D(figure)

X = np.arange(-10, 10, 0.25)

Y = np.arange(-10, 10, 0.25)

#网格化数据

X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)

Z = np.cos(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()
