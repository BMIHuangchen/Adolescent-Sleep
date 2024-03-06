import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scipy.io import loadmat

eeg = loadmat("E:\\文件\\毕业\\论文\\2022\\论文\\2022论文\\论文资料\\数据集\\SEED_IV\\eeg_feature_smooth\\1\\1_20160518.mat")
eye = loadmat("E:\\文件\\毕业\\论文\\2022\\论文\\2022论文\\论文资料\\数据集\\SEED_IV\\eye_feature_smooth\\1\\1_20160518.mat")
# print(eeg['de_LDS1'][:,:,4])
# print(eye['eye_1'][0,:])
# result = eeg['de_LDS1'][:,:,4].*eye['eye_1'][0,:]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(np.array(eeg['de_LDS1'][:,:,4]*eye['eye_1'][0,:]/31/3/1.1),label='特征融合')
plt.plot(np.array(eeg['de_LDS1'][:,:,4]),label='单一脑电')
plt.savefig("fusion.png")
plt.show()

# test1 = [[1,2],
#         [3,4]]
# test2 = [[1,2],
#         [3,4]]
# print(eeg['de_LDS1'][:,:,4]*eye['eye_1'][0,:])