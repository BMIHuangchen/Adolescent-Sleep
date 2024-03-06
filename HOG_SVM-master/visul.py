from skimage.feature import hog
from skimage import io
from scipy.io import loadmat
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np

#梯度特征可视化与梯度值
img = cv2.cvtColor(cv2.imread('image/EEG_1_1_14_1.jpg'), cv2.COLOR_BGR2GRAY)
print(img.shape)
normalised_blocks, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8),
                                   block_norm='L2-Hys', visualize=True)
io.imshow(hog_image)
io.show()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(hog_image)
plt.xlabel("区间",size=14)
plt.ylabel("梯度值",size=14)
plt.savefig("hog.png",bbox_inches='tight')

plt.show()

#脑电信号归一化
def plot(data, title):
    sns.set_style('dark')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    f, ax = plt.subplots()

    ax.set(ylabel='频率')
    ax.set(xlabel='电位')
    ax.yaxis.get_label().set_fontsize(14)
    ax.xaxis.get_label().set_fontsize(14)
    # ax.set(xlabel='height(blue) / weight(green)')
    # ax.set(title=title)
    sns.distplot(data[:, 0:1], color='blue')
    sns.distplot(data[:, 1:2], color='green')
    #去除白边
    # plt.subplots_adjust(top=0.9,bottom=-0.1,right=0.9,left=-0.1,hspace=0,wspace=0)
    # plt.savefig(title + '.png',bbox_inches='tight')
    plt.show()

# np.random.seed(42)
# height = np.random.normal(loc=168, scale=5, size=1000).reshape(-1, 1)
# weight = np.random.normal(loc=70, scale=10, size=1000).reshape(-1, 1)

# original_data = np.concatenate((height, weight), axis=1)
# plot(original_data, 'Original')

m = loadmat("F:\\matlabFunc\\SEED_IV\\eeg_raw_data\\1\\1_20160518.mat")
# print(m['cz_eeg1'])
original_data = m['cz_eeg1']
# print(original_data)
plot(original_data, 'Original')

# original_data = m['de_movingAve1'][1]
# # print(original_data)
# plot(original_data, 'Original')

# standard_scaler_data = preprocessing.StandardScaler().fit_transform(original_data)
# plot(standard_scaler_data, 'StandardScaler')
#
# min_max_scaler_data = preprocessing.MinMaxScaler().fit_transform(original_data)
# plot(min_max_scaler_data, 'MinMaxScaler')
#
# max_abs_scaler_data = preprocessing.MaxAbsScaler().fit_transform(original_data)
# plot(max_abs_scaler_data, 'MaxAbsScaler')
#
# normalizer_data = preprocessing.Normalizer().fit_transform(original_data)
# plot(normalizer_data, 'normalizer_data')

# robust_scaler_data = preprocessing.RobustScaler().fit_transform(original_data)

# gamme频带上的脑电信号
#DE-movingAve特征提取
m = loadmat("F:\\matlabFunc\\SEED_IV\\eeg_feature_smooth\\1\\1_20160518.mat")
# print(m)
plt.plot(np.array(m['de_movingAve1'][:,:,0]))
# plt.savefig("0.png",bbox_inches='tight')
plt.show()

# svm柱状图
x_data1 = [0.4,   0.5,   0.6,   0.65,  0.7,   0.75,  0.8,   0.9]
y_data1 = [0.7261,0.8364,0.8605,0.923,0.9125,0.8893,0.8929,0.7910]

x_data2 = [0.4,   0.5,   0.6,   0.65,  0.7,   0.75,  0.8,   0.9]
y_data2 = [0.9308,0.9295,0.9304,0.9335,0.9310,0.9309,0.9311,0.93095]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(x_data1,y_data1,color='red', marker='o', mec='r', mfc='w',label='线性核')
plt.plot(x_data2,y_data2,color='blue', marker='^', ms=10,label='径向基核')

# plt.plot(x_data1,y_data1,color='red',marker='o',linewidth=2.0,linestyle='--',label='线性核')
# plt.plot(x_data2,y_data2,color='blue',marker='*',linewidth=2.0,linestyle='-.',label='径向基核')
plt.legend()
# plt.savefig("zexiantu.png",bbox_inches='tight')
plt.show()


#实验结果统计图
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入统计数据
# waters = ('线性动态-微分熵', '移动平均-微分熵', '线性动态-功率谱密度', '移动平均-功率谱密度')
# delta = [0.706647399, 0.584537572, 0.655, 0.65774]
# theta = [0.679913295, 0.615606936, 0.658959537572254, 0.644537572]
# alpha = [0.731133590237636, 0.668959537572254, 0.7053, 0.6838]
# beta = [0.738800578, 0.675913294797687, 0.72421965317919, 0.72554]
# gamma = [0.7466, 0.735549132947976, 0.738670520231213, 0.70664739884393]

waters = ('1', '2', '3', '4','5','6','7','8','9','10','11','12','13','14','15')
session1 = [0.9714, 0.8090, 1.00, 0.9819, 0.8457, 0.9266, 0.9519, 0.9898, 1.00, 0.8794, 0.9994, 0.9807, 0.9589, 0.8885, 0.9860]
session2 = [0.9780, 0.9836, 1.00, 0.9884, 0.9991, 0.6564, 0.9738, 0.9468, 0.9783, 0.9477, 0.9263, 0.9970, 0.7782, 1.00, 0.9625]
session3 = [0.9841, 0.9254, 0.9958, 0.9877, 0.9979, 1.00, 0.9988, 1.00, 0.9967, 0.9634, 0.9841, 0.9996, 1.00, 0.9982, 1.00]


bar_width = 0.1 # 条形宽度
index_session1 = np.arange(len(waters)) # 男生条形图的横坐标
index_session2 = index_session1 + bar_width # 女生条形图的横坐标
index_session3 = index_session2  + bar_width

# 使用两次 bar 函数画出两组条形图
plt.bar(index_session1, height=session1, width=bar_width, color='red', label='实验1')
plt.bar(index_session2, height=session2, width=bar_width, color='yellow', label='实验2')
plt.bar(index_session3, height=session3, width=bar_width, color='blue', label='实验3')


plt.legend(loc='lower center') # 显示图例
plt.xticks(index_session1 + bar_width/2, waters) # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('平均准确率') # 纵坐标轴标题
plt.xlabel('受试者')
#plt.title('购买饮用水情况的调查结果') # 图形标题

# plt.savefig("SVM.png")
plt.show()

