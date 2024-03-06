import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

def list_generator(mean, dis, number):  # 封装一下这个函数，用来后面生成数据
    return np.random.normal(mean, dis * dis, number)  # normal分布，输入的参数是均值、标准差以及生成的数量


# 我们生成四组数据用来做实验，数据量分别为70-100
y1 = [0.7822, 0.1483]
y2 = [0.7435, 0.1409]
y3 = [0.7239, 0.1179]
y4 = [0.7355, 0.1019]
y5 = [0.7325, 0.2197]
y6 = [0.7271, 0.0438]
y7 = [0.7029, 0.1264]
# 如果数据大小不一，记得需要下面语句，把数组变为series100
y1 = pd.Series(np.array(y1))
y2 = pd.Series(np.array(y2))
y3 = pd.Series(np.array(y3))
y4 = pd.Series(np.array(y4))
y5 = pd.Series(np.array(y5))
y6 = pd.Series(np.array(y6))
y7 = pd.Series(np.array(y7))
data = pd.DataFrame({"GLCM-DRN": y1, "BiHDM": y2, "EmotionMeter": y3, "DOGNN": y4,"HOG+SVM": y5, "MPGAT": y6, "BiDANN": y7},dtype='float')
data.boxplot(fontsize=12, rot=30)  # 这里，pandas自己有处理的过程，很方便哦。
plt.ylabel("平均分类准确率", fontsize=15)
# plt.xlabel("算法")  # 我们设置横纵坐标的标题。
plt.savefig('res.png', bbox_inches='tight')
plt.show()