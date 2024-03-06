import matplotlib.pyplot as plt
import numpy as np

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入统计数据
# waters = ('线性动态-微分熵', '移动平均-微分熵', '线性动态-功率谱密度', '移动平均-功率谱密度')
# delta = [2.706647399, 2.584537572, 2.655, 2.65774]
# theta = [2.679913295, 2.615606936, 2.658959537572254, 2.644537572]
# alpha = [2.731133590237636, 2.668959537572254, 2.7053, 2.6838]
# beta = [2.738800578, 2.675913294797687, 2.72421965317919, 2.72554]
# gamma = [2.7466, 2.735549132947976, 2.738670520231213, 2.70664739884393]

waters = ('1', '2', '3', '4','5','6','7','8','9','10','11','12','13','14','15')
session1 = [0.7714, 0.7090, 0.6970, 0.7319, 0.7457, 0.7266, 0.7519, 0.7698, 0.7370, 0.7694, 0.7493, 0.7507, 0.7289, 0.7485, 0.7560]
session2 = [0.7380, 0.7836, 0.6470, 0.7384, 0.7991, 0.6564, 0.7738, 0.7468, 0.7783, 0.7477, 0.7263, 0.7470, 0.7382, 0.7170, 0.7225]
session3 = [0.7841, 0.7254, 0.7258, 0.7277, 0.6979, 0.7158, 0.7188, 0.7130, 0.7567, 0.7634, 0.7341, 0.7696, 0.7320, 0.7382, 0.7170]


bar_width = 0.1 # 条形宽度
index_session1 = np.arange(len(waters))
index_session2 = index_session1 + bar_width
index_session3 = index_session2  + bar_width

plt.tick_params(labelsize=14)  #刻度字体大小16
# 使用两次 bar 函数画出两组条形图
plt.bar(index_session1, height=session1, width=bar_width, color='red', label='实验1')
plt.bar(index_session2, height=session2, width=bar_width, color='yellow', label='实验2')
plt.bar(index_session3, height=session3, width=bar_width, color='blue', label='实验3')


plt.ylim(ymax=0.95)
plt.legend() # 显示图例
plt.xticks(index_session1 + bar_width/2, waters) # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('平均准确率',size=15,) # 纵坐标轴标题
plt.xlabel('受试者',size=15)
#plt.title('购买饮用水情况的调查结果') # 图形标题

plt.savefig("SVM.png",bbox_inches='tight')
plt.show()
