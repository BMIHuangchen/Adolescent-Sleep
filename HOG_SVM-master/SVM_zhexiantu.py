import matplotlib.pyplot as plt

x_data1 = [0.4,   0.5,   0.6,   0.65,  0.7,   0.75,  0.8,   0.9]
y_data1 = [0.7261,0.8364,0.8605,0.923,0.9125,0.8893,0.8929,0.7910]

x_data2 = [0.4,   0.5,   0.6,   0.65,  0.7,   0.75,  0.8,   0.9]
y_data2 = [0.9308,0.9295,0.9304,0.9335,0.9310,0.9309,0.9311,0.93095]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(x_data1,y_data1,color='red', marker='o', mec='r', mfc='w',label='线性核')
plt.plot(x_data2,y_data2,color='blue', marker='^', ms=10,label='径向基核')

# plt.plot(x_data1,y_data1,color='red',marker='o',linewidth=2.2,linestyle='--',label='线性核')
# plt.plot(x_data2,y_data2,color='blue',marker='*',linewidth=2.2,linestyle='-.',label='径向基核')
plt.legend()
plt.savefig("zexiantu.png",bbox_inches='tight')
plt.show()
