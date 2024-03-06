import matplotlib; matplotlib.use('TkAgg')
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')



name_list=["中性","悲伤","恐惧","开心"]
num_list = [0.67,0.62,0.66,0.65]#分解
# num_list1 = [248.50326,56.595,17.913,283.33]#真实
x = list(range(len(num_list)))
width=0.3;#柱子的宽度
index=np.arange(len(name_list));
plt.bar(index,num_list,width,color='red',tick_label = name_list)
# plt.legend(['分解能耗','真实能耗'],prop=zhfont1,labelspacing=1)

for a,b in zip(index,num_list):   #柱子上的数字显示
 plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=14);

# plt.title('相似百分比')
# plt.ylabel('百分率')

# plt.axis('off')
# plt.gcf().set_size_inches(512 / 100, 512 / 100)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=0.5, bottom=0, right=0.4, left=0.2, hspace=0, wspace=0)

# plt.margins(0, 0)
plt.legend()
plt.savefig('SSIM.png', bbox_inches='tight')
# plt.savefig('SSIM.png')
plt.show()

