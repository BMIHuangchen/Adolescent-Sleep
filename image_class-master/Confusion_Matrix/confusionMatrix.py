# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

# #人物1
# confusion = np.array(([109,8,0,2],
#                       [68,19,0,15],
#                       [0,44,85,0,],
#                       [24,11,6,34]
#                       ))

#人物2
# confusion = np.array(([119,0,0,0],
#                       [0,88,11,3],
#                       [39,0,90,0,],
#                       [0,0,0,75]
#                       ))

# #人物4
# confusion = np.array(([110,5,0,4],
#                       [0,58,43,0],
#                       [9,28,93,0,],
#                       [0,0,0,75]
#                       ))

#人物7
confusion = np.array(([76,0,2,41],
                      [42,19,0,41],
                      [0,30,99,0,],
                      [0,17,0,58]
                      ))


classes=['中性','悲伤','恐惧','开心']
#画出混淆矩阵
def confusion_matrix(confMatrix):
    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.imshow(confMatrix, cmap=plt.cm.Blues)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(confMatrix))
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    # plt.xticks(indices, [0, 1, 2])
    # plt.yticks(indices, [0, 1, 2])
    plt.xticks(indices, classes,rotation=45)
    plt.yticks(indices, classes)

    plt.colorbar()

    plt.xlabel('预测值')
    plt.ylabel('真实值')
    # plt.title('Persion One')

    # plt.rcParams两行是用于解决标签不能显示汉字的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 显示数据
    for first_index in range(len(confMatrix)):  # 第几行
        for second_index in range(len(confMatrix[first_index])):  # 第几列
            if first_index==second_index:
                plt.text(first_index, second_index, confMatrix[first_index][second_index],va='center',ha='center',color='white')
            else:
                plt.text(first_index, second_index, confMatrix[first_index][second_index], va='center', ha='center')
    # 在matlab里面可以对矩阵直接imagesc(confusion)
    # 显示

    #解决label被挡住问题
    plt.tight_layout()

    plt.savefig('编号7')
    plt.show()


#计算准确率
def calculate_all_prediction(confMatrix):
    '''
    计算总精度,对角线上所有值除以总数
    :return:
    '''
    total_sum=confMatrix.sum()
    correct_sum=(np.diag(confMatrix)).sum()
    prediction=round(100*float(correct_sum)/float(total_sum),2)
    print('准确率:'+str(prediction)+'%')

def calculae_lable_prediction(confMatrix):
    '''
    计算每一个类别的预测精度:该类被预测正确的数除以该类的总数
    '''
    l=len(confMatrix)
    for i in range(l):
        label_total_sum = confMatrix.sum(axis=1)[i]
        label_correct_sum=confMatrix[i][i]
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
        print('精确率:'+classes[i]+":"+str(prediction)+'%')

def calculate_label_recall(confMatrix):
    l = len(confMatrix)
    for i in range(l):
        label_total_sum = confMatrix.sum(axis=0)[i]
        label_correct_sum = confMatrix[i][i]
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
        print('召回率:'+classes[i] + ":" + str(prediction) + '%')


confusion_matrix(confusion)
calculae_lable_prediction(confusion)
calculate_label_recall(confusion)
