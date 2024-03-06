import cv2
import matplotlib.pyplot  as plt

#实际应用代码

#img1
img1 = cv2.imread('E:\\image_class-master\\Confusion_Matrix\\num1.png',
                  cv2.IMREAD_COLOR)
img2 = cv2.imread('E:\\image_class-master\\Confusion_Matrix\\num2.png',
                  cv2.IMREAD_COLOR)
img3 = cv2.imread('E:\\image_class-master\\Confusion_Matrix\\num3.png',
                  cv2.IMREAD_COLOR)
img4 = cv2.imread('E:\\image_class-master\\Confusion_Matrix\\num4.png',
                  cv2.IMREAD_COLOR)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)

# plt.rcParams两行是用于解决标签不能显示汉字的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10,8),dpi=50)
plt.subplot(2, 2, 1)
plt.title("编号1")
plt.imshow(img1)

plt.subplot(2, 2, 2)
plt.title("编号2")
plt.imshow(img2)

plt.subplot(2, 2, 3)
plt.title("编号3")
plt.imshow(img2)

plt.subplot(2, 2, 4)
plt.title("编号4")
plt.imshow(img2)

plt.savefig('pingjie.png')
plt.show()

