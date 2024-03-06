import cv2
import matplotlib.pyplot  as plt


#img2
img1 = cv2.imread('7.jpg',
                  cv2.IMREAD_COLOR)
img2 = cv2.imread('8.jpg',
                  cv2.IMREAD_COLOR)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.subplot(1, 2, 1)
plt.title("原始图")
plt.imshow(img1)

plt.subplot(1, 2, 2)
plt.title("纹理提取图")
plt.imshow(img2)

plt.savefig('对比图.jpg',bbox_inches='tight')
plt.show()