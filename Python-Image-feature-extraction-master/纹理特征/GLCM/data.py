# coding: utf-8

import numpy as np
from skimage import data
from matplotlib import pyplot as plt
import fast_glcm
from PIL import Image

def main():
    pass


if __name__ == '__main__':
    #main()
    image = r"C:\Users\Administrator\Desktop\Python-Image-feature-extraction-master\eeg_picture\1.jpg";
    img=np.array(Image.open(image).convert('L'))

    mean = fast_glcm.fast_glcm_mean(img)

    # plt.figure(figsize=(8,4.5))
    fs = 15
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(mean)
    plt.savefig(r'C:\Users\Administrator\Desktop\Python-Image-feature-extraction-master\img\output.jpg',bbox_inches='tight')

    plt.show()

