from skimage.measure import compare_ssim
from imageio import imread
import numpy as np

img1 = imread('1/EEG_1_1_14_1.jpg')
img2 = imread('2/EEG_1_1_18_1.jpg')

img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))

print(img2.shape)
print(img1.shape)
ssim = compare_ssim(img1, img2, multichannel=True)

print(ssim)