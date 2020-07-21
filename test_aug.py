#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:53:17 2020

@author: aj611
"""

from PIL import Image
from skimage.io import imread, imshow
from skimage.util import random_noise
from skimage.transform import rotate
from skimage import exposure
import numpy as np


#im = Image.open("krishna.jpg")
#im.rotate(90).show()
import matplotlib.pyplot as plt
#im2 = imread("krishna.jpg") 
#im2 = imread('/dresden/users/aj611/experiments/biomed/feature_ensembling/all_patients_images_resized/Patient_1/10.png')
im2 = imread('12.png')
imshow(im2)
plt.show()
im3 = random_noise(im2, var=0.0025)
imshow(im3)
plt.show()
im4 = rotate(im2, 180)
imshow(im4)
plt.show()
#im5 = exposure.equalize_hist(im2)
#imshow(im5)
#plt.imshow()

#im_new = imread('/dresden/users/aj611/experiments/biomed/feature_ensembling/all_patients_images_resized/Patient_1/10.png')
#imshow(im_new)
#plt.show()

im5 = exposure.equalize_hist(im2)
imshow(im5, cmap='gray')
plt.show()

im6 = exposure.equalize_adapthist(im2, clip_limit=0.012)
imshow(im6, cmap='gray')
plt.show()

p1, p2 = np.percentile(im2, (0.05, 99.05))
im7 = exposure.rescale_intensity(im2, in_range=(p1, p2))
imshow(im7, cmap='gray')
plt.show()

im8 = exposure.adjust_gamma(im2, 2)
imshow(im8, cmap='gray')
plt.show()

im9 = exposure.adjust_log(im2, 1)
imshow(im9, cmap='gray')
plt.show()