# -*- coding: utf-8 -*-
from skimage import io
import numpy as np


def salt_noise(img):
    rows,cols,dims=img.shape
    for i in range(5000):
        x=np.random.randint(0,rows)
        y=np.random.randint(0,cols)
        img[x,y,:]=255
    return img

img_path='images/timg.jpg'
img=io.imread(img_path)

img=salt_noise(img)
io.imshow(img)

