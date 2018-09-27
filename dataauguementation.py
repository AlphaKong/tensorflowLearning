# -*- coding: utf-8 -*-
from skimage import io,data
import numpy as np
import tensorflow as tf

img=data.logo()
img=np.array(img,dtype=np.float)
img=(img-128)/128
#print(img)
new_img = tf.cast(img, tf.float32)

print(new_img.shape)
print(type(new_img))
##new_img=tf.reshape(new_img, [1, 300, 300, 4])
##print(new_img.shape)
#new_img = tf.random_crop(new_img, size=(200, 200, 3)) #从原图像中切割出子图像
#new_img = tf.image.random_brightness(new_img, max_delta=63) #随机调节图像的亮度
new_img = tf.image.random_flip_left_right(new_img) #随机地左右翻转图像

#new_img=tf.image.random_flip_up_down(new_img)
#new_img = tf.image.random_contrast(new_img, lower=0.2, upper=1.8) #随机地调整图像对比度
#final_img = tf.image.per_image_standardization(new_img) #对图像进行whiten操作，目的是降低输入图像的冗余性，尽量去除输入特征间的相关性

with tf.Session() as sess:
      #print(new_img.eval(session=sess))
      io.imshow(new_img.eval(session=sess))
#print(final_img.shape)
#io.imshow(final_img.eval())

