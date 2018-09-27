# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def DA_brightness(image):#亮度
    image=tf.image.adjust_brightness(image,np.random.randint(-8,8)*0.001)
    return image
#   
def DA_saturation(image):#饱和度
    image = tf.image.random_saturation(image, lower=0.1, upper=1.0)
    return image

def DA_hue(image):#色相
    image=tf.image.random_hue(image, max_delta=0.25)
    return image

def DA_contrast(image):#对比度
    image=tf.image.random_contrast(image, lower=0.5, upper=1.0)
    return image

def DA_gamma(image):#伽马校正
    image = tf.image.adjust_gamma(image,gamma=np.random.randint(5,30)*0.1,gain=1)
    return image

def DA_color(image):
    name=['DA_brightness','DA_saturation','DA_hue','DA_contrast','DA_gamma']
    np.random.shuffle(name)
    for i in name:
        image=globals().get(i)(image)
    return image

#for i in name:
#    globals().get(i)()


def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)#亮度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)#饱和度
        image = tf.image.random_hue(image, max_delta=0.2)#色相
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)#对比度
    if color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    if color_ordering == 2:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    if color_ordering == 3:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image, 0.0, 1.0)
 
def preprocess_for_train(image, height, width, bbox=None):
#    print(image.dtype)
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
#    if image.dytpe != tf.float32:
#        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
#    distorted_image = tf.image.resize_images(distorted_image, height, width, method=np.random.randint(4))
#    distorted_image = tf.image.random_flip_left_right(distorted_image)
#    distorted_image = distort_color(distorted_image, np.random.randint(4))
    return distorted_image

'''
tf.image.decode_bmp
tf.image.decode_gif
tf.image.decode_jpeg
tf.image.encode_jpeg
tf.image.decode_png
tf.image.encode_png
tf.image.decode_image
'''

def read_jpg_images(img_path):
    image_raw_data = tf.gfile.FastGFile(img_path,'rb').read()
    img_data = tf.image.decode_jpeg(image_raw_data)
    #return Tensor
    return img_data

if __name__=='__main__':
    img_path='images/timg.jpg'
    img_data=read_jpg_images(img_path)
    '''
    tf.image.rgb_to_grayscale
    tf.image.grayscale_to_rgb
    tf.image.hsv_to_rgb
    tf.image.rgb_to_hsv
    tf.image.convert_image_dtype
    '''
    img_data=tf.image.convert_image_dtype(img_data, dtype=tf.float32)
#    print(type(img_data))
    
    '''
    tf.image.flip_up_down
    tf.image.random_flip_up_down
    tf.image.flip_left_right
    tf.image.random_flip_left_right
    tf.image.transpose_image
    tf.image.rot90
    '''
    
#    img_data=tf.image.random_flip_left_right(img_data)
    
    '''
    tf.image.resize_images
    tf.image.resize_area
    tf.image.resize_bicubic
    tf.image.resize_bilinear
    tf.image.resize_nearest_neighbor
    '''
    
#    img_data=tf.image.resize_images(img_data,[299,299])
    
    
    '''
    tf.image.adjust_brightness
    tf.image.random_brightness
    tf.image.adjust_contrast
    tf.image.random_contrast
    tf.image.adjust_hue
    tf.image.random_hue
    tf.image.adjust_gamma
    tf.image.adjust_saturation
    tf.image.random_saturation
    tf.image.per_image_standardization
    '''
    #亮度
#    img_data=tf.image.adjust_brightness(img_data,np.random.randint(-10,10)*0.001)
    #对比度
#    img_data=tf.image.random_contrast(img_data, lower=0.5, upper=1.0)
    #色相
#    img_data=tf.image.random_hue(img_data, max_delta=0.3)
    #饱和度
#    img_data = tf.image.random_saturation(img_data, lower=0.1, upper=1.0)
    #伽马校正
#    img_data = tf.image.adjust_gamma(img_data,gamma=np.random.randint(5,30)*0.1,gain=1)
    
#    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
#    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(img_data), bounding_boxes=bbox)
#    img_data = tf.slice(img_data, bbox_begin, bbox_size)
    
    
#    img_data=DA_color(img_data)
#    img_data=preprocess_for_train(img_data,150,150)
    
    with tf.Session() as sess:
#        boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
#        result = preprocess_for_train(img_data, 299, 299, boxes)
        tt=sess.run(img_data)
        print(type(tt))
    plt.imshow(tt)
    plt.show()
    
    

        
   

