# -*- coding: utf-8 -*-
#images data augmentation
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.misc as misc


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
#    image = tf.image.adjust_gamma(image,gamma=np.random.randint(5,30)*0.1,gain=1)
    return image

def DA_color(image):
    name=['DA_brightness','DA_saturation','DA_hue','DA_contrast','DA_gamma']
    np.random.shuffle(name)
    for i in name:
        image=globals().get(i)(image)
    return image

def read_jpg_images(img_path):
    image_raw_data = tf.gfile.FastGFile(img_path,'rb').read()
    img_data = tf.image.decode_jpeg(image_raw_data)
    #return Tensor
    return img_data

def salt_noise(shape,seed=100):
    noise=np.zeros(shape)
#    rows,cols,dims=shape
#    for i in range(5000):
#        x=np.random.randint(0,rows)
#        y=np.random.randint(0,cols)
#        tmp=np.random.randint(2)
#        if tmp==0:
#            noise[x,y,:]=-1
#        else:
#            noise[x,y,:]=1
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            tmp=np.random.randint(seed)
            if tmp==10:
                noise[i][j][:]=-1
            if tmp==50:
                noise[i][j][:]=1
    noise=tf.convert_to_tensor(noise,dtype=tf.float32)
    return noise


def random_rotate_image_func(image):
    angle = np.random.uniform(low=-30.0, high=30.0)
    return misc.imrotate(image, angle, 'bicubic')

def random_rotate_image(image):
    image = tf.py_func(random_rotate_image_func, [image], tf.uint8)
    image=tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

def preprocess_for_train(image, shape , bbox=None):#shape=[height, width, channels]
    
    tmp=np.random.randint(5)
    if tmp==0:
        return image
    elif tmp==2:
        image = random_rotate_image(image)
        image = tf.image.central_crop(image,np.random.randint(50,100)/100.0)
        image = tf.image.resize_images(image, [shape[0], shape[1]], method=np.random.randint(4))
        image = tf.image.random_flip_left_right(image)
        return image
    elif tmp==4:
        noise = salt_noise(shape)
        image = image+noise
        image = tf.clip_by_value(image,0.0,1.0)
        image = random_rotate_image(image)
        image = tf.image.central_crop(image,np.random.randint(50,100)/100.0)
        image = tf.image.resize_images(image, [shape[0], shape[1]], method=np.random.randint(4))
        image = tf.image.random_flip_left_right(image)
        return image
    else:
        noise = salt_noise(shape)
        image = image+noise
        image = tf.clip_by_value(image,0.0,1.0)
        image = random_rotate_image(image)
        image = tf.image.central_crop(image,np.random.randint(50,100)/100.0)
        image = tf.image.resize_images(image, [shape[0], shape[1]], method=np.random.randint(4))
        image = tf.image.random_flip_left_right(image)
        image = DA_color(image)
        
        return image



if __name__=='__main__':
    start = datetime.datetime.now()
    img_path='images/timg.jpg'
    img_data=read_jpg_images(img_path)
    img_data=tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    print(tf.shape(img_data)[0])
    with tf.Session() as sess:
        img_data = preprocess_for_train(img_data, [596,391,3])
        tt=sess.run(img_data)
    end = datetime.datetime.now()
    plt.imshow(tt)
    plt.show()
    t=end-start
    print(t.microseconds) # 29522 毫秒数


    
    
   