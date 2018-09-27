# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os


'''
读取原始数据集
'''

def read_data(hotcode=True):
      data_dir='data/mnist'#数据目录
      fd=open(os.path.join(data_dir,'train-images.idx3-ubyte'))
      #转化成numpy数组
      loaded=np.fromfile(file=fd,dtype=np.uint8)
      #根据mnist官网描述的数据格式，图像像素从16字节开始
      
      trX=loaded[16:].reshape((60000,784))
      #print(trX.shape)
      
      #训练label
      fd=open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
      loaded=np.fromfile(file=fd,dtype=np.uint8)
      trY=loaded[8:].reshape((60000)).astype(np.int64)
      
      # 测试数据
      fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
      loaded = np.fromfile(file=fd,dtype=np.uint8)
      teX = loaded[16:].reshape((10000,784))
      #print(teX.shape)
      
      # 测试 label
      fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
      loaded = np.fromfile(file=fd,dtype=np.uint8)
      teY = loaded[8:].reshape((10000)).astype(np.int64)
      
      if hotcode==True:
          #热编码
          trYL = []
          for i in trY:
                hot=np.zeros(10)
                hot[int(i)]=1
                trYL.append(hot)
          trY=np.array(trYL)
          teYL = []
          for i in teY:
                hot=np.zeros(10)
                hot[int(i)]=1
                teYL.append(hot)
          teY=np.array(teYL)
      return trX,trY,teX,teY


#FloatList , BytesList , Int64List

#trX,trY,teX,teY=read_data(hotcode=False)
#
#'''make tfrecords '''
#
#writer=tf.python_io.TFRecordWriter("tfrecords/mnist.tfrecords")
##writer=tf.python_io.TFRe
#cout=0
#for i in range(trX.shape[0]):
#    print(cout)
#    cout+=1
#    img_raw=trX[i].tobytes()
#    labels=trY[i]
#    example=tf.train.Example(features=tf.train.Features(feature={
#            'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
#            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[labels]))}))
##    example = tf.train.Example(features=tf.train.Features(feature={
##            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
##            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
##        }))
#
#    writer.write(example.SerializeToString())
#    
#writer.close()



#read tfrecords

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image' : tf.FixedLenFeature([], tf.string),
                                           'label' : tf.FixedLenFeature([], tf.int64)
                                       })

    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
#    label = tf.decode_raw(features['label'], tf.)
    label = tf.cast(features['label'], tf.int32)

    return img,label

image,label=read_and_decode("tfrecords/mnist.tfrecords")

img_batch, label_batch=tf.train.shuffle_batch([image,label],batch_size=10,capacity=200,min_after_dequeue=50)
#imgbatch=tf.train.batch([img],batch_size=1)

init = tf.global_variables_initializer()

epochs=1

with tf.Session() as sess:
    coord=tf.train.Coordinator()
    sess.run(init)
    threads = tf.train.start_queue_runners(coord=coord)
    
    for i in range(epochs):
#        imgbatch=tf.train.shuffle_batch([img],batch_size=1,capacity=20,min_after_dequeue=10)
#        imgbatch=tf.train.batch([img],batch_size=1,capacity=20)
       
        for j in range(200):
            img,lab= sess.run([img_batch, label_batch])
            print(img.shape)
            print(lab)
            
            
    coord.request_stop()  
    coord.join(threads)
