# -*- coding: utf-8 -*-
import tensorflow as tf
import nextpatch
import numpy as np
import os

def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            # lrelu = 1/2 * (1 + leak) * x + 1/2 * (1 - leak) * |x|
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak*x)

def max_pool_2x2(x,k=3):
  return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def fc(x, output_size, stddev=0.02, name="fc"):
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        W = tf.get_variable("W", [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [output_size],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, W) + b


def instance_norm(x):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale',[x.get_shape()[-1]], 
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset

        return out
    
    
def general_conv2d(inputconv, o_d=64, k_s=7, s=1, stddev=0.02, padding="VALID", name="conv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):
        
        conv = tf.contrib.layers.conv2d(inputconv, o_d, k_s, s, padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.01))

        if do_norm:
#            conv = instance_norm(conv)
            conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")
            
        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv

def cnn_network(input_image,ndf=16, name="cnn_network"):
    with tf.variable_scope(name):
        padding='SAME'
        h1 =general_conv2d(input_image, ndf,k_s=3,s=1 ,padding=padding,do_relu=True, name="c1")
        h1=max_pool_2x2(h1)
        print(h1.get_shape().as_list())
        h2 = general_conv2d(h1, o_d=ndf * 2,k_s=3,s=1,padding=padding, do_relu=True,name="c2")
        h2=max_pool_2x2(h2)
        print(h2.get_shape().as_list())

        fc1 = fc(tf.reshape(h2, [-1,7*7*h2.get_shape().as_list()[-1]]),10, name="d_fc1")
            
        print(fc1.get_shape().as_list())
        
        return fc1



def read_data(hotcode=True):
      data_dir='data/mnist'#数据目录
      fd=open(os.path.join(data_dir,'train-images.idx3-ubyte'))
      #转化成numpy数组
      loaded=np.fromfile(file=fd,dtype=np.uint8)
      #根据mnist官网描述的数据格式，图像像素从16字节开始
      
      trX=loaded[16:].reshape((60000,28,28,1)).astype(np.float)
      #print(trX.shape)
      
      #训练label
      fd=open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
      loaded=np.fromfile(file=fd,dtype=np.uint8)
      trY=loaded[8:].reshape((60000)).astype(np.float)
      
      # 测试数据
      fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
      loaded = np.fromfile(file=fd,dtype=np.uint8)
      teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)
      #print(teX.shape)
      
      # 测试 label
      fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
      loaded = np.fromfile(file=fd,dtype=np.uint8)
      teY = loaded[8:].reshape((10000)).astype(np.float)
      
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


#load data
#读取数据集，以np.array的方式输入
trX,trY,teX,teY=read_data(hotcode=True)
#归一化
trX=trX/255.0
teX=teX/255.0

batchsize=50
n_classes=10
#x和y的占位符
with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,shape=[None,28,28,1],name='x')
    y=tf.placeholder(tf.float32,shape=[None,10],name='y')


#神经网络模型计算得出的结果
y_cnn =cnn_network(x)

#将结果和正确的标签进行softmax逻辑回归比较
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=y_cnn))
    tf.summary.scalar('cross_entropy',cross_entropy)
'''
对神经网络模型进行训练，AdamOptimizer是增强版的梯度下降方法，能自动调节
learningrate， 1e-4就是learningrate，值就是0.0001
tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)就是老式的
随机梯度下降法，learningrate是固定的，报告中可以拿他们进行比较，生成的图像
是有一定区别的
'''
#AdamOptimizer
#
train_step = tf.train.RMSPropOptimizer(1e-4).minimize(cross_entropy)
#用神经网络模型对数据进行计算，与正确的结果对比，计算精确率
correct_prediction = tf.equal(tf.argmax(y_cnn,1), tf.argmax(y,1))
with tf.name_scope('tr_acc'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('tr_acc',accuracy)

test_correct_prediction = tf.equal(tf.argmax(y_cnn,1), tf.argmax(y,1))
with tf.name_scope('test_acc'):
    test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))
    tf.summary.scalar('test_acc',test_accuracy)

saver=tf.train.Saver(max_to_keep=3)
savetmp=0.90




model_dir='./CNNtf/model/'
'''test restored!
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(model_dir)
    print(ckpt)
    if ckpt:
        saver.restore(sess,ckpt)
        print("restored model %s" % model_dir)
    else:
        print("fail to restore model %s" % model_dir)
    test_acc=sess.run([test_accuracy],feed_dict={
                    x: teX, y: teY})
    print(test_acc)
'''
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(model_dir)
    if ckpt:
        saver.restore(sess,ckpt)
        print("restored model %s" % model_dir)
    else:
        print("fail to restore model %s" % model_dir)
    merged = tf.summary.merge_all()
    print(1)
    writer=tf.summary.FileWriter('./CNNtfbn/log',sess.graph)
    #初试化tensorflow里面的所有变量
    sess.run(tf.global_variables_initializer())
    print(2)
    steps=0
    for epoch in range(10):#10000
      print(3)
      '''
      训练数据集对象，每次从中抽取50个样本
      是随机梯度下降算法SGD的重要实现
      '''
      trainds=nextpatch.next_batch_dataset(trX,trY)
      for i in range(int(trX.shape[0]/batchsize)):
          print(i)
          batch = trainds.next_batch(50)
          steps=steps+1
          train_step.run(feed_dict={x: batch[0], y: batch[1]})
          
      summary,train_accuracy,loss=sess.run([merged,accuracy,cross_entropy],feed_dict={
                    x:batch[0], y: batch[1]})
      writer.add_summary(summary,steps)
      summary2,test_acc=sess.run([merged,test_accuracy],feed_dict={
                    x: teX, y: teY})
      writer.add_summary(summary2,steps)
      if savetmp<test_acc:
          savetmp=test_acc
          print("saving model!-----------------")
          saver.save(sess,model_dir+'cnnbnmodel.ckpt',global_step=steps)
          
      print("epoches:{}, steps:{}, tr_acc:{:.4f} loss:{:.4f},test_acc:{:.4f}".format(
                        epoch, steps, train_accuracy,loss,test_acc))

    writer.close()


#if __name__=='__main__':
#    img_height = 28
#    img_width = 28
#    img_layer = 1
#    img_size = img_height * img_width
#    batch_size = 12
#    ngf = 32
#    x_input=tf.placeholder(tf.float32, [batch_size,img_height,img_width, img_layer], name="input_A")
#    dnet=cnn_network(x_input)


