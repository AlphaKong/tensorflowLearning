# -*- coding: utf-8 -*-
import nextpatch
import tensorflow as tf
import numpy as np
import os

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
#dropout的占位符
#keep_prob=tf.placeholder(tf.float32)

'''
对w进行初始化，shape是w的模型,stddev是标准偏差
'''
def weight_variable(shape,stddev=0.02):
  initial = tf.truncated_normal(shape, stddev=stddev)
  return tf.Variable(initial)
'''
对b进行初始化，shape是b的模型,一般b初始化为比较小的数即可
'''
def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)
'''
卷积神经网络的卷积计算
'''
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
'''
最大池maxpool
'''
def max_pool_2x2(x,k):
  return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
'''
平均池avgpool
'''  
def avg_pool_2x2(x,k):
  return tf.nn.avg_pool(x, ksize=[1, k, k, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

weights = {  
    #每次pool了之后图片减半,3是指3通道，mnist的例子中1是指1个通道
    #案例中的图像为32*32*3
    'wc1': weight_variable([3, 3, 1, 16]),  #14*14
    'wc2': weight_variable([3, 3, 16, 16]),  #7*7
    'wd1': weight_variable([7 * 7 * 16, 32]),
    'out': weight_variable([32, n_classes]) 
}

biases = {
    'bc1': bias_variable([16]),
    'bc2': bias_variable([16]),
    'bd1': bias_variable([32]),
    'out': bias_variable([n_classes])
} 

#x_image的shape为[-1,28,28,1]
def conv_net(x_image, _weights, _biases):
#    x_image= tf.reshape(x_image, shape=[-1, 28,28,1])
    '''
    第一层
    '''
    h_conv1 = conv2d(x_image, _weights['wc1']) + _biases['bc1']
    h_conv1 = tf.contrib.layers.batch_norm(h_conv1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
    h_conv1 = tf.nn.relu(h_conv1)
    h_pool1 = max_pool_2x2(h_conv1,k=3)
    '''
    第二层
    '''
    h_conv2 = conv2d(h_pool1, _weights['wc2']) + _biases['bc2']
    h_conv2 = tf.contrib.layers.batch_norm(h_conv2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
    h_conv2 = tf.nn.relu(h_conv2)
    h_pool2 = max_pool_2x2(h_conv2,k=3)
#    h_conv2 = tf.nn.relu(conv2d(h_pool1, _weights['wc2']) + _biases['bc2'])
#    h_pool2 = avg_pool_2x2(h_conv2,k=3)
    '''
    第三层
    '''
#    h_conv3 = tf.nn.relu(conv2d(h_pool2, _weights['wc3']) + _biases['bc3'])
#    h_pool3 = avg_pool_2x2(h_conv3,k=2)
    
    h_pool3_flat = tf.reshape(h_pool2, [-1, _weights['wd1'].get_shape().as_list()[0]])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, _weights['wd1']) + _biases['bd1'])
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.85)    
    
    out=tf.matmul(h_fc1_drop, _weights['out']) + _biases['out']
    return out


#神经网络模型计算得出的结果
y_cnn =conv_net(x, weights,biases)

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
    writer=tf.summary.FileWriter('./CNNtf/log',sess.graph)
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
          saver.save(sess,model_dir+'cnnmodel.ckpt',global_step=steps)
          
      print("epoches:{}, steps:{}, tr_acc:{:.4f} loss:{:.4f},test_acc:{:.4f}".format(
                        epoch, steps, train_accuracy,loss,test_acc))

    writer.close()



