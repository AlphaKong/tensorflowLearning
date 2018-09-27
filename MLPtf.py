# -*- coding: utf-8 -*-
import nextpatch
import readmnist
import tensorflow as tf
import numpy as np


#load data
#读取数据集，以np.array的方式输入
trX,trY,teX,teY=readmnist.read_data(hotcode=True)
#归一化
trX=trX/255.0
teX=teX/255.0

batchsize=50
#x和y的占位符
with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,shape=[None,784],name='x')
    y=tf.placeholder(tf.float32,shape=[None,10],name='y')
#dropout的占位符
#keep_prob=tf.placeholder(tf.float32)


def mlp_layer(input_x,input_dim,output_dim,name='',drop_out=1.0,is_output=False):
    with tf.name_scope(name):
        weights=tf.Variable(tf.truncated_normal([input_dim,output_dim],stddev=0.02),name='weights_{}'.format(name))
        bias=tf.Variable(tf.truncated_normal([output_dim],stddev=0.02),name='bias_{}'.format(name))
        if is_output==False:
            mlp_l=tf.nn.relu(tf.add(tf.matmul(input_x,weights),bias))
            mlp_l=tf.nn.dropout(mlp_l,drop_out)
        else:
            mlp_l=tf.add(tf.matmul(input_x,weights),bias)
        return mlp_l


def mlp_network(input_x):
    _x=tf.reshape(input_x,shape=[-1,784])
    #256 0.2
    #128 0.5
    #64 0.8
    
    layer_1=mlp_layer(_x,784,64,name='l1',drop_out=0.8)
    
#    layer_2=mlp_layer(layer_1,256,64, name='l2', drop_out=0.3)
    
#    layer_3=mlp_layer(layer_2,64,32, name='l3', drop_out=0.9)
    
    net=mlp_layer(layer_1,64,10,name='out_layer',is_output=True)
    
    return net


#神经网络模型计算得出的结果
y_mlp = mlp_network(x)

#将结果和正确的标签进行softmax逻辑回归比较
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=y_mlp))
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
correct_prediction = tf.equal(tf.argmax(y_mlp,1), tf.argmax(y,1))
with tf.name_scope('tr_acc'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('tr_acc',accuracy)

test_correct_prediction = tf.equal(tf.argmax(y_mlp,1), tf.argmax(y,1))
with tf.name_scope('test_acc'):
    test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))
    tf.summary.scalar('test_acc',test_accuracy)

saver=tf.train.Saver(max_to_keep=3)
savetmp=0.90




model_dir='./MLPtf/model/'
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
    print(ckpt)
    if ckpt:
        saver.restore(sess,ckpt)
        print("restored model %s" % model_dir)
    else:
        print("fail to restore model %s" % model_dir)
    merged = tf.summary.merge_all()
    writer=tf.summary.FileWriter('./MLPtf/log',sess.graph)
    #初试化tensorflow里面的所有变量
    sess.run(tf.global_variables_initializer())
    
    steps=0
    for epoch in range(10):#10000
      '''
      训练数据集对象，每次从中抽取50个样本
      是随机梯度下降算法SGD的重要实现
      '''
      trainds=nextpatch.next_batch_dataset(trX,trY)
      for i in range(int(trX.shape[0]/batchsize)):
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
          saver.save(sess,model_dir+'mlp.ckpt',global_step=steps)
          
      print("epoches:{}, steps:{}, tr_acc:{:.4f} loss:{:.4f},test_acc:{:.4f}".format(
                        epoch, steps, train_accuracy,loss,test_acc))

    writer.close()



