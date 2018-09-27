# -*- coding: utf-8 -*-

import pickle as pkl
import nextpatch
import readmnist
import tensorflow as tf

'''
神经网络层数较多，神经元数量庞大，即可称为深度学习
本案例是全连接神经网络，数据集是mnist手写数字数据集
注意：生成的图片和pickle后缀文件，在重新测试时候，应该删除或者
      剪切进行另外保存，否则会被覆盖。
'''


sess = tf.InteractiveSession()
#十分类，手写数字是0-9，一共10个数字
n_classes=10
'''
dropout变量，dropout是神经网络里防止过拟合最有效和直接的重要方法
dropout的取值为0.0-1.0，例如0.5即为抛弃百分之50的神经网络里的神经元
可以取值为0.25，0.5，0.7，0.8进行比较
dropout可以防止过拟合，但是值太小也会产生欠拟合，应适当取值
'''
keep_prob = tf.placeholder(tf.float32)
trX,trY,teX,teY=readmnist.read_data()

trainds=nextpatch.next_batch_dataset(trX,trY)
'''
x和y变量
变量的维度与读取的数据的维度要相似，None表示不限
'''
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

'''
w的初始化
w的初始化比较重要，w初始化的好，可以使神经外网络的训练更加有效，甚至影响
最后的精确度，因为神经网络很容易陷入局部最优解
stddev是高斯分布的标准差
'''

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
'''
b的初始化
'''
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

'''
神经网络的结构conv_net，weights，biases
层数是可以自定的，而且增加层数的同时，不需要修改其他的函数
神经网络越深越能表示抽象的数据关系，但并不是越深越好，应适当
'''

def mlp_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 784])
    # Fully connected layer
#    dense1 = tf.reshape(_X, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
    dense1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['wd1']), _biases['bd1'])) # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout
    
#    dense2 = tf.reshape(dense1, [-1, _weights['wd2'].get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd2']), _biases['bd2'])) # Relu activation
    dense2 = tf.nn.dropout(dense1, _dropout) # Apply Dropout
    
    # Output, class prediction
    out = tf.add(tf.matmul(dense2, _weights['out']), _biases['out'])
    return out

weights = { 
    'wd1': weight_variable([784, 1024]), 
    'wd2': weight_variable([1024, 2048]),
    'out': weight_variable([1024, n_classes]) 
}

biases = {
    'bd1': bias_variable([1024]),
    'bd2': bias_variable([2048]),
    'out': bias_variable([n_classes])
} 

#神经网络模型计算得出的结果
y_mlp=mlp_net(x, weights, biases, keep_prob)

#将结果和正确的标签进行softmax逻辑回归比较
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_mlp))

'''
对神经网络模型进行训练，AdamOptimizer是增强版的梯度下降方法，能自动调节
learningrate， 1e-4就是learningrate，值就是0.0001
tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)就是老式的
随机梯度下降法，learningrate是固定的，报告中可以拿他们进行比较，生成的图像
是有一定区别的
'''

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#用神经网络模型对数据进行计算，与正确的结果对比，计算精确率
correct_prediction = tf.equal(tf.argmax(y_mlp,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#初试化tensorflow里面的所有变量
sess.run(tf.global_variables_initializer())


'''
测试过程中保存训练集精确度trainAcc和验证集精确度testAcc两个数据列表
用于后面生成数据图表，进行比较
从图像总体趋势看
trainAcc大于testAcc，则欠拟合
trainAcc小于testAcc，则过拟合
trainAcc与testAcc相近为佳
'''

#trainAcc=[]
#with open('trainAcc.pickle','wb') as f:
#            pkl.dump(trainAcc,f)
#testAcc=[]
#with open('testAcc.pickle','wb') as f:
#            pkl.dump(testAcc,f)

'''
next_batch设置为50,而训练数据集为50000
则range(1000)表示遍历完所有的训练集一次，完成一次神经网络的训练，
即为1个epoch，epoch次数建议在10个左右，因为mnist数据集相对比较小，
很快可以达到比较好的精确度。当然epoch越多，计算的精确度越高，同时也
容易导致过拟合。
'''
for i in range(1000):#10000
  '''
  训练数据集对象，每次从中抽取50个样本
  是随机梯度下降算法SGD的重要实现
  '''
  batch = trainds.next_batch(50)
  if i%10 == 0:
    #训练集上的精确度
    train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y: batch[1], keep_prob: 1.0})
    #代价函数，值越小，则结果越好，有时候也成为cost
    loss=sess.run(cross_entropy,feed_dict={
                x:batch[0], y: batch[1], keep_prob: 1.0})
    #训练集上的精确度
    acc=accuracy.eval(feed_dict={
                x: teX, y: teY, keep_prob: 1.0})
#    #以下知识请自行查资料
#    with open('trainAcc.pickle','rb') as f:
#                trainAcc=pkl.load(f)
#                trainAcc.append(train_accuracy)
#    with open('trainAcc.pickle','wb') as f: 
#                pkl.dump(trainAcc,f)
#    with open('testAcc.pickle','rb') as f:
#                testAcc=pkl.load(f)
#                testAcc.append(acc)
#    with open('testAcc.pickle','wb') as f: 
#                pkl.dump(trainAcc,f) 
                
    print("step:{}, training accuracy:{} loss:{},test accuracy:{}".format(
                i, train_accuracy,loss,acc))
  #对神经网络模型进行训练
  train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

#print("The last test accuracy %g"%accuracy.eval(feed_dict={
#    x: teX, y: teY, keep_prob: 1.0}))
#      
#import matplotlib.pyplot as plt
#t1=[t1 for t1 in range(len(trainAcc))]#长度
#plt.plot(t1,testAcc,color='blue',linewidth=2,linestyle='-')
#plt.plot(t1,trainAcc,color='red',linewidth=2,linestyle='-')
#
#plt.ylim(0.0,1.0)#图表范围是0.0至1.0范围
#plt.savefig('total.png')
#plt.show()
#
#plt.plot(t1,testAcc,color='blue',linewidth=2,linestyle='-')
#plt.show()
#plt.plot(t1,trainAcc,color='red',linewidth=2,linestyle='-')
#plt.show()



