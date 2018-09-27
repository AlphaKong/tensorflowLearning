# -*- coding: utf-8 -*-
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

'''
cifar10是一个32*32的彩色图像数据集，图片数据是3通道，计算量大
'''

trainfilelist=['data/cifar-10-batches-py/data_batch_1',
          'data/cifar-10-batches-py/data_batch_2',
          'data/cifar-10-batches-py/data_batch_3',
          'data/cifar-10-batches-py/data_batch_4',
          'data/cifar-10-batches-py/data_batch_5']
testfilelist=['data/cifar-10-batches-py/test_batch']

def load_CIFAR(filelist):
    cifardata=[]
    cifarlabel=[]
    for fl in filelist:
        print(fl)
        with open(fl, 'rb') as f:
           datadict = pickle.load(f, encoding='bytes') 
           for d in datadict[b'data']:
               cifardata.append(np.reshape(np.reshape(d,[3,1024]).T,[32,32,3]))
           for l in datadict[b'labels']:
               hot=np.zeros(10)
               hot[int(l)]=1
               cifarlabel.append(hot)
    data=np.array(cifardata,dtype=np.float)
    data=(data-128)/128
    label=np.array(cifarlabel,dtype=np.float)
    return data,label

dirpath='data/cifarnpy/'

def read_cifar10_traindataset():
    ctrX=np.load(dirpath+'cdata.npy')
    ctrY=np.load(dirpath+'clabel.npy')
    return ctrX,ctrY

def read_cifar10_testdataset():
    cteX=np.load(dirpath+'ctdata.npy')
    cteY=np.load(dirpath+'ctlabel.npy')
    return cteX,cteY  


if __name__=="__main__":
      d,l=load_CIFAR(trainfilelist)      
      np.save(dirpath+'cdata',d)
      np.save(dirpath+'clabel',l)
      td,tl=load_CIFAR(testfilelist)
      np.save(dirpath+'ctdata',td)
      np.save(dirpath+'ctlabel',tl)
#
#np.save('cdata',d)
#np.save('clabel',l)
#d=np.load('cdata.npy')
##l=np.load('data/cifarnpy/clabel.npy')
#print(d.shape)
#
#print(d[0])

#d,l=load_CIFAR(testfilelist)
#np.save('ctdata',d)
#np.save('ctlabel',l)
#
#print(d.shape)
'''
print(d.shape)
print(l.shape)
import nextpatchtest
myd=nextpatchtest.next_batch_dataset(d,l,d.shape[0])
d1,l1=myd.next_batch(100)
plt.imshow(d1[0])
plt.imsave('1.png',d1[0])
plt.show()
d2,l2=myd.next_batch(100)
plt.imshow(d2[0])
plt.imsave('2.png',d2[0])
plt.show()
'''
















