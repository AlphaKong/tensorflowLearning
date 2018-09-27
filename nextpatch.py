# -*- coding: utf-8 -*-

import numpy as np

#x=np.arange(0,50,dtype=np.int64)
#y=np.arange(1,51,dtype=np.int64)
'''
非常重要，务必好好的理解
面向对象的重要知识
将一个数据集当成一个对象进行操作
'''
class next_batch_dataset(object):
    def __init__(self,data,label):
        self.data=data
        self.label=label
        self._num_examples=data.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        '''
        如果遍历完一遍数据集，就将数据集打乱，然后进行第二次遍历
        '''
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.data = self.data[perm]
            self.label = self.label[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.data[start:end], self.label[start:end]


#ds=next_batch_dataset(x,y,50)
#for i in range(11):
#    d,l=ds.next_batch(10)
#    print("step:{}---d:{}---l:{}".format(i,d,l))

