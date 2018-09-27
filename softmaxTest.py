# -*- coding: utf-8 -*-

import numpy as np

def softmax(x_input):
    x_exp_sum=np.sum(np.exp(x_input))
    result=np.zeros(x_input.shape)
    for i in range(result.shape[0]):
        result[i]=np.exp(x_input[i])/x_exp_sum
    return result

x_input=np.array([1,1,1,1,2,1,1])

ret=softmax(x_input).astype(np.float32)
np.set_printoptions(precision=3)
print(ret)




