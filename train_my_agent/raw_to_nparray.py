import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D,ZeroPadding2D
#from keras.utils.training_utils import multi_gpu_model   #导入keras多GPU函数
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


np.random.seed(10)

x=np.zeros((419966,16,11))
y=np.zeros((419966,1))
with open("G:\\learning_material\\lab\\2048\\2048-api\\data\\2048.txt",'r') as f:
    lines=f.readlines()

    for i,line in enumerate(lines):
        line=line.strip('\n')
        board=line.split(' ')
        for j in range(0,len(board)-1):
            x[i][j][int(board[j])]=1
        y[i][0]=int(board[-1])
y=np_utils.to_categorical(y)
x=np.reshape(x,(419966,4,4,11))
np.save("G:\\learning_material\\lab\\2048\\2048-api\\data\\x.npy",x)
np.save("G:\\learning_material\\lab\\2048\\2048-api\\data\\y.npy",y)

