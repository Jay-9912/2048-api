import tensorflow as tf
import keras
from keras.optimizers import*
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import*
#from keras.utils.training_utils import multi_gpu_model   #导入keras多GPU函数
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig('loss.png')
        plt.show()

x_train=np.load("data/x_train.npy")
x_test=np.load("data/x_test.npy")
x_val=np.load("data/x_val.npy")
y_train=np.load("data/y_train.npy")
y_test=np.load("data/y_test.npy")
y_val=np.load("data/y_val.npy")
val=(x_val,y_val)
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(2,2),strides=(1,1),input_shape=(4,4,11),padding='same',kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))#0.25
model.add(Conv2D(filters=64,kernel_size=(2,2),strides=(1,1),padding='same',kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.2))#0.25
model.add(Dense(128,activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)))

model.add(Dense(4,activation='softmax',kernel_regularizer=keras.regularizers.l2(0.01)))
model.compile(loss='categorical_crossentropy',optimizer='Adadelta',metrics=['accuracy'])
history=LossHistory()
train_history = model.fit(x=x_train,  
                          y=y_train,   
                          validation_data=val, 
                          epochs=5,                 
                          batch_size=250,        
                          verbose=2,
                          callbacks=[history])
model.save_weights("testmodel.h5")
print("model save success!")
# show_train_history(train_history,'acc','val_acc')
# show_train_history(train_history, 'loss', 'val_loss')
scores = model.evaluate(x_test,
                        y_test,
                        verbose=0)
history.loss_plot('epoch')
print('accuracy=', scores[1])

