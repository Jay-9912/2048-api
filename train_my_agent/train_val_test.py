import numpy as np
import random
x=np.load("G:\\learning_material\\lab\\2048\\2048-api\\data\\x.npy")
y=np.load("G:\\learning_material\\lab\\2048\\2048-api\\data\\y.npy")
#altogether 419966 samples
#x is a 419966*4*4*11 array
#y is a 419966*4 array
index=[i for i in range(len(x))]

random.shuffle(index)
x=x[index]
y=y[index]

ytest=y[-5000:]
yval=y[-15000:-5000]
ytrain=y[:-15000]
xtest=x[-5000:]
xval=x[-15000:-5000]
xtrain=x[:-15000]
np.save("G:\\learning_material\\lab\\2048\\2048-api\\data\\y_train.npy",ytrain)
np.save("G:\\learning_material\\lab\\2048\\2048-api\\data\\y_val.npy",yval)
np.save("G:\\learning_material\\lab\\2048\\2048-api\\data\\y_test.npy",ytest)
np.save("G:\\learning_material\\lab\\2048\\2048-api\\data\\x_train.npy",xtrain)
np.save("G:\\learning_material\\lab\\2048\\2048-api\\data\\x_val.npy",xval)
np.save("G:\\learning_material\\lab\\2048\\2048-api\\data\\x_test.npy",xtest)
print("save completed")



