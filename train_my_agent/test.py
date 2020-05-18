import numpy as np
import random

 x=np.load("g/learning_material/lab/2048/2048-api/data/x.npy")
 y=np.load("g/learning_material/lab/2048/2048-api/data/y.npy")

#altogether 419966 samples
#x is a 419966*4*4*11 array
#y is a 419966*4 array

x=x[:10000]
y=y[:10000]
index=[i for i in range(len(x))]

random.shuffle(index)
x=x[index]
y=y[index]

ytest=y[-1000:]
yval=y[7000:9000]
ytrain=y[:7000]
xtest=x[-1000:]
xval=x[7000:9000]
xtrain=x[:7000]
np.save("G:\\learning_material\\lab\\2048\\2048-api\\test\\y_train.npy",ytrain)
np.save("G:\\learning_material\\lab\\2048\\2048-api\\test\\y_val.npy",yval)
np.save("G:\\learning_material\\lab\\2048\\2048-api\\test\\y_test.npy",ytest)
np.save("G:\\learning_material\\lab\\2048\\2048-api\\test\\x_train.npy",xtrain)
np.save("G:\\learning_material\\lab\\2048\\2048-api\\test\\x_val.npy",xval)
np.save("G:\\learning_material\\lab\\2048\\2048-api\\test\\x_test.npy",xtest)
print("save completed")



