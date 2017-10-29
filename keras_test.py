
##Test Keras
import numpy as np
np.random.seed(33)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

(x_train,y_train),(x_test,y_test) = mnist.load_data()


X_train = x_train.reshape(60000,784).astype('float32')
X_test = x_test.reshape(10000,784).astype('float32')

X_train /=255
X_test /=255

model = Sequential()
model.add(Dense(64),activation='sigmoid',input_shape=(784,))
model.add(Dense(10),activation='softmax')

model.compile(loss='mean_squared_error',optimizer=SGD(lr=0.01),metrics={'accuracy'})

model.fit(X_train,y_train,batch_size=128,epochs=100,verbose=1,validation_data=(X_test,y_test))

