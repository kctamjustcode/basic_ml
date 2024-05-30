import numpy as np     
import keras
np.random.seed(0)  #for reproducibility            
 
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten
from keras.utils import np_utils
 
input_size = 784

classes = 10    
 
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
 
X_train = X_train.reshape(60000, 28, 28, 1)     
X_test = X_test.reshape(10000, 28, 28, 1)
 
X_train = X_train.astype('float32')     
X_test = X_test.astype('float32')     
X_train /= 255    
X_test /= 255
 
Y_train = np_utils.to_categorical(Y_train, classes)     
Y_test = np_utils.to_categorical(Y_test, classes)
 
model = Sequential() 
model.add(Convolution2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2)) 
model.add(Dropout(0.15)) 
model.add(Convolution2D(64, (3, 3),  ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2)) 
model.add(Dropout(0.35))  
                
model.add(Flatten())
  
model.add(Dense(100)) 
model.add(Activation('tanh'))  
model.add(Dropout(0.25))      
model.add(Dense(classes)) 
model.add(Activation('softmax'))
      
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', 
    metrics=['accuracy'], optimizer=opt)
 
model.fit(X_train, Y_train, batch_size=128, epochs=30, 
    validation_split = 0.1, verbose=1)
 
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test accuracy:', score[1]) 
