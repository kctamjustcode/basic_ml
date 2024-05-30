##############################
###DNN in TensorFlow KeRas ###
##############################
###1. Load Data and Splot Data
from keras.datasets import mnist, cifar10
from keras.models import Sequential 
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt

 
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
 
###2.Preprocess
X_train = X_train.reshape(50000, 1024*3)     
X_test = X_test.reshape(10000, 1024*3)
X_train = X_train.astype('float32')     
X_test = X_test.astype('float32')     
X_train /= 255    
X_test /= 255
classes = 10
Y_train = np_utils.to_categorical(Y_train, classes)     
Y_test = np_utils.to_categorical(Y_test, classes)
 
###3. Set up parameters
input_size = 3072
batch_size = 100    
hidden_neurons = 400    
epochs = 30
 
###4.Build the model
model = Sequential()     
model.add(Dense(hidden_neurons, input_dim=input_size)) 
model.add(Activation('relu'))     
model.add(Dense(classes, input_dim=hidden_neurons)) 
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy', 
    metrics=['accuracy'], optimizer='adam')
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))
 


plt.plot(history.history['loss'], color='b')
plt.plot(history.history['val_loss'], color='r')
plt.show()
plt.plot(history.history['accuracy'], color='b')
plt.plot(history.history['val_accuracy'], color='r')
plt.show()

pred = model.predict(X_test)
pred = np.argmax(pred, axis= 1)
pred = tf.keras.utils.to_categorical(pred, num_classes= 10)


from sklearn.metrics import classification_report
print(classification_report(Y_test, pred))
