#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''


# In[3]:


from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K


# In[6]:


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

batch_size = 128
epochs = 12


# In[7]:


# input image dimensions
img_rows, img_cols = x_train.shape[1:]
x_train.shape


# In[8]:


set(y_train)


# In[9]:


num_classes = len(set(y_train))


# In[10]:


K.image_data_format() #formato de la imagen, si los canales de color van al principio o fin


# In[11]:


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# In[12]:


x_train = x_train.astype('float32') #fuerzo a 32bit de resolución de float en lugar de 64, pero es más rápido
x_test = x_test.astype('float32')
x_train /= 255 #normalizo entre 0 y 1
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[17]:


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes) #lo convierte la salida de 0 a 9 a un vector 001000000 (binario con solo uno activo) (one hot encoding)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train.shape


# In[20]:


model = Sequential()
#primera capa conv
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#capa conv
model.add(Conv2D(64, (3, 3), activation='relu'))
#pooling - version reducida
model.add(MaxPooling2D(pool_size=(2, 2)))
#dropout para regularizar, para desactivar una serie de neuronas al azar para forzar que todas las neuronas sean robustas
model.add(Dropout(0.25))
#flatten
model.add(Flatten())
#paso a hacer el clasificador con una red dense. clasificador feed forward estandar
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#capa de salida
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[21]:


model.summary()


# In[23]:


hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:




