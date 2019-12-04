#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras


# In[2]:


mnist = tf.keras.datasets.mnist


# In[3]:


(x_train, y_train),(x_test, y_test) = mnist.load_data() #y_test e y_test no nos interesan para un autoencoder
x_train, x_test = x_train / 255.0, x_test / 255.0 #normalizo (entre 0 y 1)


# In[4]:


x_train.max()


# In[5]:


x_train.min()


# In[6]:


x_train.shape


# In[28]:


encoder = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), #entrada, con flaten paso de 28x28 a un vector y le digo el shape
  tf.keras.layers.Dense(50, activation=tf.nn.relu), #encoder, capa de activacíon
  tf.keras.layers.Dense(2, activation=tf.nn.relu), #coded data que es los digitos de 0 a 9, reduzo a dos dimensones
])

decoder = tf.keras.models.Sequential([
  tf.keras.layers.Dense(50, input_shape=(2,),activation=tf.nn.relu),#decoder la salida entre 0 y 1
  tf.keras.layers.Dense(28*28,activation=tf.nn.sigmoid), #salida
  tf.keras.layers.Reshape((28,28)) #hago una nueva capa reshape para sacar la imagen de 28x28
])

model = tf.keras.models.Sequential([
    encoder,
    decoder
])


# In[29]:


model.compile(optimizer='Adam',
              loss=['binary_crossentropy'],
              metrics=['mse']) #error cuadratico medio


# In[30]:


model.fit(x_train, x_train,    
         validation_data=(x_test, x_test),
          validation_steps= len(x_test),
          epochs=5)


# In[31]:


#aplicar el modelo a las imagenes de test, es decir reducir una imagen y volver a generarla


# In[32]:


x_test_pred = model .predict(x_test)


# In[33]:


x_test.shape


# In[34]:


get_ipython().run_line_magic('matplotlib', 'widget')


# In[44]:


import matplotlib.pyplot as plt


# In[45]:


plt.figure()
plt.imshow(x_test[0])
plt.figure()
plt.imshow(x_test_pred[0])


# In[46]:


plt.figure()
plt.imshow(x_test[100])
plt.figure()
plt.imshow(x_test_pred[100])


# In[47]:


plt.figure()
plt.imshow(x_test[5000])
plt.figure()
plt.imshow(x_test_pred[5000])


# In[48]:


codigo = encoder.predict(x_test)


# In[49]:


codigo.shape #paso a dos dimensiones


# In[50]:


#resultado de una reducción a dos dimensiones
#Puedo generar imagenes con dos coordinadas, he hecho sintesis
plt.figure()
plt.scatter(codigo[:,0],codigo[:,1],c=y_test)
plt.colorbar()


# In[51]:


codigo_sintetica = [[6,30]]


# In[57]:


import numpy as np
x_sintetica = decoder.predict(np.array(codigo_sintetica))


# In[59]:


plt.figure()
plt.imshow(x_sintetica[0])


# In[60]:


x_sintetica.shape


# In[61]:


codigo_sintetica = [[60,30]]
x_sintetica = decoder.predict(np.array(codigo_sintetica))
plt.figure()
plt.imshow(x_sintetica[0])


# In[63]:


codigo_sintetica = [[20,5]]
x_sintetica = decoder.predict(np.array(codigo_sintetica))
plt.figure()
plt.imshow(x_sintetica[0])


# In[77]:


f, ax = plt.subplots(10,10)
for ix, c1 in enumerate(np.linspace(0,50,10)):
    for iy, c2 in enumerate(np.linspace(0,80,10)):
        x_sintesis = decoder.predict(np.array[[c1,c2]])
        ax[iy][ix].imshow(x_sintesis[0])


# In[78]:


#añado ruido a las imagens
ruido = 0.01*np.random.randn(60000,28,28) #ruido para todas las imagenes
x_train_ruido = x_train + ruido
x_train_ruido.shape


# In[80]:


encoder = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), #entrada, con flaten paso de 28x28 a un vector y le digo el shape
  tf.keras.layers.Dense(50, activation=tf.nn.relu), #encoder, capa de activacíon
  tf.keras.layers.Dense(32, activation=tf.nn.relu), #coded data que es los digitos de 0 a 9, reduzo a dos dimensones
])

decoder = tf.keras.models.Sequential([
  tf.keras.layers.Dense(50, input_shape=(32,),activation=tf.nn.relu),#decoder la salida entre 0 y 1
  tf.keras.layers.Dense(28*28,activation=tf.nn.sigmoid), #salida
  tf.keras.layers.Reshape((28,28)) #hago una nueva capa reshape para sacar la imagen de 28x28
])

model = tf.keras.models.Sequential([
    encoder,
    decoder
])


# In[81]:


model.compile(optimizer='Adam',
              loss=['binary_crossentropy'],
              metrics=['mse']) #error cuadratico medio


# In[83]:


x_test.shape
ruido = 0.01*np.random.randn(10000,28,28) #ruido para todas las imagenes
x_test_ruido = x_test + ruido


# In[84]:


model.fit(x_train_ruido, x_train,    
         validation_data=(x_test_ruido, x_test),
          validation_steps= len(x_test),
          epochs=5)


# In[86]:


x_sin_ruido = model.predict(x_test_ruido)
plt.figure()
plt.imshow(x_test_ruido[0])
plt.figure()
plt.imshow(x_sin_ruido[0])


# In[92]:


idx = 7777
f,ax = plt.subplots(1,2)
ax[0].imshow(x_test_ruido[idx])
ax[1].imshow(x_sin_ruido[idx])


# In[ ]:




