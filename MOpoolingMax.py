#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import csv 
from datetime import datetime

from tensorflow import keras
from keras import layers
from keras import models
from keras import callbacks

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout 
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, History 

from data_tools import load_preprocessed, dataPrep, nameModel


# In[2]:


simPrefix = 'C:/Users/MOliv/Documents/PHY460/simFiles'
x, y = load_preprocessed(simPrefix, 'train')


# In[3]:


name = 'MOpoolingMax' 
i = 0
while (os.path.exists('models/{}.h5'.format(name+str(i)))):
    i = i + 1 
name = name+str(i) 
numepochs = 100
prep = {'q':None, 't':False, 'normed':True, 'reco':'small', 'cosz':True}


# In[4]:


charge_input=keras.Input(shape=(10,10,2,))

conv1_layer = layers.Conv2D(64,kernel_size=3,padding='same',activation='relu')(charge_input)
conv2_layer = layers.Conv2D(32,kernel_size=3,padding='same',activation='relu')(conv1_layer)
conv3_layer = layers.Conv2D(16, kernel_size=3, padding='same',activation="relu")(conv2_layer)
Mpool_layer = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None) (conv3_layer)

flat_layer = layers.Flatten()(Mpool_layer)
zenith_input=keras.Input(shape=(1,))
concat_layer = layers.Concatenate()([flat_layer,zenith_input])

dense1_layer = layers.Dense(256,activation='relu')(concat_layer)
dense2_layer = layers.Dense(256,activation='relu')(dense1_layer)
dense3_layer = layers.Dense(256,activation="relu")(dense2_layer)

output = layers.Dense(1)(dense3_layer)

model = models.Model(inputs=[charge_input,zenith_input],outputs=output,name=name)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae','mse'])


# In[5]:


model.summary()


# In[6]:


len(np.sum(x,axis=(1,2,3)))


# In[7]:


th, _ = y['small_dir'].transpose()
len(th)


# In[8]:


x_i = dataPrep(x, y, **prep)


# In[9]:


energy = y['energy']
comp = y['comp']
theta, phi = y['dir'].transpose()
nevents = len(energy)
trainCut = (np.random.uniform(size=nevents) < 0.85)
testCut = np.logical_not(trainCut)
temp_y = energy


# In[10]:


np.count_nonzero(x_i[1]==None)


# In[11]:


csv_logger = CSVLogger('models/{}'.format(name))
early_stop = EarlyStopping(monitor="val_loss",min_delta=0,patience=10,verbose=0,mode="auto",baseline=None,restore_best_weights=False,) 
callbacks = [early_stop, csv_logger]
history = model.fit(
    {"charge":x_i[0],"zenith":x_i[1].reshape(-1,1)}, temp_y, epochs=numepochs,validation_split=0.15,callbacks=callbacks)


# In[ ]:


np.save("MOpoolingMax.npy",prep)
model.save('models/%s.h5' % name)
f = open("results.txt", "a")
now = datetime.now()
f.write("{}\t{}\tepochs:{}\tloss:{},{}\n".format(
    now.strftime("%m/%d/%Y %H:%M:%S"),
    name,
    len(history.history['loss']),
    history.history['loss'][len(history.history['loss'])-1],
    history.history['val_loss'][len(history.history['loss'])-1]
))
f.close()

