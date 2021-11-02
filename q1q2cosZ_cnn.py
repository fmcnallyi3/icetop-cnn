#!/usr/bin/env python
# coding: utf-8

# # Energy Reconstruction Using CNN - Both Charges and Cos(Zenith)

# In[1]:


import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from datetime import datetime

from tensorflow import keras
from keras import layers
from keras import models
from keras import callbacks

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Conv2D, Flatten
#from tensorflow.keras.callbacks import ModelCheckpoint

from data_tools import load_preprocessed, dataPrep, nameModel

simPrefix = os.getcwd()+'\\simdata'


# ## Data Input

# In[2]:


x, y = load_preprocessed(simPrefix, 'train')


# In[3]:


print(x.shape)
print(y.keys())
# each station has 2 tanks, each tank has 2 DOMs (high/log gain)
# each tank measures charge and time
# each station gives 2 charges and 2 times, 4 total pieces of data per station
# stations arranged in 10x10 square lattice, 2 corners of square unused
# charge measured in VEM, vertical equivalent muon

# 'dir' is true direction, rest of dir are reconstruted by simulations
# 'plane_dir' assumes shower is flat plane
# 'laputop_dir' performs likelihood analysis
# 'small_dir' compromises between plane and laputop


# In[4]:


# 85/15 split for training/validation
energy = y['energy']
comp = y['comp']
theta, phi = y['dir'].transpose()
nevents = len(energy)
trainCut = (np.random.uniform(size=nevents) < 0.85)
testCut = np.logical_not(trainCut)


# ## Model Training

# ### Alpha Model
# - Input: no charge merge, no time layers included, normalized data, combined with cosine of zenith angle
# - Layers: Two convolutional layers for charge, then combined with zenith
# - Output: Energy

# In[3]:


# Name for model
key = 'q1q2cosZ'
i = 0
while(os.path.exists('models/model_{}.h5'.format(key+str(i)))):
    i = i + 1
key = key+str(i)
numepochs = 6
# Data preparation: no merging of charge (q), no time layers included (t=False), data normalized from 0-1
prep = {'q':None, 't':False, 'normed':True, 'reco':'plane_', 'cosz':True}


# In[6]:


# Create model using functional API for multiple inputs
charge_input=keras.Input(shape=(10,10,2,),name="charge")

conv1_layer = layers.Conv2D(64,kernel_size=3,activation='relu')(charge_input)
batch1_layer = layers.BatchNormalization()(conv1_layer) # default -> axis = -1, 
drop1_layer = layers.Dropout(0.2)(batch1_layer)

conv2_layer = layers.Conv2D(32,kernel_size=3,activation='relu')(drop1_layer)
batch2_layer = layers.BatchNormalization()(conv2_layer)
drop2_layer = layers.Dropout(0.2)(batch2_layer)

flat_layer = layers.Flatten()(drop2_layer)
zenith_input=keras.Input(shape=(1,),name="zenith")
concat_layer = layers.Concatenate()([flat_layer,zenith_input])
#output = layers.Dense(1)(concat_layer)

dense1_layer = layers.Dense(128)(concat_layer)
batch3_layer = layers.BatchNormalization()(dense1_layer)
drop3_layer = layers.Dropout(0.2)(batch3_layer)

dense2_layer = layers.Dense(128)(drop3_layer)
batch4_layer = layers.BatchNormalization()(dense2_layer)
drop4_layer = layers.Dropout(0.2)(batch4_layer)

output = layers.Dense(1)(drop4_layer)

model = models.Model(inputs=[charge_input,zenith_input],outputs=output,name=key)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae','mse'])

## Old model used for reference
#model = Sequential(name=nameModel(prep, 'test'))  # Automatic naming for flexible assessment later
## Add model layers
#model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(10,10,2)))
#model.add(Conv2D(32, kernel_size=3, activation='relu'))
#model.add(Flatten())
#model.add(Dense(1)) # No activation function for last layer of regression model

## Compile model
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae','mse'])


# In[7]:



# Establish arrays to be trained on
x_i = dataPrep(x, y, **prep)
temp_y = energy


# In[8]:


x_i[1][0].shape


# In[9]:


np.count_nonzero(x_i[1]==None)


# In[10]:


model.summary()


# In[11]:


keras.utils.plot_model(model,"model.png")


# In[ ]:


# Train
csv_logger = callbacks.CSVLogger('models/{}'.format(key))
early_stop = callbacks.EarlyStopping(patience=2) # default -> val_loss
callbacks = [early_stop, csv_logger]
history = model.fit(
    {"charge":x_i[0],"zenith":x_i[1].reshape(-1,1)}, temp_y, epochs=numepochs,validation_split=0.15,callbacks=callbacks)


# In[ ]:


# Save model to file for easy loading
## WHERE ARE YOU SAVING TO?
model.save('models/model_%s.h5' % key)
f = open("results.txt", "a")
now = datetime.now()
f.write("{}\t{}\tepochs:{}\tloss:{},{}\n".format(
    now.strftime("%m/%d/%Y %H:%M:%S"),
    key,
    len(history.history['loss']),
    history.history['loss'][len(history.history['loss'])-1],
    history.history['val_loss'][len(history.history['loss'])-1]
))
f.close()


# ## Your task

# - **Create your own model**
# - Replace the model here w/ *simplified* form of Brandon's model (focus: including zenith)
# - change the zenith input to cosine(zenith) input

# In[11]:


model1 = models.load_model("models/model_q1q2cosZ0.h5")


# In[12]:


for layer in model1.layers:
    print(layer.get_weights())

