#!/usr/bin/env python
# coding: utf-8

# # Energy Reconstruction Using CNN - Zenith Input

# In[20]:


import numpy as np
import os
import time
from csv import writer
from tensorflow import keras 
from keras import layers, models
from keras.callbacks import CSVLogger, EarlyStopping
from data_tools import load_preprocessed, dataPrep, filterReco


# ## Model Design

# In[21]:


# File directory to folder that holds simulation data 
simPrefix = '/home/mays_k/simdata'

# Sim data to reconstruct (dir produces theta & phi, make sure to transpose)
sim = 'energy'

# Set the number of epochs the model should run for 
numepochs = 100

# Name for model
name = 'baseline'

# Baseline data prep
prep = {'q':None, 't':False, 'normed':True, 'reco':'plane', 'cosz':False}


# In[22]:


# Add identifying number to name
i = 0
# Saves the h5 file of the model in a folder named models 
while(os.path.exists('models/{}.h5'.format(name+str(i)))): 
    i += 1
name += str(i)
print(name)


# In[23]:


# Create model using functional API for multiple inputs

# Input layer 
charge_input = keras.Input(shape=(10,10,2,), name='charge')

# Starts off with three convolutional layers, each one has half the neurons of the previous one 
conv1_layer = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(charge_input)
conv2_layer = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(conv1_layer)
conv3_layer = layers.Conv2D(16, kernel_size=3, padding='same', activation="relu")(conv2_layer)

# Layers are flattened before Zenith information is added 
flat_layer = layers.Flatten()(conv3_layer)
zenith_input = keras.Input(shape=(1,), name='zenith')
concat_layer = layers.Concatenate()([flat_layer, zenith_input])

# The flattened layers and the Zenith layer run through 3 dense layers
dense1_layer = layers.Dense(256, activation='relu')(concat_layer)
dense2_layer = layers.Dense(256, activation='relu')(dense1_layer)
dense3_layer = layers.Dense(256, activation="relu")(dense2_layer)

# This last dense layer is the output of the model
output = layers.Dense(1)(dense3_layer)

model = models.Model(inputs=[charge_input, zenith_input], outputs=output, name=name)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae','mse'])


# In[24]:


model.summary()


# In[25]:


# Load simulation data from files for training
x, y = load_preprocessed(simPrefix, 'train')

# Rotate each event randomly by 0, 90, 180,or 270 degrees
'''for int in range(549773):
    rots = np.random.randint(0,high=4)
    x[int]=np.rot90(x[int],rots)
    
# In[26]:


# Prepare event data
x_i = dataPrep(x, y, **prep)

# Filter NaNs from reconstruction data
filterReco(prep, y, x_i)


# In[27]:


# Logs metrics into a csv file between epochs
csv_logger = CSVLogger('models/{}.csv'.format(name))

# Earlystoping stops the model from training when it starts to overfit to the data
# The main parameter we change is the patience 
early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=0, mode="auto", baseline=None, restore_best_weights=False) 
callbacks = [early_stop, csv_logger]

# Training
print("Now training a model...")
history = model.fit({"charge":x_i[0], "zenith":x_i[1].reshape(-1,1)}, y=y[sim], epochs=numepochs, validation_split=0.15, callbacks=callbacks, verbose=0)


# In[37]:


# Save the model results as a .npy and .h5 file
print("Saving info to models folder")
model.save('models/%s.h5' % name)
np.save('models/%s.npy' % name, prep)

type(history.history['loss'])
# Get the results of the best epoch and write them to a .csv file
num_epoch=len(history.history['loss'])
val_loss=np.min(history.history['val_loss'])
index=history.history['val_loss'].index(val_loss)
loss=history.history['loss'][index]
new_row=[name, num_epoch, loss, val_loss]
with open('models/results.csv', 'a') as f_object:
    csv_writer=writer(f_object)
    csv_writer.writerow(new_row)
f_object.close() 

