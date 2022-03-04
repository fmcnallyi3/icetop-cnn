import sys

import pickle
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


if len(sys.argv) != 3:
    print('Did not specify model to train and/or number of epochs.')
    quit()

numepochs = int(sys.argv[2])
name = str(sys.argv[1])

print("Loading model '%s'..." % name)
model = models.load_model("untrainedModels/"+name+'.h5')
prep = np.load("untrainedModels/"+name+'.npy',allow_pickle=True).item()
#model.summary()

print("Loading simulation data...")
simPrefix = os.getcwd()+'/simdata'
x, y = load_preprocessed(simPrefix, 'train')
x_i = dataPrep(x, y, **prep)

if (prep['reco']!=None):#if zenith is used
    nancut=(x_i[1]==x_i[1])
    x_i[0]=x_i[0][nancut]
    x_i[1]=x_i[1][nancut]
    for key in y.keys():
        y[key] = y[key][nancut]
    energy=y['energy']

print("Beginning to train for %s epochs..." % str(numepochs))

csv_logger = callbacks.CSVLogger('trainedModels/{}'.format(name))
early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True) # default -> val_loss
checkpoint = callbacks.ModelCheckpoint('trainedModels/%s.h5' % name,save_best_only=True)
callbacklist = [early_stop, csv_logger,checkpoint]

history = model.fit(
    x=x_i, y=energy, epochs=numepochs,validation_split=0.15,callbacks=callbacklist)

np.save('trainedModels/%s.npy' % name,prep)
with open('trainedModels/%s.pickle' % name, 'wb') as f:
    pickle.dump(history.history, f)