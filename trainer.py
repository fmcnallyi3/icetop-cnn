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
    comp = y['comp']

comp[comp == 1] = 0
comp[comp == 56] = 1

#shuffler = np.random.permutation(len(comp))
#comp = comp[shuffler]
#for i in x_i:
#    i = i[shuffler,...]

#trainCut = (np.random.uniform(size=len(comp)) < 0.85)
#testCut = np.logical_not(trainCut)

print("Beginning to train for %s epochs..." % str(numepochs))

csv_logger = callbacks.CSVLogger('trainedModels/{}'.format(name))
early_stop = callbacks.EarlyStopping(patience=30, restore_best_weights=True) # default -> val_loss
checkpoint = callbacks.ModelCheckpoint('trainedModels/%s.h5' % name,save_best_only=True)
callbacklist = [early_stop, csv_logger,checkpoint]

history = model.fit(
    x=x_i, y=comp, epochs=numepochs,validation_split=0.15,callbacks=callbacklist,verbose=2)

#history = model.fit(
#    x=[x_i[0][trainCut],x_i[1][trainCut]], y=comp[trainCut], epochs=numepochs,validation_data=([x_i[0][testCut],x_i[1][testCut]],comp[testCut]),callbacks=callbacklist)

np.save('trainedModels/%s.npy' % name,prep)
with open('trainedModels/%s.pickle' % name, 'wb') as f:
    pickle.dump(history.history, f)