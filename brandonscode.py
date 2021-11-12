#!/usr/bin/env python

import matplotlib.pyplot
import argparse
import os
import h5py
import tensorflow as tf
import numpy as np
import math

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers

from Tools.data_tools2 import load_preprocessed, qualityCut, dataPrep, nameModel, name2prep
from Tools.extend_data import extend_data
from Tools.learningratefinder import LearningRateFinder
from Tools.clr_callback import CyclicLR

struct = 'SimplifiedDECO'
parser = argparse.ArgumentParser(description = 'Parser')
parser.add_argument('key', type=str,help='Key to find model')
parser.add_argument('family',type=str,help='Family name of models')
parser.add_argument('--lr',type=str,choices=['clr','min','max','mid'],default='mid',help='Choose learning rate type')
parser.add_argument('--lrmin',type=float,help='Input min learning rate if not find_lr')
parser.add_argument('--lrmax',type=float,help='Input max learning rate if not find_lr')
parser.add_argument('--rtheta',default=False,action='store_true',help='Include direction')
parser.add_argument('--thetatype',type=str,choices=['laputop','plane','small'],help='Choose which direction values to use.',default='plane')
parser.add_argument('--time',default=False,action='store_true',help='Include Time Layers')
parser.add_argument('--timetype',type=str,choices=['qmax','false'],help='Method of merging time layers fed to dataPrep')
parser.add_argument('--chargetype',type=str,choices=['product','sum','tmin','none'],help='Method of merging charge layers fed to dataPrep')
parser.add_argument('--comp',default=False,action='store_true',help='Return composition')
parser.add_argument('--remcomp',type=str,nargs='+',help='Choose which composition types to remove: p, h, o, f')
parser.add_argument('--normcomp',default=False,action='store_true',help='Normalize composition numbers to between 0 and 1')
parser.add_argument('--normalization',default=False,action='store_true')
parser.add_argument('--ext_data',type=str,help='Extend data')
parser.add_argument('--qcut',type=float,help='Maximum charge for charge cut')
parser.add_argument('--zcut',type=float,help='Minimum zenith for zenith cut')
parser.add_argument('--bsize',type=int,help='Batch Size',default=512)
parser.add_argument('--epochs',type=int,help='Number of epochs',default=500)
parser.add_argument('--patience',type=int,help='Early stop patience',default=10)
parser.add_argument('--conv_f',type=int,nargs='+',default=[32,64],help='Features per convolutional layer. Input, at most, two values')
parser.add_argument('--dense_f',type=int,nargs='+',default=[512,512,512],help='Features per dense layer. Input, at most, three values')
parser.add_argument('--conv_do',type=float,nargs='+',default=[.20,.20],help='Dropout for convolutional layers. Input, at most, two values')
parser.add_argument('--dense_do',type=float,nargs='+',default=[.20,.20,.20],help='Dropout for dense layers. Input, at most, three values')
parser.add_argument('--conv_act',type=float,default=.30,help='Activation for convolutional layers')
parser.add_argument('--dense_act',type=float,default=.30,help='Activation for dense layers')
args=parser.parse_args()

#Exits program if lr was not specified at all
if None in [args.lr,args.lrmin,args.lrmax]:
    print('Missing Arguments')
    exit(0)

#Sets learning rate according to input
if args.lr.lower() =='min':
    LR=args.lrmin
elif args.lr.lower() == 'max':
    LR = args.lrmax
else:
    LR = 10**((np.log10(args.lrmin)+np.log10(args.lrmax))/2)

#Assigning Feature/Dropout
if len(args.conv_f)<2 or len(args.dense_f)<3 or len(args.conv_do)<2 or len(args.dense_do)<3:
    print('Missing Features/Dropout Arguments')
    exit(0)
if len(args.conv_do) == 1:
    args.conv_do = [args.conv_do[0] for i in args.conv_f]
if len(args.dense_do) == 1:
    args.dense_do = [args.dense_do[0] for i in args.dense_f]

#Storing Model
filepath = '/home/pascali_b/MachineLearning/Models'
directoryname = '{}/{}/{}/{}'.format(filepath,struct,args.family,args.key)
if not os.path.exists(directoryname):
    os.makedirs(directoryname)
    print('Checkpoint Directory '+directoryname+ ' Created')
#Create Configuration file
cfg_file = 'config.txt'
cfg_file = '{}/{}'.format(directoryname, cfg_file)

complist=['p','h','o','f']
if args.remcomp !=None:    
    for i in args.remcomp:
        if i.lower() in complist:
            complist.remove(i)
print(complist)

#Data Preprocessing
simPrefix = '/home/pascali_b/MachineLearning/Data/Preprocessed4/preprocessed'
x, y = load_preprocessed(simPrefix, 'train',comp=complist)

if args.rtheta==True:
    if args.thetatype=='plane':
        rtheta, _ = y['plane_dir'].transpose()
    elif args.thetatype=='laputop':
        rtheta, _ = y['laputop_dir'].transpose()
    elif args.thetatype=='small':
        rtheta, _ = y['small_dir'].transpose()

# Optional cuts
cut = np.ones(x.shape[0], dtype=bool)

#Quality Cut
if args.qcut != None or args.zcut != None:
    cut *= qualityCut(x, rtheta, qmax=args.qmax, zmax=args.zmax)

#Theta Cut
if args.rtheta==True:
    cut *= (rtheta!=None) * (rtheta==rtheta)

#Cosine
if args.rtheta==True and args.cos==True:
    rtheta=np.cos(rtheta)

# Data preparation
x = x[cut]
if args.comp ==False:
    y = y['energy'][cut]
    y = y.reshape(-1,1)
elif args.comp == True:
    if args.normcomp==True:
        y['comp']=((y['comp']-1)/55)
    y1 = y['energy'][cut]
    y2 = y['comp'][cut]
    y = np.column_stack((y1,y2))
    print('Comp Included')
if args.rtheta==True:
    rtheta = rtheta[cut].astype(float)
    rtheta = rtheta.reshape(-1,1)
    rtheta = np.pi-rtheta
if args.rtheta==False:
    args.thetatype=False
    rtheta=np.ones(2)
#Time
if args.time==False:
    args.timetype=False
#Prep Data
prep = {'q':args.chargetype,'t':args.timetype,'normed':args.normalization,'reco':args.thetatype}
if args.rtheta==True:
    x,rtheta = dataPrep(x,rtheta,q=args.chargetype,t=args.timetype, normed=args.normalization)
if args.rtheta==False:
    x,nothing= dataPrep(x,rtheta,q=args.chargetype,t=args.timetype, normed=args.normalization)

# Extrapolate Data
#if args.ext_data != None:
    #x, y = extend_data(x, y, args.ext_data)
#x,y = extend_data(x,y,'2x')

#85/15 split for training/testing and remove nans
nevents = y.shape[0]
trainCut = (np.random.uniform(size=nevents) < 0.85)
testCut = np.logical_not(trainCut)

#Create Model
ip1 = keras.Input(shape=x[0].shape)
for i,feat in enumerate(args.conv_f):
    if i ==0:
        l = layers.Conv2D(args.conv_f[i],kernel_size=3,padding='same',use_bias=False)(ip1)
    else:
        l = layers.Conv2D(args.conv_f[i],kernel_size=3,padding='same',use_bias=False)(l)
    l = layers.BatchNormalization()(l)
    l = layers.LeakyReLU(alpha=args.conv_act)(l)
    l = layers.Conv2D(args.conv_f[i],kernel_size=3,padding='same',use_bias=False)(l)
    l = layers.BatchNormalization()(l)
    l = layers.LeakyReLU(alpha=args.conv_act)(l)
    l = layers.Dropout(args.conv_do[i])(l)
if args.rtheta==True:
    l = layers.Flatten()(l)
    ip2 = keras.Input(shape=rtheta[0].shape)
    merge = layers.Concatenate()([l,ip2])
else:
    merge=layers.Flatten()(l)
for i,feat in enumerate(args.dense_f):
    if i ==0:
        l = layers.Dense(args.dense_f[i],use_bias=False)(merge)
    else:
        l = layers.Dense(args.dense_f[i],use_bias=False)(l)
    l = layers.BatchNormalization()(l)
    l = layers.LeakyReLU(alpha=args.dense_act)(l)
    l = layers.Dropout(args.dense_do[i])(l)
if args.comp==False:
    output = layers.Dense(1)(l)
if args.comp == True:
    output = layers.Dense(2)(l)
if args.rtheta==True:
    model = keras.Model(inputs = [ip1,ip2], outputs = output,name=nameModel(prep,prefix='brandon'))
elif args.rtheta==False:
    model = keras.Model(inputs = ip1, outputs = output,name=nameModel(prep,prefix='brandon'))

#Create Configuration file
cfg_file = 'config.txt'
cfg_file = '{}/{}'.format(directoryname, cfg_file)
with open(cfg_file, 'w') as config:
    config.write('Configuration file for {} in {} \n'.format(args.key, args.family))
    config.write('Learning Rate is '+str(LR)+'\n')
    config.write('Learning Rate Method is '+args.lr+'\n')
    config.write('Learning Rate Min: ' + str(args.lrmin) + '\n')
    config.write('Learning Rate Max: ' + str(args.lrmax) + '\n')
    config.write('Normalization? ' +str(args.normalization)+'\n')
    config.write('Quality Cut Charge is '+str(args.qcut)+'\n')
    config.write('Zenith Cut is '+str(args.zcut)+'\n')
    config.write('Direction? '+str(args.rtheta)+'\n')
    config.write('Direction Type is: '+str(args.thetatype)+'\n')
    config.write('Time? '+str(args.time)+'\n')
    config.write('Time type is '+str(args.timetype)+'\n')
    config.write('ChargeType is '+str(args.chargetype)+'\n')
    config.write('Composition? '+str(args.comp)+'\n')
    for i in complist:
        config.write(str(i)+' included'+'\n')                                               
    config.write('Extrapolated Data? '+str(args.ext_data)+'\n')                             
    config.write('Number of Epochs is '+str(args.epochs)+'\n')
    config.write('Early Stop Patience is '+str(args.patience)+'\n')
    for i, n in enumerate(args.conv_f):
        config.write('Double Convolution Layer {} Features: {}\n'.format(i, n))
    for i, n in enumerate(args.dense_f):
        config.write('Dense Layer {} Features: {}\n'.format(i, n))
    for i, n in enumerate(args.conv_do):
        config.write('Double Convolution Layer {} Dropout: {}\n'.format(i, n))
    for i, n in enumerate(args.dense_do):
        config.write('Dense Layer {} Dropout: {}\n'.format(i, n)) 
    config.write('Convolution Activation: {}\n'.format(args.conv_act))
    config.write('Dense Activation: {}\n'.format(args.dense_act))
    model.summary(print_fn=lambda x: config.write(x + '\n'))
    config.close()

#Callbacks/save model weights at epochs
csv_logger = CSVLogger('{}/training_log.csv'.format(directoryname))
early_stop = EarlyStopping(monitor='val_loss',patience=args.patience,restore_best_weights=True)

opt = Adam(lr=LR)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])

print('[INFO] training network...')
callbacks = [early_stop, csv_logger]
if args.rtheta==True:
    H = model.fit(x =[x[trainCut],rtheta[trainCut]],y=y[trainCut],batch_size=args.bsize,validation_data=([x[testCut],rtheta[testCut]], y[testCut]),epochs=args.epochs,callbacks=callbacks)
else:
    H = model.fit(x =x[trainCut],y=y[trainCut],batch_size=args.bsize,validation_data=(x[testCut], y[testCut]),epochs=args.epochs,callbacks=callbacks)
# Save model to file
name1=nameModel(prep,prefix='brandon')
model.save('{}/{}.h5'.format(directoryname,name1))
