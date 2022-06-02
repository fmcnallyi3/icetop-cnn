from data_tools import load_preprocessed, dataPrep
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
import pickle
import os
import sys
from csv import writer
from multiprocessing import Process
from itertools import product


def generate_data_prep(q = None, t = None,  normed = None, reco = None, cosz = None):
    #parameters specific combos
    default_options = { 'q' :  ["mean","sum","product","min","max", None], 
                        't' : ["mean","sum","product","min","max", None, False], #this may be wrong, check in datatool
                        'normed' : [True, False],
                        'reco' : [ None, "plane", "laputop", "small"],
                        'cosz' : [False, True]}

    if q!=None:
        if q=="None":
            default_options["q"]=[None]
        else:
            default_options["q"]=[q]
    if t!=None:
        if t=="None":
            default_options["t"]=[None]
        else:
            default_options["t"]=[t]
    if normed!=None:
        default_options["normed"]=[normed]
    if reco!=None:
        if reco=="None":
            default_options["reco"]=[None]
        else:
            default_options["reco"]=[reco]
    if cosz!=None:
        if cosz==True:
            default_options["cosz"]=[True]
            if None in default_options["reco"]:
                default_options["reco"].remove(None)
        else:
            default_options["cosz"]=[False]

    data_preps = []

    for charge in default_options["q"]:
        for time in default_options["t"]:
            for norm in default_options["normed"]:
                for rec in default_options["reco"]:
                    for cos in default_options["cosz"]:
                        data_preps.append({"q": charge, "t": time, 
                                                                    "normed": norm, "reco": rec,"cosz": cosz })
                        
    
    """
    for charge in default_options["q"]:
        data_preps.append({"q": charge, "t": False, "normed": True, "reco": "plane","cosz":False})
    for time in default_options["t"]:
        data_preps.append({"q": None, "t": time, "normed": True, "reco": "plane","cosz":False})
    for norm in default_options["normed"]:
        data_preps.append({"q": None, "t": False, "normed": norm, "reco": "plane","cosz":False})
    for rec in default_options["reco"]:
        data_preps.append({"q": None, "t": False, "normed": True, "reco": rec,"cosz":False})
    for cos in default_options["cosz"]:
        data_preps.append({"q": None, "t": False, "normed": True, "reco": "plane","cosz":cos})
    """

    return data_preps


def compileModel(name, q=None, t=None, normed=False, reco=None, cosz=False):
        
    if t is None:
        tdim = 2
    elif not t:
        tdim = 0
    else:
        tdim = 1

    if q is None:
        qdim = 2
    elif q is False:
        qdim = 0
    else:
        qdim = 1


    #actual model layers
    ##CHARGE
    if qdim != 0:
        charge_input = keras.Input(shape=(10,10,qdim,),name="charge")
        #charge_input=keras.Input(shape=(10,10,2,))
        convq1_layer = layers.Conv2D(64,kernel_size=3,padding='same',activation='relu')(charge_input)
        convq2_layer = layers.Conv2D(32,kernel_size=3,padding='same',activation='relu')(convq1_layer)
        convq3_layer = layers.Conv2D(16, kernel_size=3, padding='same',activation="relu")(convq2_layer)
        flatq_layer = layers.Flatten()(convq3_layer)

    ##TIME
    if tdim != 0:
        time_input = keras.Input(shape=(10,10,tdim,),name="time")
        #time_input=keras.Input(shape=(10,10,2,))
        convt1_layer = layers.Conv2D(64,kernel_size=3,padding='same',activation='relu')(time_input)
        convt2_layer = layers.Conv2D(32,kernel_size=3,padding='same',activation='relu')(convt1_layer)
        convt3_layer = layers.Conv2D(16, kernel_size=3, padding='same',activation="relu")(convt2_layer)
        flatt_layer = layers.Flatten()(convt3_layer)

    if not (reco is None):
        zenith_input=keras.Input(shape=(1,))
        if tdim != 0:
            if qdim != 0:
                concat_layer = layers.Concatenate()([flatq_layer,flatt_layer,zenith_input])
                dense1_layer = layers.Dense(256,activation='relu')(concat_layer)
            else:
                concat_layer = layers.Concatenate()([flatt_layer,zenith_input])
                dense1_layer = layers.Dense(256,activation='relu')(concat_layer)
        elif qdim != 0:
            concat_layer = layers.Concatenate()([flatq_layer,zenith_input])
            dense1_layer = layers.Dense(256,activation='relu')(concat_layer)
        else:
            dense1_layer = layers.Dense(256,activation='relu')(zenith_input)
    else:
        if tdim != 0:
            if qdim != 0:
                concat_layer = layers.Concatenate()([flatq_layer,flatt_layer])
                dense1_layer = layers.Dense(256,activation='relu')(concat_layer)
            else:
                dense1_layer = layers.Dense(256,activation='relu')(flatt_layer)
        elif qdim != 0:
            dense1_layer = layers.Dense(256,activation='relu')(flatq_layer)
        else:
            print("All inputs removed!")
            return

    dense2_layer = layers.Dense(256,activation='relu')(dense1_layer)
    dense3_layer = layers.Dense(256,activation="relu")(dense2_layer)
    output = layers.Dense(1)(dense3_layer)

    if not (reco is None):
        if tdim != 0:
            if qdim != 0:
                model = models.Model(inputs=[charge_input,time_input,zenith_input],outputs=output,name=name)
            else:
                model = models.Model(inputs=[time_input,zenith_input],outputs=output,name=name)
        elif qdim != 0:
            model = models.Model(inputs=[charge_input,zenith_input],outputs=output,name=name)
        else:
            model = models.Model(inputs=[zenith_input],outputs=output,name=name)
    else:
        if tdim != 0:
            if qdim != 0:
                model = models.Model(inputs=[charge_input,time_input],outputs=output,name=name)
            else:
                model = models.Model(inputs=[time_input],outputs=output,name=name)
        elif qdim != 0:
            model = models.Model(inputs=[charge_input],outputs=output,name=name)
        else:
            print("All inputs removed! Also, first check didn't trigger!")
            return

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae','mse'])
    return model


def train(data_prep, x, y, numepochs=200):
    specs="000plane0"
    name=""
    for _,value in data_prep.items():
        name+=str(value)

    sys.stdout = open('trainedModels/%s/%s.out' %(specs, name),'w')

    #os.nice(10)
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True

    x_i = dataPrep(x, y, **data_prep)

    for key in y:
        if key == "energy":
            energy = y[key]
    for key in data_prep:
        if key == "reco":
            if data_prep[key] != None:
                l = len(x_i)
                nancut=(x_i[l-1]==x_i[l-1])
                for i in range(0,l):
                    x_i[i]=np.array(x_i[i])[nancut]
                energy = np.array(energy)[nancut]

    model = compileModel(name, **data_prep)

    print("Training %s..." % str(data_prep))
    csv_logger = callbacks.CSVLogger('trainedModels/{}.csv'.format(name))
    early_stop = callbacks.EarlyStopping(patience=20, restore_best_weights=True) # default -> val_loss
    checkpoint = callbacks.ModelCheckpoint('trainedModels/%s.h5' % name,save_best_only=True)
    callbacklist = [early_stop, csv_logger,checkpoint]
    history = model.fit(x=x_i, y=energy, epochs=numepochs,validation_split=0.15,callbacks=callbacklist,verbose=2)
    np.save('trainedModels/%s.npy' % name,data_prep)
    with open('trainedModels/%s.pickle' % name, 'wb') as f:
        pickle.dump(history.history, f)

    ##compile info to keep here
    q = data_prep['q']
    t = data_prep['t']
    normed = data_prep['normed']
    reco = data_prep['reco']
    cosz = data_prep['cosz']
    num_epoch=len(history.history['loss'])
    best_training_loss=np.min(history.history['loss'])
    best_val_loss=np.min(history.history['val_loss'])
    new_row=[q,t,normed,reco,cosz,num_epoch, best_training_loss,best_val_loss]

    #save info after every trained model here
    with open('trainedModels/%s/results.csv' %(specs), 'a') as f_object:
        csv_writer=writer(f_object)
        csv_writer.writerow(new_row)
        f_object.close()
    f_object.close()

    """f = open("trainedModels/%s/results.txt" %(specs), "a")
    f.write("{}\tepochs:{}\tloss:{},{}\n".format(
        name,
        len(history.history['loss']),
        np.min(history.history['loss']),
        np.min(history.history['val_loss'])
    ))
    f.close()"""
    
    sys.stdout.close()


