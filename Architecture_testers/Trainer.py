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

#naming convention for results
it=0
while(os.path.isfile('DataResults/results%.csv' %it)):
    it+=1

def generate_data_prep(q = None, t = None,  normed = None, reco = None, cosz = None):
    #parameters specific combos
    default_options = { 'q' :  ["mean","sum","product","min","max", None], 
                        't' : ["mean","sum","product","min","max", None, False], #this may be wrong, check in datatool
                        'normed' : [True, False],
                        'reco' : [ None, "plane", "laputop", "small"],
                        'cosz' : [False, True]}

    if q!=None:
        default_options["q"]=[q]
    if t!=None:
        default_options["t"]=[t]
    if normed!=None:
        default_options["normed"]=[normed]
    if reco!=None:
        default_options["reco"]=[reco]
    if cosz!=None:
        default_options["cosz"]=[cosz]


    data_preps = []

    """for charge in default_options["q"]:
        for time in default_options["t"]:
            for norm in default_options["normed"]:
                for rec in default_options["reco"]:
                    for cos in default_options["cosz"]:
                        if ( (rec!=None and cosz!=True) ): #impossible cases go here
                            data_preps.append({"q": charge, "t": time, 
                                                                    "normed": norm, "reco": rec,"cosz": cosz })
    """
    for charge in default_options["q"]:
        data_preps.append({"q": charge, "t": None, "normed": True, "reco": "plane","cosz":True})
    for time in default_options["t"]:
        data_preps.append({"q": None, "t": time, "normed": True, "reco": "plane","cosz":True})
    for norm in default_options["normed"]:
        data_preps.append({"q": None, "t": None, "normed": norm, "reco": "plane","cosz":True})
    for rec in default_options["reco"]:
        data_preps.append({"q": None, "t": None, "normed": True, "reco": rec,"cosz":True})
    for cos in default_options["cosz"]:
        data_preps.append({"q": None, "t": None, "normed": True, "reco": "plane","cosz":cos})

    #data_preps = product()
    return data_preps


def compileModel(name, q=None, t=None, normed=False, reco=None, cosz=False):
    
    ##4 layers | t=none & q = none
    #if data_prep["t"] == None & data_prep["q"] == None:
    #    dimensions=4
    ##3 layers | t!=none & t!= false & q = none
    #elif data_prep["t"] == False:
    #    if data_prep["q"] == None:
    #        dimensions=2
    #    else:
    #        dimensions=1
    #else:
    #    dimensions=3
    ##3 layers | t=none & q != none
    ##2 layers | t=false & q = none
    ##1 layer | t=false & q != none

    
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

#outside training loop:
    #load data, name., data_prep combos, 
#every iteration:
    #specific data prep, model, save function.

def train(data_prep, x, y, numepochs=200):
    name=""
    for _,value in data_prep.items():
        name+=str(value)

    sys.stdout = open('trainedModels/%s.out' % name,'w')

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
    #num_epoch=len(history.history['loss'])
    #best_training_loss=np.min(history.history['loss'])
    #best_val_loss=np.min(history.history['val_loss'])
    #data_prep_index=name
    #new_row=[num_epoch, best_training_loss, best_val_loss, data_prep_index]

    #save info after every trained model here
    #with open('DataResults/results%.csv' %it, 'a') as f_object:
    #    csv_writer=writer(f_object)
    #    csv_writer.writerow(new_row)
    #    f_object.close()
    f = open("results.txt", "a")
    f.write("{}\tepochs:{}\tloss:{},{}\n".format(
        name,
        numepochs,
        np.min(history.history['loss']),
        np.min(history.history['val_loss'])
    ))
    f.close()
    
    sys.stdout.close()



#name file, make a method for it later or just input one for now

#train (public)
    #employ early stopping 

#save everything function (private)

    #def __save():

    #save: trained model, validation loss, training loss, numEpochs, 
#save csv, DO NOT SAVE DATA PREP DIRECTLY, reconstuct from index
#records data prep index, lowest loss, and number of epochs
#stored into a csv file at each iteration

#create row for saved information


#make sure to append to file
#save the dataprep index as its own column so the info isn't lost?

    
    
#save model

#def fileName():
    #call this at the start of training set to determine where to save the file
    #naming convention
    #key='nameHere'
    #it=1 #should this be universal?
    #while(os.path.isfile('DataResults/results%.csv' %it)):
           # it+1
    #key+=it
        
        

#reconstruct data_prep from index(private)
   # def __fetch_dataprep():
            #retrieves the index from the given ...
#######################################################################

#analysis methods

#get_lowest_loss 
        ##def get_lowest_loss(r=None):
            #retrieves the information from a file
            #converts information to a dataframe
            #retrieves the lowest or sorts then returns a range of the lowest loss
            #returns another dataframe
            #if r=None:
            #    return min(history.history(['loss']))
                #get the lowest l
            #elif r>0:
            #    return np.partition(history.history['loss'],r)
                #get a list of length r of the lowest loss values
            #else:
              #  return 'please enter a valid number'
                #ask the use to enter a valid number 
                
#get_num_epochs

#get_lowest_loss (with respect to number of epochs) (i.e loss vs epoch)

#plot specific 


