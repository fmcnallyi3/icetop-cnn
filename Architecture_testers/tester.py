from data_tools import load_preprocessed, dataPrep
import numpy as np
from tensorflow import keras
from keras import layers, models, callbacks
import pickle
import os
import multiprocessing
from time import sleep
from Trainer import generate_data_prep,train

data_preps = generate_data_prep()
if not data_preps:
    print("Could not generate data preps.")
    quit()

simPrefix = '/home/richardson_p/icetop-cnn'+'/simdata'
x, y = load_preprocessed(simPrefix, 'train')
energy = y["energy"]

l = len(data_preps)

#for prep in data_preps:
for i in range(0,len(data_preps),30): #train 30 models at a time
    processes = []
    for j in range(i,i+30):
        #print("Starting process for %s" % str(prep))
        proc = multiprocessing.Process(target=train,args=(data_preps[i],x,y,))
        proc.start()
        print("Started process %i: %s" % (proc.pid,str(data_preps[i])) )
        processes.append(proc)
        sleep(1)
#       i += 1
    
    for proc in processes:
        proc.join()
