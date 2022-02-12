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
for i in range(0,len(data_preps),5): #train 15 models at a time
    processes = []
    for j in range(i,i+5):
        #print("Starting process for %s" % str(prep))
        proc = multiprocessing.Process(target=train,args=(data_preps[j],x,y,))
        proc.start()
        print("Started process %i: %s" % (proc.pid,str(data_preps[j])) )
        processes.append(proc)
        sleep(1)
#       i += 1
    
    for proc in processes:
        proc.join()
