from data_tools import load_preprocessed, dataPrep
import numpy as np
from tensorflow import keras
from keras import layers, models, callbacks
import pickle
import os
import multiprocessing
from time import sleep
from Trainer import generate_data_prep,train

#put None in quotation marks for it to use None a
#update specs under folder location in train method!
#call nohup while running script on supercomputer so it keeps running while you're gone
data_preps = generate_data_prep(reco="plane")
if not data_preps:
    print("Could not generate data preps.")
    quit()

for prep in data_preps:
    print("Planning to train a model for %s" % str(prep))

simPrefix = '/home/mays_k/icetop-cnn/simdata'
x, y = load_preprocessed(simPrefix, 'train')
energy = y["energy"]

l = len(data_preps)
num = 1 #train num models at a time
for i in range(0,len(data_preps),num):
    processes = []
    for j in range(i,i+num):
        if j < len(data_preps):#checks for overflow
            #print("Starting process for %s" % str(prep))
            proc = multiprocessing.Process(target=train,args=(data_preps[j],x,y,))
            proc.start()
            print("Started process %i: %s" % (proc.pid,str(data_preps[j])) )
            processes.append(proc)
            sleep(1)
    
    for proc in processes:
        proc.join()