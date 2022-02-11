from data_tools import load_preprocessed, dataPrep
import numpy as np
from tensorflow import keras
from keras import layers, models, callbacks
import pickle

from multiprocessing import Process
import Trainer

trainer = Trainer()
trainer.generate_data_prep()
if not trainer.data_preps:
    print("Could not generate data preps.")
    quit()
for prep in trainer.data_preps:
    print(str(prep))
processes = []
if __name__ == '__main__':
    i = 0
    for prep in trainer.data_preps:
        proc = Process(target=trainer.train(i))
        proc.start()
        i += 1
    for proc in processes:
        proc.join()