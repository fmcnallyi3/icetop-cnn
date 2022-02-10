from data_tools import load_preprocessed, dataPrep
import numpy as np
from tensorflow import keras
from keras import layers, models, callbacks
import pickle

class Trainer:
    
    def __init__(self, name = None, model = None , data_prep = None):
        self.model = None 
        self.data_preps = [] #list of all possible data preps based on user choice
        #self.data_prep=None
        self.name=None

        #self.test_specific = False #is true when the user wants to test a particular data_prep
    
        if model!=None:
            self.model=model
        if data_prep!=None:
            self.data_preps=[data_prep]
        if name!=None:
            self.name=name
        elif(data_prep != None):
            self.name=""
            for _,value in data_prep.items():
                if not value:
                    self.name+=value
                elif value is None:
                    self.name+='None'
                else:
                    self.name+='False'
            
        
        #load the simulation files
        simPrefix = '/Users/kmays/simFiles'

        #load the data into their arrays
        self.x, self.y = load_preprocessed(simPrefix, 'train')
        self.x_i = None
        self.temp_y = self.y['energy']
        self.trainCut = (np.random.uniform(size=len(self.temp_y)) < 0.85)
        self.testCut = np.logical_not(self.trainCut)
        #changes later during training iteration


    def main():
        Trainer()


    def generate_data_prep(self, q = None, t = None,  normed = None, reco = None, cosz = None):
        #parameters specific combos
        default_options = { 'q' :  ["mean","sum","product","min","max", None], 
                            't' : ["mean","sum","product","min","max", None, False], #this may be wrong, check in datatool
                            'normed' : [True, False],
                            'reco' : [ None, "plane", "laputop", "small"],
                            'cosz' : [False, True]}
    
        if q!=None:
            default_options["q"]=q
        if t!=None:
            default_options["t"]=t
        if normed!=None:
            default_options["normed"]=normed
        if reco!=None:
            default_options["reco"]=reco
        if cosz!=None:
            default_options["cosz"]=cosz


      
        for charge in default_options["q"]:
            for time in default_options["t"]:
                for norm in default_options["normed"]:
                    for rec in default_options["reco"]:
                        for cos in default_options["cosz"]:
                            if (rec!='None' and cosz!=True): #impossible cases go here
                                self.data_preps.append({"q": charge, "t": time, 
                                                                      "normed": norm, "reco": rec,"cosz": cos })
    

    def compileModel(self, data_prep):
        #4 layers | t=none & q = none
        if data_prep["t"] == None & data_prep["q"] == None:
            dimensions=4
        #3 layers | t!=none & t!= false & q = none
        elif data_prep["t"] == False:
            if data_prep["q"] == None:
                dimensions=2
            else:
                dimensions=1
        else:
            dimensions=3
        #3 layers | t=none & q != none
        #2 layers | t=false & q = none
        #1 layer | t=false & q != none

        #actual model layers
        charge_input = keras.Input(shape=(10,10,dimensions,),name="charge")
        charge_input=keras.Input(shape=(10,10,2,))

        conv1_layer = layers.Conv2D(64,kernel_size=3,padding='same',activation='relu')(charge_input)
        conv2_layer = layers.Conv2D(32,kernel_size=3,padding='same',activation='relu')(conv1_layer)
        conv3_layer = layers.Conv2D(16, kernel_size=3, padding='same',activation="relu")(conv2_layer)
        flat_layer = layers.Flatten()(conv3_layer)
        zenith_input=keras.Input(shape=(1,))
        concat_layer = layers.Concatenate()([flat_layer,zenith_input])
        dense1_layer = layers.Dense(256,activation='relu')(concat_layer)
        dense2_layer = layers.Dense(256,activation='relu')(dense1_layer)
        dense3_layer = layers.Dense(256,activation="relu")(dense2_layer)
        output = layers.Dense(1)(dense3_layer)

        self.model = models.Model(inputs=[charge_input,zenith_input],outputs=output,name=self.name)
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae','mse'])

    #outside training loop:
        #load data, name., data_prep combos, 
    #every iteration:
        #specific data prep, model, save function.
    
    def train(self,numepochs=100):
        #if self.data_prep !=None:
        #    self.x_i = dataPrep(self.x, self.y, **self.data_prep)
        #    self.compileModel(self.data_prep)
        #    
        #    csv_logger = callbacks.CSVLogger('trainedModels/{}'.format(self.name))
        #    early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True) # default -> val_loss
        #    checkpoint = callbacks.ModelCheckpoint('trainedModels/%s.h5' % self.name,save_best_only=True)
        #    self.callbacklist = [early_stop, csv_logger,checkpoint]
        #    history = self.model.fit(x=self.x_i, y=self.temp_y, epochs=numepochs,validation_split=0.15,callbacks=self.callbacklist)
        #    with open('trainedModels/%s.pickle', 'wb') as f:
        #        pickle.dump(history.history, f)

        #else:
            for data_prep in self.data_preps:
                self.x_i = dataPrep(self.x, self.y, **data_prep)
                self.compileModel(data_prep)
                
                self.name=""
                for _,value in data_prep.items():
                    if not value:
                        self.name+=value
                    elif value is None:
                        self.name+='None'
                    else:
                        self.name+='False'

                csv_logger = callbacks.CSVLogger('trainedModels/{}'.format(self.name))
                early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True) # default -> val_loss
                checkpoint = callbacks.ModelCheckpoint('trainedModels/%s.h5' % self.name,save_best_only=True)
                self.callbacklist = [early_stop, csv_logger,checkpoint]
                history = self.model.fit(x=self.x_i, y=self.temp_y, epochs=numepochs,validation_split=0.15,callbacks=self.callbacklist)
                with open('trainedModels/%s.pickle', 'wb') as f:
                    pickle.dump(history.history, f)



    #name file, make a method for it later or just input one for now
    
    #train (public)
        #employ early stopping 
    
    #save everything function (private)

            """ def __save():
            #save csv, DO NOT SAVE DATA PREP DIRECTLY, reconstuct from index
            #records data prep index, lowest loss, and number of epochs
            #stored into a csv file at each iteration
            
            #create row for saved information
            best_loss=history.history(['loss'][-1])
            num_epoch=len(history.history(['loss']))
            #data_prep_index=data_preps...
            new_row=[best_loss, num_epoch]
            
            #make sure to append to file
            #save the dataprep index as its own column so the info isn't lost?
            with open('DataResults/results%.csv' %it, 'a') as f_object:
                csv_writer=writer(f_object)
                csv_writer.writerow(new_row)
                f_object.close()
            
            
        #save model
        
        def fileName():
            #call this at the start of training set to determine where to save the file
            #naming convention
            key='nameHere'
            it=1 #should this be universal?
            while(os.path.isfile('DataResults/results%.csv' %it)):
                  it+1
            key+=it
            
            
    
    #reconstruct data_prep from index(private)
        def __fetch_dataprep():
              #retrieves the index from the given ...
    #######################################################################
    
    #analysis methods
    
    #get_lowest_loss 
         def get_lowest_loss(r=None):
              #retrieves the information from a file
              #converts information to a dataframe
              #retrieves the lowest or sorts then returns a range of the lowest loss
              #returns another dataframe
              if r=None:
                  return min(history.history(['loss']))
                  #get the lowest l
              elif r>0:
                  return np.partition(history.history['loss'],r)
                  #get a list of length r of the lowest loss values
              else:
                  return 'please enter a valid number'
                  #ask the use to enter a valid number 
                  
    #get_num_epochs
    
    #get_lowest_loss (with respect to number of epochs) (i.e loss vs epoch)
    
    #plot specific 
    
     """