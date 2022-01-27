#import tensorflow as tf
#won't properly import packages???????
import keras
import numpy as np
from keras import layers, models
class Backup:
    
    
    
    
    
    
    #constructor(give default model)
    def __init__(self, model = None , data_prep = None):
        #either use enters a specific model- data_prep combo or the program produces a
        self.model = None #only used when in specific
        self.data_preps = {} #list of all possible data preps based on user choice
        self.eval_info = None # list[
        self.data_prep = None #if the user is in specific mode
        self.specific = False #true when the user wants to test a a single model-data_prep combo

        
        if model != None and data_prep != None: # if user wants to test only one model
            
            self.model == model 
            self.data_prep == data_prep
            self.specific = True
        elif model != None or data_prep != None: #if user gives only model or data_prep
            raise Exception("To train a specific model, you must pass model = your_model and data_prep = your_data_prep")
        else: #default
            self.model = None #FIX WITH DEFAULT BUILDER
            self.generate_data_prep() 
            self.specific = False
        

    
        
            
            
        
    ##########################################################################
    #actual training methods
    
    #generate all data preps
    def generate_data_prep(self, q = None, t = None,  normed = None, reco = None, cosz = None):
    
        default_options = { 'q' :  ['mean','sum','product','min','max', None], 
                            't' : ['mean','sum','product','min','max', None], #this may be wrong, check in datatool
                            'normed' : [True, False],
                            'reco' : [ None, 'plane', 'laputop', 'small'],
                            'cosz' : [False, True]}
                       
    
        if q == None and normed == None and  reco == None and cosz == None: #generates all combos
            pass

        elif q == False and normed == False and  reco == False and cosz == False: #if you want to exclude some entirely
            #this should be or?
            if not q:
                default_options['q'] = None
            if not t:
                default_options['t'] = None
            if not normed:
                default_options['normed'] = None
            if not reco or not cosz: #need both or neither to function 
                default_options['reco'] = None
                default_options['cosz'] = None
        else: # if you want specific values set

            if q != None:
                default_options['q'] = q
            if t != None:
                default_options['t'] = t
            if normed != None:
                default_options['normed'] = normed
            if reco != None:
                default_options['reco'] = reco
            if cosz != None: #need both or neither to function 
                default_options['cosz'] = cosz



        i = 0
        
        for charge in default_options['q']:
            for time in  default_options['t']:
                for norm in  default_options['normed']:
                    for rec in  default_options['reco']:
                        for cosz in  default_options['cosz']:
                            if (rec == None and not cosz):
                                self.data_preps['Model_' + str(i)] = {'q': charge, 't': time, 
                                                                      'normed': norm, 'reco': rec,'cosz' : cosz }
                            
                            elif (rec == None and  cosz):
                                continue
                            else:
                                self.data_preps['Model_' + str(i)] = {'q': charge, 't': time, 
                                                                      'normed': norm, 'reco': rec,'cosz' : cosz }
                            i = i + 1
    
    
    #define_model(tuned_model) (public)
    def define_model(self, data_prep, model = None): #MODIFY MODEL HERE
        
        
        if (model == None):#this is the default model
            
            if data_prep['q'] == None:
                size = 2
            else:
                size = 1
                
            


            ip1 = keras.Input(shape=(10,10,size))
            l = layers.Conv2D(64,kernel_size=3,padding='same',use_bias=False)(ip1)
            l = layers.BatchNormalization()(l)
            l = layers.LeakyReLU(alpha=.3)(l)
            l = layers.Conv2D(32,kernel_size=3,padding='same',use_bias=False)(l)
            l = layers.BatchNormalization()(l)
            l = layers.LeakyReLU(alpha=.3)(l)
            l = layers.Dropout(.6)(l)
            l = layers.Flatten()(l)

            if data_prep['reco'] == None:
                ips = ip1
            else:
                ip2 = keras.Input(shape= (1,))
                merge = layers.Concatenate()([l,ip2])
                ips= [ip1,ip2]

            l = layers.Dense(1,use_bias=False)(l)
            l = layers.BatchNormalization()(l)
            l = layers.LeakyReLU(alpha=.3)(l)
            l = layers.Dropout(.5)(l)
            output = layers.Dense(1)

            model = models.Model(inputs = ips, outputs = output,name=nameModel(prep,prefix='test'))
            model.summary()
            return model

           
        else:
            model = model
            model.summary()
            
        model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mse'])
    
    #define_data_prep(data_prep) #if the user wants to run on only one (public)
    def set_data_prep(self, data_prep):
        
        if not self.specific:
            raise Exception ('You must be in specific mode to use this method')
        
        self.data_prep = data_prep
    
    
    #train (public)
        #employ early stopping 
        #different behavior depending on if in specific mode or not
    def train(self):
        
        if self.specific:
            #train specific model
            pass
        else:
            for data_prep in self.data_preps:
                model, data = self.define_model(data_prep)
                model.fit()#fix this
                
    
    #save everything function (private)
    
    def __save():
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
        
    def nameModel(self,prep=None):
        #call this at the start of training set to determine where to save the file
        #naming convention
        if:
            prep==None
            
        it=1 #should this be universal?
        while(os.path.isfile('DataResults/results%.csv' %it)):
              it+1
        key+=it
            
            
    
    #reconstruct data_prep from index(private)
    #def __fetch_dataprep():
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
    
    