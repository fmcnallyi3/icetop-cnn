class Trainer:
    #import tensorflow as tf
    #from tensorflow import keras as ks
    
    
    
    
    
    #constructor(give default model)
    def __init__(self, model = None , data_prep = None):
        
        self.model = None #only used when in specific
        self.data_preps = {} #list of all possible data preps based on user choice
        self.eval_info = None # list[
        self.data_prep = None #if the user is in specific mode
        self.specific = False #is true when the user wants to test a particular data_prep

        
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
    
        default_options = { 'q' :  ['sum','product','min','max', None], 
                            't' : ['mean','sum','product','min','max'], #this may be wrong, check in datatool
                            'normed' : [True, False],
                            'reco' : [ None, 'plane', 'laputop', 'small'],
                            'cosz' : [False, True]}
                       
    
        if q == None and normed == None and  reco == None and cosz == None: #if you want all
            pass

        elif q == False and normed == False and  reco == False and cosz == False: #if you want to exclude some entirely

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
        
        
        if (model == None):
            
            if data_prep['q'] == None:
                size = 2
            else:
                size = 1
                
            


            ip1 = Input(shape=(10,10,size))
            l = Conv2D(64,kernel_size=3,padding='same',use_bias=False)(ip1)
            l = BatchNormalization()(l)
            l = LeakyReLU(alpha=.3)(l)
            l = Conv2D(32,kernel_size=3,padding='same',use_bias=False)(l)
            l = BatchNormalization()(l)
            l = LeakyReLU(alpha=.3)(l)
            l = Dropout(.6)(l)
            l = Flatten()(l)

            if data_prep['reco'] == None:
                ips = ip1
            else:
                ip2 = Input(shape= (1,))
                merge = Concatenate()([l,ip2])
                ips= [ip1,ip2]

            l = Dense(1,use_bias=False)(l)
            l = BatchNormalization()(l)
            l = LeakyReLU(alpha=.3)(l)
            l = Dropout(.5)(l)
            output = Dense(1)

            model = Model(inputs = ips, outputs = output,name=nameModel(prep,prefix='test'))
            return model, data
            #model.summary()

           
        else:
            model = model
            
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
                model.fit(
                
    
    #save everything function (private)
    
        #to csv
            #save csv, DO NOT SAVE DATA PREP DIRECTLY, reconstuct from index
            #save model location
            #save epock
            #save iteration index
        #save model
    
    #reconstruct data_prep from index(private)
    #######################################################################
    
    #analysis methods
    
    #get_lowest_loss 
    
    #get_lowest_loss (with respect to number of epochs) (i.e loss vs epoch)
    
    #plot all models with specific data_prep value against each other
    
    