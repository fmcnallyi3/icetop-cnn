class Trainer:
    
    
    self.model = None 
    self.data_preps = None #list of all possible data preps based on user choice
    self.eval_info = None # list[
    
    
    
    self.test_particular = False #is true when the user wants to test a particular data_prep
    
    
    
    
    
    
    
    
    
    
    #constructor(give default model)
    def __init__(self, model = None , data_prep = None, ):
        
    ##########################################################################
    #actual training methods
    
    #define_model(tuned_model) (public)
    
    #define_data_prep(data_prep) #if the user wants to run on only one (public)
    
    
    #train (public)
        #employ early stopping 
    
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
    
    #plot specific 
    
    