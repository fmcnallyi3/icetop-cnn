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
    
        def __save():
            #save csv, DO NOT SAVE DATA PREP DIRECTLY, reconstuct from index
            #save model location???
            #save epoch
            #save iteration index
            
            #create row for saved information
            best_loss=history.history(['loss'][-1])
            num_epoch=len(history.history(['loss']))
            new_row=[best_loss, num_epoch]
            
            #save said information
            with open('DataResults/results%.csv' %it, 'a') as f_object:
                csv_writer=writer(f_object)
                csv_writer.writerow(new_row)
                f_object.close()
            
            
        #save model
        
        def fileName():
            #call this at the start of training set to determine where to save the file
            #naming convention
            it=1 #should this be universal?
            while(os.path.isfile('DataResults/results%.csv' %it):
                  it+=1
            
            
    
    #reconstruct data_prep from index(private)
        def __fetch_dataprep():
              #retrieves the index
    #######################################################################
    
    #analysis methods
    
    #get_lowest_loss 
         def get_lowest_loss(r=None):
              #retrieves the lowest or a range of the lowest loss
              if r=None:
                  return min(history.history(['loss']))
                  #get the lowest loss value in the list
              elif r>0:
                  return np.partition(history.history['loss'],r)
                  #get a list of length r of the lowest loss values
              else:
                  return 
                  #ask the use to enter a valid number 
                  
    #get_num_epochs
    
    #get_lowest_loss (with respect to number of epochs) (i.e loss vs epoch)
    
    #plot specific 
    
    