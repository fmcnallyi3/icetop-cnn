#wrapper script that isolates a hyperparameter & trains iterations of a model w/ that variation
#import baseline
import random
#import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    isolated_parameter="isolatedCutoff" #num epochs & lr patience
    data_points=15 #number of points for plot
    #create data points to train on and plot
    isolated_val_range=[random.randrange(10,250) for i in range(data_points)]
    x = [] #val_loss
    y = [] #cutoff
    """for point in isolated_val_range:
        i = 0
        while(os.path.exists('models/%s/%s.csv' % (isolated_parameter, isolated_parameter+str(i)))): i += 1
        name = isolated_parameter+str(i)
        dir = 'models/%s' % isolated_parameter
        #run the model, train it
        baseline.main(point, name, dir) """
        #recall & plot the data
    for num in range(7):
        d = pd.read_csv('%s/%s.csv' % ('models', 'ESRotations'))
        x.append(d['val_loss'][-1])
        y.append(d['epoch'].count())
        
    #plot x and y values in scatter plot
    plt.scatter(x,y)
    plt.xlabel("Validation Loss")
    plt.ylabel("Cutoff (Max Epochs & Patience)")
    plt.savefig(f'assessment/isolated/isolatedParameter.png', format='png')