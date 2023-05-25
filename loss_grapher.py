# LOSS VISULARIZER
# ASSUMES PROPERLY FORMATTED CSV FILES (GIGO)
# USE IN THE FOLLOWING FORMAT: 'loss_grapher.py [modelname1, modelname2, ...]'

import os
from sys import argv

import matplotlib.pyplot as plt
import numpy as np

'''
    THE USER SHOULD FEEL FREE TO EDIT THE THREE VALUES BELOW AS THEY PLEASE
'''
MODELS_FOLDER_PATH = os.path.join(os.getcwd(), 'models')              # Folder that contains model .csv files
LOSS_GRAPHS_FOLDER_PATH = os.path.join(os.getcwd(), 'loss_graphs')    # Folder to hold loss graphs
ADDITIONAL_LOSS_METRICS = ['mae', 'mse']                              # Additional loss metrics to graph - will usually be ['mae', 'mse']
'''
    THE USER SHOULD FEEL FREE TO EDIT THE THREE VALUES ABOVE AS THEY PLEASE
'''


# Ensure that 'models' folder is found
assert os.path.exists(MODELS_FOLDER_PATH), f'Could not find model folder specified: {MODELS_FOLDER_PATH}'

# Create loss_graphs directory if it does not already exist
if not os.path.exists(LOSS_GRAPHS_FOLDER_PATH):
    os.mkdir(LOSS_GRAPHS_FOLDER_PATH)

# Guarantee to at least graph the loss function used during training if the user specifies no additional metrics
loss_labels = ['loss'] + ADDITIONAL_LOSS_METRICS

# Iterate over all models provided as command line arguments
for model_name in argv[1:]:

    # Build model name into file path
    model_path = os.path.join(MODELS_FOLDER_PATH, model_name + '.csv')

    # Ensure that the model actually exists (no typos)
    if not os.path.exists(model_path):
        print(f'WARNING: {model_name} not found at {MODELS_FOLDER_PATH}')
        continue

    # Extract header information
    with open(model_path, 'r') as model_file:
        header = model_file.readline().strip().split(',')

    # Extract all of the actual model data
    model_data = np.genfromtxt(model_path, delimiter=',', skip_header=1)
    # Pull out the epochs specifically
    epochs = model_data[:, header.index('epoch')]

    # Create plot and subplots
    fig, axs = plt.subplots(len(loss_labels))
    # Name the entire plot the same as the name of the model to easily keep track
    fig.suptitle(model_name)

    # Loop over each loss function while keeping track of the subplot index
    for axs_idx, loss_function in enumerate(loss_labels):
        axs[axs_idx].plot(epochs, model_data[:, header.index(loss_function)], label=loss_function)                      # Plot training loss
        axs[axs_idx].plot(epochs, model_data[:, header.index('val_' + loss_function)], label='val_' + loss_function)    # Plot validation loss
        axs[axs_idx].legend(loc='upper right', fontsize='x-small')                                                      # Plot legend
        axs[axs_idx].set_yscale('log')                                                                                  # Set logarithmic scale in y-axis
    
    # Save image as a png
    plt.savefig(os.path.join(LOSS_GRAPHS_FOLDER_PATH, model_name + '.png'), format='png')