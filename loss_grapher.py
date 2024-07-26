#!/usr/bin/env python3

# LOSS VISULARIZER
# ASSUMES PROPERLY FORMATTED CSV FILES (GIGO)
# USE IN THE FOLLOWING FORMAT: 'loss_grapher.py [modelname1, modelname2, ...]'

import os
from sys import argv

import matplotlib.pyplot as plt
import numpy as np

ICETOP_CNN_DATA_DIR = os.getenv('ICETOP_CNN_DATA_DIR')
MODELS_FOLDER_PATH = os.path.join(ICETOP_CNN_DATA_DIR, 'models')

# Ensure that 'models' folder is found
assert os.path.exists(MODELS_FOLDER_PATH), f'Could not find models folder specified: {MODELS_FOLDER_PATH}'

# Iterate over all models provided as command line arguments
for model_name in argv[1:]:

    # Build model name into file path
    model_path = os.path.join(MODELS_FOLDER_PATH, model_name, model_name + '.csv')

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

    # Terrible. Utterly terrible
    unique_labels = ['loss'] + sorted([label for label in header if 'val' not in label and label != 'epoch' and label != 'loss'])
    
    # Create plot and subplots
    fig, axs = plt.subplots(len(unique_labels))
    # Name the entire plot the same as the name of the model to easily keep track
    fig.suptitle(model_name)

    # Loop over each loss function while keeping track of the subplot index
    # Need to check if non metrics were passed in -> no subplots generated 
    for axs_idx, loss_function in enumerate(unique_labels):
        (axs[axs_idx] if type(axs) is np.ndarray else axs).plot(epochs, model_data[:, header.index(loss_function)], label=loss_function)                      # Plot training loss
        (axs[axs_idx] if type(axs) is np.ndarray else axs).plot(epochs, model_data[:, header.index(f'val_{loss_function}')], label=f'val_{loss_function}')    # Plot validation loss
        (axs[axs_idx] if type(axs) is np.ndarray else axs).legend(loc='upper right', fontsize='x-small')                                                      # Plot legend
        (axs[axs_idx] if type(axs) is np.ndarray else axs).set_yscale('log')                                                                                  # Set logarithmic scale in y-axis
    
    # Save image as a png
    plt.savefig(os.path.join(MODELS_FOLDER_PATH, model_name, f'{model_name}.png'), format='png')
