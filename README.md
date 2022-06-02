# IceTop Energy Reconstruction using a CNN

The machine learning team led by Dr. McNally conducted this research to create a Convoutional Neural Network (CNN) model that reconstructs the energy of an air shower within ... error.

### Common Errors
- Edit the simPrefix variable to the location of the simFiles on your device, failure to do so may result in division by zero error after calling load_preprocessed

### Overview
- The main components include the data tools python script, the model training jupyter notebook, & the assessment jupyter jobtebook
- A folder labeled trained models will be present. It will contain model variations and a text file to summarise the results of each model variation
- Not included in this repository is a copy of simFiles, which is a folder of files needed to train a model

### Data Tools
- This python script is used to load & manipulate the data
- Calls to load_preprocessed, dataPrep, and filterReco are made within the trainging notebook
- It also contains a variety of methods to allow functionality for quality cuts, naming convention, converting a dictionary to a multi-dimensional array, ect

### Model Training Jupyter Notebook 
Note: the lines below explain the training notebook in chronological order, lines that begin with "*" denote nonessential code & can be commented out without affecting performance
- After importing the necessary packages, there's variables listed that must be personalized by the user
- *The following cell contains a 'while' loop to allow for a naming convention to prevent trained models from being overwritten
- The baseline model is built using the functional API to accomodate multiple inputs (zenith is the second input)
- the notebook calls load_preprocessed from data tools to load x(events) into a numpy array where each event is a  10x10 grid with 4 dimensions and load y(event specs) into a dictionary of air shower specifications where each item in the list of values correlates to an event. It also cuts NaNs from x and removes the associated spec data
-  dataPrep is called to produce x_i that will be one of two input layers of the model used to reconstruct the air shower specification(energy) and filterReco is called to filter NaNs from spec data and its associated event
- *callbacks are specified to determine actions that will be carried out between epochs
- A call to fit initiates the training process. Charge and Zenith are specified as information for the model to use for learning while the list of energy values in y is specified as the target data that the model will learn to reconstruct
- *There's 3 saving conventions: save the model layers in a .h5 file, the prep dictionary in a .npy file, and the best performing epoch in a collective results.csv file

### Assessment Juypter Notebook
(explain each graph here)