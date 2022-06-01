# IceTop Energy Reconstruction using a CNN

The machine learning team led by Dr. McNally conducted this research to create a Convoutional Neural Network (CNN) model that reconstructs the energy of an air shower within ... error.

### Common Errors
- edit the simFile path to the location of the simFiles on your device, failure to do so may result in division by zero error after calling load_preprocessed

### Overview
- the main components include the data tools python script, the model training jupyter notebook, & the assessment jupyter jobtebook
- a folder labeled trained models will be present. It will contain model variations and a text file to summarise the results of each model variation

### Data Tools
- this python script is used to load & manipulate the data
- load_preprocessed...
- data prep...
- it also contains a variety of methods to allow functionality for quality cuts, ...

#### Model Training Jupyter Notebook 
Note: lines that begin with '*' denote nonessential code & can be commented out without affecting performance
- after importing the necessary ..., the notebook calls preprocessed from data tools to load x(events) into a numpy array where each event is a  10x10 grid with 4 dimensions and load y(event specs) into a dictionary of air shower specifications where each item in the list of values correlates to an event
- calls are made to the dictionary keys of y(simulation info) to split them into their respective lists
- *there may be a 'while' loop present to allow for a naming convention to prevent trained models from overwriting
- specify dictionary values for prep & the number of epochs
- a call to dataprep will produce an array of x_i that will be the input layer of the model and the information the model uses to reconstruct spec informtion
- the temp_y is specified as the air shower spec that the model is aiming to learn about, which is energy
- ...

### Assessment Juypter Notebook
(explain each graph here)