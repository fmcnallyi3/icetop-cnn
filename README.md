# IceTop Cosmic-Ray Energy Reconstruction using a Convolutional Neural Network

### Overview
- 'data_utils.py' is a utility file that is used by both the training script and assessment notebook.
- 'trainer.py' is the training script and is designed not be edited. It is to work with the files 'config.py' and 'model.py'.
- 'config.py' contains all of the different options for training models. This does not include options related to the model architecture.
- 'model.py' contains the model architecture using the Keras Functional API. This file is meant to be edited and shared frequently.
- 'loss_grapher.py' is a script that will graph the loss curves for models from the data in their .csv files. See opening comments in file for usage.
- 'energy_assessment.ipynb' is an interactive notebook designed to assess any combination of models all at once. Other than the initial constants, make changes at your own risk.
- 'simdata.txt' gives detailed instructions for how to download the data provided by IceCube.

### Notes
- This project requires NumPy, Keras/TensorFlow, and Matplotlib.
- The training script will automatically create the directory needed to store the model files.
- The loss grapher script will automatically create the directory needed to store the image files.
- Many different configuration options are input validated, but some still are not. Expect some GIGO behavior.
- The datasets used are extremely large. (TODO: Implement lazy evaluation.) Expect to hit OOM errors without a powerful machine.

If you are unable to reach me by any other platform, feel free to reach out to egdorr@gmail.com if there are any questions or concerns.
