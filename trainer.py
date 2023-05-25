import os

import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model

import config as cg
from data_utils import load_preprocessed, data_prep, get_training_assessment_cut, add_reco
from model import get_compiled_model

# Limit scope of trainer to a single GPU - can easily adapt to support GPU parallelization
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=cg.GPU_ID

def main():
    '''Creates/restores and trains a model based on the settings located in "config"'''

    # Get datasets
    training_dataset, validation_dataset, input_shapes = get_datasets()

    # Get model
    model = make_or_restore_model(input_shapes)

    # Train the model
    train_model(model, training_dataset, validation_dataset)


def get_datasets():
    '''Loads simulation data from files for training'''

    # Load detector inputs and event parameters
    detector_inputs, event_parameters = load_preprocessed(cg.SIMDATA_FOLDER_PATH, composition=cg.NUCLEI)

    # Prepare simulation data
    model_inputs = data_prep(detector_inputs, **cg.PREP)

    # Get training cut
    cut_events = get_training_assessment_cut(event_parameters, 'training', cg.PREP['sta5'])

    # Add reconstruction data if it is used
    model_inputs, cut_events = add_reco(model_inputs, cut_events, event_parameters, cg.PREP['reco'], cg.PREP['normed'])

    # Apply cuts to data
    model_inputs = {input_name: model_input[cut_events] for input_name, model_input in model_inputs.items()}
    for key in event_parameters.keys():
        event_parameters[key] = event_parameters[key][cut_events]

    # Get number of events from an arbitrary key
    num_events = len(event_parameters['energy'])

    # Create training / validation split
    training_cut = np.random.rand(num_events) > cg.VALIDATION_SPLIT
    validation_cut = np.logical_not(training_cut)

    # Assign training data
    training_model_inputs = [model_input[training_cut] for model_input in model_inputs.values()]
    training_energy = event_parameters['energy'][training_cut]

    # Assign validation data
    validation_model_inputs = [model_input[validation_cut] for model_input in model_inputs.values()]
    validation_energy = event_parameters['energy'][validation_cut]

    # This format for organizing the training/validation data allows for unlimited inputs/outputs
    # Currently has a single output - energy
    # To adjust for multiple outputs, convert to a tuple
    # Also returns the shapes of all inputs so that we can use the Keras functional API
    return (
        tf.data.Dataset.from_tensor_slices(( tuple(training_model_inputs), training_energy )).shuffle(np.sum(training_cut)).batch(cg.BATCH_SIZE),
        tf.data.Dataset.from_tensor_slices(( tuple(validation_model_inputs), validation_energy )).batch(cg.BATCH_SIZE),
        {input_name: model_input.shape[1:] for input_name, model_input in model_inputs.items()}
    )
    

def make_or_restore_model(input_shapes):
    '''Creates a model if there is none saved with the same name on disk. Otherwise loads the saved model for further training.'''

    # Check model file path for a model with the same name as the current model to be trained
    if os.path.exists(os.path.join(cg.MODELS_FOLDER_PATH, cg.MODEL_NAME + '.h5')):
        # Model found - load and return the saved model
        return load_model(os.path.join(cg.MODELS_FOLDER_PATH, cg.MODEL_NAME + '.h5'))
    # Model not found - must create and compile a new model
    return get_compiled_model(input_shapes)


def train_model(model, training_dataset, validation_dataset):
    '''Trains the model using the provided training/validation datasets and settings from "config"'''

    # Create models folder if it does not already exist
    if not os.path.exists(cg.MODELS_FOLDER_PATH):
        os.mkdir(cg.MODELS_FOLDER_PATH)

    # Save model parameters
    np.save(os.path.join(cg.MODELS_FOLDER_PATH, cg.MODEL_NAME + '.npy'), cg.PREP)

    # Let 'er rip
    model.fit(
        training_dataset,
        epochs=cg.NUM_EPOCHS,
        validation_data=validation_dataset,
        callbacks=[
            CSVLogger(os.path.join(cg.MODELS_FOLDER_PATH, cg.MODEL_NAME + '.csv'), append=True),
            ModelCheckpoint(os.path.join(cg.MODELS_FOLDER_PATH, cg.MODEL_NAME + '.h5')),
            EarlyStopping(
                min_delta=cg.MIN_ES_DELTA,
                patience=cg.ES_PATIENCE,
                mode='min',
                restore_best_weights=cg.RESTORE_BEST_WEIGHTS
            ),
            ReduceLROnPlateau(
                factor=cg.LR_FACTOR,
                patience=cg.LR_PATIENCE,
                mode='min',
                min_delta=cg.MIN_LR_DELTA,
                cooldown=cg.LR_COOLDOWN,
                min_lr=cg.MIN_LR
            ),
        ]
    )


if __name__ == '__main__':
    main()
