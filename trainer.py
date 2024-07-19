#!/usr/bin/env python3

import argparse
import json
import os

# Check that the environment has been activated
# Should only check when not run on the condor cluster
# Should check before importing external libraries
if not os.getenv('_CONDOR_SLOT'):
    venv_path = os.path.join(os.getenv('ICETOP_CNN_DIR', ''), '.venv')
    assert os.getenv('VIRTUAL_ENV') == venv_path, 'ERROR: Virtual environment not activated. Activate with the command "icetop-cnn"'

# Supress debugging information
# Can remove safely, just makes for cleaner output
# Should be set before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

import config as cg
from utils import data_prep, get_preprocessed, get_training_assessment_cut
from model import get_compiled_model

ICETOP_CNN_DATA_DIR = os.getenv('ICETOP_CNN_DATA_DIR')    

def main():
    '''Creates/restores and trains a model based on the settings located in "config"'''

    # Get datasets
    if args.test: print('Getting datasets...')
    (training_dataset, validation_dataset), input_shapes = get_training_datasets()

    # Get model
    if args.test: print('Fetching model...')
    model = create_or_restore_model(input_shapes)

    # Train the model
    if args.test: print('Training model...')
    model = train_model(model, training_dataset, validation_dataset)

    # Assess the model
    if args.test: print('Assessing model...')
    assess_model(model)
    

def get_training_datasets():
    '''Prepares tensorflow datasets used for training'''

    # Load data used for training/testing in its entirety
    model_inputs, event_parameters = get_datasets(args.composition, 'training', args.test)

    # One-hot encode the composition data if composition is to be predicted
    if 'comp' in args.predict:
        one_hot_map = {
            1:  [1, 0, 0, 0],
            4:  [0, 1, 0, 0],
            16: [0, 0, 1, 0],
            56: [0, 0, 0, 1]
        }
        event_parameters['comp'] = np.array([one_hot_map[c] for c in event_parameters['comp']])

    # Get number of events from an arbitrary key
    num_events = event_parameters['file_info'].shape[0]

    # Create training / validation split
    training_cut = np.random.rand(num_events) > cg.VALIDATION_SPLIT
    validation_cut = np.logical_not(training_cut)

    # Assign training data
    training_inputs = [model_input[training_cut] for model_input in model_inputs.values()]
    training_outputs = [event_parameters[output][training_cut] for output in args.predict]

    # Assign validation data
    validation_inputs = [model_input[validation_cut] for model_input in model_inputs.values()]
    validation_outputs = [event_parameters[output][validation_cut] for output in args.predict]

    input_shapes = {input_name: model_input.shape[1:] for input_name, model_input in model_inputs.items()}

    # This format for organizing the training/validation data allows for unlimited inputs/outputs
    # To adjust for multiple outputs, convert to a tuple
    # Also returns the shapes of all inputs so that we can use the Keras functional API
    return (
        # Training data
        tf.data.Dataset.from_tensor_slices(
            (tuple(training_inputs), tuple(training_outputs))).shuffle(np.sum(training_cut)).batch(cg.BATCH_SIZE),
        # Validation data
        tf.data.Dataset.from_tensor_slices(
            (tuple(validation_inputs), tuple(validation_outputs))).batch(cg.BATCH_SIZE),
    ), input_shapes


def get_datasets(composition, mode, test=False):
    '''Loads and prepares simulation data from files'''
    
    # Load detector inputs and event parameters
    simdata_folder_path = os.path.join(os.sep, 'data', 'user', 'fmcnally', 'icetop-cnn', 'simdata')
    detector_data, event_parameters = get_preprocessed(simdata_folder_path, cg.PREP['infill'], composition=composition, test=test)

    # Get training/assessment cut
    training_assessment_cut = get_training_assessment_cut(event_parameters, mode, cg.PREP)

    # Apply cuts
    detector_data = {
        detector_name: data[training_assessment_cut] for detector_name, data in detector_data.items()
    }
    event_parameters = {
        key: val[training_assessment_cut] for key, val in event_parameters.items()
    }
    
    # Prepare simulation data
    model_inputs = data_prep(detector_data, cg.PREP)
    
    # Add reconstruction data if it is used
    # TODO: Isolate into its own thing? Should not go in data_prep, that is designed for detector data
    if cg.PREP['reco']:
        assert f'{cg.PREP["reco"]}_dir' in event_parameters, f"Invalid reco choice, received {cg.PREP['reco']}"
        # Get and compute the zenith angle
        zenith = np.pi - event_parameters[f"{cg.PREP['reco']}_dir"].transpose()[0].astype('float32')
        # Normalize if specified
        if cg.PREP['normed']:
            zenith /= np.amax(zenith)
        # Add zenith to model inputs
        model_inputs[cg.PREP['reco']] = zenith
    
    return model_inputs, event_parameters


def create_or_restore_model(input_shapes):
    '''Creates or restores a model based on the command line arguments passed in by the user'''

    # User indicated that they would like to restore the model.
    # The submission script should have already confirmed its existence.
    if args.restore:
        return tf.keras.models.load_model(
            os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name, f'{args.model_name}.keras')
        )
    # Otherwise, create and return a new model
    return get_compiled_model(input_shapes, args.model_name, cg.PREP, args.predict)


def train_model(model, training_dataset, validation_dataset):
    '''Trains the model using the provided training/validation datasets and settings from "config"'''

    # Create models folder if it does not already exist
    if not os.path.exists(os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name)):
        os.makedirs(os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name))

    # Save model parameters and nuclei
    with open(os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name, f'{args.model_name}.json'), 'w') as f:
        json.dump({**cg.PREP, 'training_nuclei':args.composition}, f, indent=4)

    # Let 'er rip
    model.fit(
        training_dataset,
        epochs=args.epochs,
        validation_data=validation_dataset,
        callbacks=[
            tf.keras.callbacks.CSVLogger(
                os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name, f'{args.model_name}.csv'),
                append=args.restore
            ),
            tf.keras.callbacks.EarlyStopping(
                min_delta=cg.MIN_ES_DELTA,
                patience=cg.ES_PATIENCE,
                mode='min',
                restore_best_weights=cg.RESTORE_BEST_WEIGHTS
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=cg.LR_FACTOR,
                patience=cg.LR_PATIENCE,
                mode='min',
                min_delta=cg.MIN_LR_DELTA,
                cooldown=cg.LR_COOLDOWN,
                min_lr=cg.MIN_LR
            ),
        ]
    )
    tf.keras.models.save_model(
        model,
        os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name, f'{args.model_name}.keras'),
        save_format='keras'
    )
    return model


def assess_model(model: tf.keras.Model, assess_comp: str = 'phof'):
    '''Assesses the model once it has finished training'''

    # Load data used for training/testing in its entirety
    model_inputs, _ = get_datasets(assess_comp, 'assessment')

    # Assess the model
    reconstructions = model.predict(model_inputs.values())

    # Save the reconstruction(s)
    for i, prediction in enumerate(args.predict):
        if not os.path.exists(os.path.join(ICETOP_CNN_DATA_DIR, 'reconstructions', prediction)):
            os.makedirs(os.path.join(ICETOP_CNN_DATA_DIR, 'reconstructions', prediction))
        np.save(
            os.path.join(ICETOP_CNN_DATA_DIR, 'reconstructions', prediction, f'{args.model_name}.npy'),
            reconstructions if len(args.predict) == 1 else reconstructions[i]
        )
    
    # Create loss graphs
    os.system(f'python loss_grapher.py {args.model_name}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description = 'Trains ML models using TensorFlow')
    p.add_argument(
        '-n', '--name', dest='model_name', type=str, required=True,
        help='Name of the model to train')
    p.add_argument(
        '-c', '--composition', type=str, required=True,
        help='Composition of datasets to load and train the model on')
    p.add_argument(
        '-e', '--epochs', type=int, default=cg.MAX_EPOCHS,
        help='The maximum number of epochs that the model should train for')
    p.add_argument(
        '-p', '--predict', nargs='+', type=str, choices=['comp', 'energy'],
        help='A list of one or more desired model outputs')
    p.add_argument(
        '-t', '--test', action='store_true',
        help='Run the script off the cluster with a limited dataset')
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        '-o', '--overwrite', action='store_true',
        help='Overwrite a model with the same name if it exists. Can not be used with restore')
    g.add_argument(
        '-r', '--restore', action='store_true',
        help='Attempt to restore and continue training a model if it exists. Can not be used with overwrite')
    args = p.parse_args()

    # Ensure that there are no unrecognized characters in the composition string
    if not all(c in 'phof' for c in args.composition):
        p.error('Unrecognized composition dataset combination requested')
    # Sort the composition string into a predictable order
    args.composition = ''.join(sorted(args.composition, key=lambda c: list('phof').index(c)))

    # Ensure epochs are a valid number
    if args.epochs and args.epochs <= 0:
        p.error('Epochs must be a positive integer')

    # Transform to a set to remove duplicates
    # Sort to maintain a predictable order to process arguments
    args.predict = sorted(set(args.predict))

    main()
