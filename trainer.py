#!/usr/bin/env python3

import argparse
import json
import os
from glob import glob

ERROR_ENVIRONMENT_NOT_ACTIVATED = 'Virtual environment not activated. Activate with the command "icetop-cnn"'

# Check that the environment has been activated
# Should only check when not run on the condor cluster
# Should check before importing external libraries
if not os.getenv('_CONDOR_SLOT'):
    assert os.getenv('VIRTUAL_ENV'), ERROR_ENVIRONMENT_NOT_ACTIVATED

# Supress debugging information
# Can remove safely, just makes for cleaner output
# Should be set before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Do not preallocate all GPU memory; big datasets OOM the copy to an
# eager constant otherwise (Dst tensor is not initialized).
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf

import config as cg
from utils import data_prep, files, get_event_parameters, load_detector_array, get_preprocessed, get_training_assessment_cut
from model import get_compiled_model

ICETOP_CNN_DIR = os.getenv('ICETOP_CNN_DIR')
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
    """Prepares streaming tensorflow datasets used for training.

    The full detector array for multi-million event datasets does not fit in
    slot memory. Instead: event parameters load whole (small), and detector
    data streams one simdata batch file at a time. Cuts and data_prep are
    applied per file exactly as the whole-load path applies them, and a global
    index permutation preserves full-event shuffle, so training semantics are
    unchanged. Peak RAM is one batch file plus parameters.
    """

    simdata_folder_path = args.simdata

    # Event parameters: small, load whole. Cuts need global event order.
    event_parameters = get_event_parameters(simdata_folder_path, composition=args.composition, test=args.test)

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

    # Assessment cut first: it seeds np.random(1148) internally, so the
    # training/validation split drawn after it is identical for every seed.
    training_assessment_cut = get_training_assessment_cut(event_parameters, 'training', cg.PREP)
    training_cut = np.random.rand(num_events) > cg.VALIDATION_SPLIT
    validation_cut = np.logical_not(training_cut)

    train_rows = np.where(training_cut & training_assessment_cut)[0]
    val_rows = np.where(validation_cut & training_assessment_cut)[0]

    # File layout: icetop and event_parameters files share order and row counts
    batch_files = files(simdata_folder_path, 'icetop', composition=args.composition, test=args.test)
    infill_files = files(simdata_folder_path, 'infill', composition=args.composition, test=args.test) if cg.PREP['infill'] else None

    # Row offset of each file (cheap: event parameter files are small)
    ep_files = files(simdata_folder_path, 'event_parameters', composition=args.composition, test=args.test)
    counts = [len(np.load(ef, allow_pickle=True).item()['energy']) for ef in ep_files]
    boundaries = np.cumsum([0] + counts)
    file_of_row = np.searchsorted(boundaries[1:], np.arange(num_events), side='right')

    # Prep detector files once at startup and keep only the prepared result.
    # Raw files are float64 with 8 tank channels (~6.4 kB/event, ~38 GB for
    # 6M events); data_prep merges to the 4 model channels, so casting the
    # result to float32 gives ~1.6 kB/event (~9.6 GB total). All prep ops are
    # per-event with fixed constants, so per-file prep is identical to
    # per-batch prep. Batches then gather from RAM with no TF-side copy, and
    # epochs run at RAM speed. mmap streaming was tried first: random page
    # reads over ceph cost ~20 min/epoch and jobs hit the 2 h slot limit
    # before early stopping fired.
    detector_mmaps = {
        'icetop': [data_prep({'icetop': np.load(bf)}, cg.PREP)['icetop'].astype('float32') for bf in batch_files],
    }
    if infill_files:
        detector_mmaps['infill'] = [data_prep({'infill': np.load(bf)}, cg.PREP)['infill'].astype('float32') for bf in infill_files]

    # Optional reco input (zenith) comes from parameters, already in RAM
    reco_inputs = {}
    if cg.PREP['reco']:
        zenith = np.pi - event_parameters[f"{cg.PREP['reco']}_dir"].transpose()[0].astype('float32')
        if cg.PREP['normed']:
            zenith /= np.amax(zenith)
        reco_inputs[cg.PREP['reco']] = zenith

    def load_rows(rows):
        """Gather detector rows for the given global indices, reading each
        involved batch file once. Returns dict of prepared input arrays."""
        # Gather per file (file order), then undo the regrouping so inputs
        # line up with outputs row-for-row. Without the reorder, labels are
        # permuted within each batch and the model only learns the mean.
        file_ids = file_of_row[rows]
        by_file = np.argsort(file_ids, kind='stable')
        back = np.empty_like(by_file)
        back[by_file] = np.arange(len(rows))
        parts = {name: [] for name in (['icetop'] + (['infill'] if infill_files else []))}
        for fi in np.unique(file_ids):
            pos = by_file[file_ids[by_file] == fi]
            local = rows[pos] - boundaries[fi]
            parts['icetop'].append(detector_mmaps['icetop'][fi][local])
            if infill_files:
                parts['infill'].append(detector_mmaps['infill'][fi][local])
        out = {name: np.concatenate(pl)[back] for name, pl in parts.items() if pl}
        out.update({name: arr[rows] for name, arr in reco_inputs.items()})
        return out

    def training_gen():
        order = np.random.permutation(train_rows)
        for s0 in range(0, len(order), cg.BATCH_SIZE):
            b = order[s0:s0 + cg.BATCH_SIZE]
            inputs = load_rows(b)
            outputs = [event_parameters[output][b] for output in args.predict]
            yield (tuple(inputs.values()), tuple(outputs))

    def validation_gen():
        order = np.random.permutation(val_rows)
        for s0 in range(0, len(order), cg.BATCH_SIZE):
            b = order[s0:s0 + cg.BATCH_SIZE]
            inputs = load_rows(b)
            outputs = [event_parameters[output][b] for output in args.predict]
            yield (tuple(inputs.values()), tuple(outputs))

    # Shapes from a probe batch (also primes the file layout)
    probe_rows = train_rows[:cg.BATCH_SIZE]
    probe = load_rows(probe_rows)
    input_shapes = {name: arr.shape[1:] for name, arr in probe.items()}
    sig = (
        tuple(tf.TensorSpec((None,)+probe[name].shape[1:], probe[name].dtype) for name in probe),
        tuple(tf.TensorSpec((None,)+event_parameters[output].shape[1:], event_parameters[output].dtype) for output in args.predict),
    )

    return (
        tf.data.Dataset.from_generator(training_gen, output_signature=sig).prefetch(2),
        tf.data.Dataset.from_generator(validation_gen, output_signature=sig),
    ), input_shapes


def get_datasets(composition, mode, test=False):
    '''Loads and prepares simulation data from files'''
    
    # Load detector inputs and event parameters
    simdata_folder_path = args.simdata
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
    # Seed all RNGs so runs are reproducible; different --seed values give
    # genuinely independent models on the same dataset (study requirement).
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    '''Creates or restores a model based on the command line arguments passed in by the user'''

    # User indicated that they would like to restore the model.
    # The submission script should have already confirmed its existence.
    if args.restore:
        return tf.keras.models.load_model(
            os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name, f'{args.model_name}.keras')
        )
    # Otherwise, create and return a new model
    return get_compiled_model(input_shapes, args.model_name, args.model_design, cg.PREP, args.predict)


def train_model(model, training_dataset, validation_dataset):
    '''Trains the model using the provided training/validation datasets and settings from "config"'''

    # Create models folder if it does not already exist
    if not os.path.exists(os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name)):
        os.makedirs(os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name))

    # Save model parameters and nuclei
    with open(os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name, f'{args.model_name}.json'), 'w') as f:
        json.dump({**cg.PREP, 'training_nuclei':args.composition, 'simdata':args.simdata, **({'seed':args.seed} if args.seed is not None else {})}, f, indent=4)

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
    reconstructions = model.predict(list(model_inputs.values()))

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
        '-n', '--name', dest='model_name', type=str,
        required=True,
        help='Name of the model to train')
    p.add_argument(
        '-c', '--composition', type=str,
        required=True,
        help='Composition of datasets to load and train the model on')
    p.add_argument(
        '-e', '--epochs', type=int,
        default=cg.MAX_EPOCHS,
        help='The maximum number of epochs that the model should train for')
    p.add_argument(
        '-m', '--model', dest='model_design', type=str,
        choices=sorted(os.path.splitext(arch)[0] for arch in glob('*.py', root_dir=f'{ICETOP_CNN_DIR}/architectures')),
        required=True,
        help='Desired model architecture')
    p.add_argument(
        '-p', '--predict', nargs='+', type=str,
        choices=['comp', 'energy'],
        required=True,
        help='A list of one or more desired model outputs')
    p.add_argument(
        '--seed', dest='seed', type=int, default=None,
        help='Random seed; different values give independent models on the same dataset')
    p.add_argument(
        '-t', '--test', action='store_true',
        help='Run the script off the cluster with a limited dataset')
    p.add_argument(
        '-d', '--simdata', dest='simdata', default=os.path.join(os.sep, 'data', 'user', 'fmcnally', 'icetop-cnn', 'simdata'),
        help='Specify the path of the simulation data to train a model off of.')
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
