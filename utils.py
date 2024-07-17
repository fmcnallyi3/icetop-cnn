from __future__ import annotations

import os
from collections import defaultdict
from glob import glob
from typing import Any
from warnings import catch_warnings, filterwarnings 

import numpy as np

ASSESSMENT_SPLIT = 0.1

########################################################################################################################
#
# USEFUL MATHEMATICAL OPERATIONS
#
########################################################################################################################

def mod_log(n):
    '''Modified logarithm convenient for dealing with zeros and maintaining sign.'''
    return np.sign(n) * np.log10(np.abs(n) + 1)

def mod_ilog(n):
    """Modified logarithm convenient for dealing with zeros and maintaining sign. \\
    This function modifies the input in-place."""
    np.multiply(np.sign(n), np.log10(np.abs(n) + 1), out=n)


def r_log(m):
    '''Modified antilogarithm convenient for dealing with zeros and maintaining sign.'''
    return np.sign(m) * (10**np.abs(m) - 1)

def r_ilog(m):
    """Modified antilogarithm convenient for dealing with zeros and maintaining sign. \\
    This function modifies the input in-place."""
    np.multiply(np.sign(m), (np.power(10, np.abs(m)) - 1), out=m)


########################################################################################################################
#
# FUNCTIONS ASSOCIATED WITH LOADING DETECTOR ARRAY DATA AND EVENT PARAMETERS
#
########################################################################################################################

def files(simdata_folder_path: str, data: str, composition: str = 'pf', test: bool = False) -> list[str]:
    '''Loads subset of desired data from given folder path based on composition.'''

    simdata_data_path = os.path.join(simdata_folder_path, data)

    # Ensure that a directory is actually found
    assert os.path.exists(simdata_data_path), f'Could not find simulation data folder: {simdata_data_path}'

    # Mapping of nuclei to IceCube Simulation IDs
    composition_aliases = {'p':'12360', 'h':'12630', 'o':'12631', 'f':'12362'}
    alias_list = [composition_aliases[nucleus] for nucleus in composition]

    composition_files = []
    for alias in sorted(alias_list):
        filenames = sorted(glob(os.path.join(simdata_data_path, f'{data}_{alias}_*.npy')))[:1 if test else None]
        composition_files.extend(filenames)

    return composition_files


### DETECTOR ARRAY DATA ###

def load_detector_array(filepaths: list[str]) -> np.ndarray:
    '''Loads data provided by IceTop detectors. Returns numpy ndarray.'''

    # Load times and memory usage can be greatly reduced by pre-allocating
    # space for the detector array. The entire sequence takes a fraction of
    # a second to compute, so we don't have to hard-code the length, which
    # may (will) change during the course of the project.
    num_events = 0
    for filepath in filepaths:
        composition_data = np.load(filepath, mmap_mode='r')
        num_events += composition_data.shape[0]
    else:
        detector_shape = composition_data.shape[1:]

    detector_array = np.empty((num_events,) + detector_shape) # Pre-allocate

    # Replace the garbage pre-allocated data with detector data in-place
    start_idx = 0
    for filepath in filepaths:
        composition_data = np.load(filepath).astype('float32')
        detector_array[start_idx : (start_idx := start_idx + composition_data.shape[0])] = composition_data

    return detector_array


def get_detector_inputs(simdata_folder_path: str, composition: str = 'pf', test: bool = False) -> dict[str, np.ndarray]:
    '''Loads detector inputs of desired composition. Returns a dictionary mapping input names to numpy ndarrays.'''

    # Prepare IceTop files/data
    icetop_files = files(simdata_folder_path, 'icetop', composition=composition, test=test)
    icetop = load_detector_array(icetop_files)

    # Prepare Infill files/data
    infill_files = files(simdata_folder_path, 'infill', composition=composition, test=test)
    infill = load_detector_array(infill_files)

    # Store detector-based data as dictionary
    detector_inputs = {'icetop':icetop, 'infill':infill}
    return detector_inputs


### EVENT PARAMETERS ###

def load_event_parameters(composition_files: list[str]) -> dict[str, np.ndarray]:
    '''Returns event parameters as a dictionary mapping strings to numpy arrays'''

    # It is faster to convert the array items to lists while
    # appending and convert back to an np.ndarray when done. I feel like this
    # probably has to do with a difference in Python vs. NumPy memory
    # allocation optimizations.

    # Load data from disk into a default-dictionary
    event_parameters = defaultdict(list)
    for composition_file in composition_files:
        composition_data = np.load(composition_file, allow_pickle=True).item()
        for key, key_data in composition_data.items():
            event_parameters[key] += list(key_data)
    
    # Convert lists to numpy arrays
    for key, key_data in event_parameters.items():
        event_parameters[key] = np.asarray(key_data)

    return event_parameters


def get_event_parameters(simdata_folder_path: str, composition: str = 'pf', test: bool = False) -> dict[str, np.ndarray]:
    '''Loads event parameters of desired composition. Returns a dictionary mapping parameter names to numpy ndarrays.'''

    # Prepare event parameter files/data
    event_files = files(simdata_folder_path, 'event_parameters', composition=composition, test=test)
    event_parameters = load_event_parameters(event_files)

    return event_parameters


### BOTH DETECTOR ARRAY DATA AND EVENT PARAMETERS ### 

def get_preprocessed(simdata_folder_path: str, composition: str = 'pf', test: bool = False) -> tuple[dict[str, np.ndarray]]:
    '''Loads both detector inputs and event parameters of desired composition.'''
    detector_inputs = get_detector_inputs(simdata_folder_path, composition=composition, test=test)
    event_parameters = get_event_parameters(simdata_folder_path, composition=composition, test=test)

    return detector_inputs, event_parameters


########################################################################################################################
#
# FUNCTIONS ASSOCIATED WITH TRANSFORMING DETECTOR ARRAY DATA
#
# ASSUMPTIONS: 
#   1) Detector array data is stored in the following format: (num_events, detector_array_shape, channels).
#      1a) detector_array_shape can be a scalar (infill) or a tuple (icetop). What matters is the channels.
#   2) There are 8 channels, and they are in the following order: (q1h, q1s, q2h, q2s, t1h, t1s, t2h, t2s).
#      2a) q/t -> charge/time | 1/2 -> tank1/tank2 | h/s -> hard local coincidence / soft local coincidence
#      2b) If you are having trouble visualizing this 4-dimensional data structure, it is essentially a row
#          of rank three tensors, which themselves can be visualized as rectangular prisms.
#
########################################################################################################################

def data_prep(detector_inputs: dict[str, np.ndarray], prep: dict[str, Any]):
    '''Augments and prepares raw data for model input'''

    # Create slices to keep track of which channels are charge and time
    layer_slices = {'charge': np.s_[:2], 'time': np.s_[-2:]}

    # Ensure prep dictionary contains all keys required for data preparation
    required_keys = ['infill', 'clc', 'sta5', 'q', 't', 't_shift', 't_clip', 'normed']
    assert all(key in prep for key in required_keys), 'Error: one or more required keys missing from prep arguments.'

    def merge_tank_layers(array: np.ndarray) -> np.ndarray:
        '''Merges tank layers depending on whether soft local coincidences are being analyzed'''

        if prep['clc']:
            # Want locations of maximum charge pulses
            tank1mask = array[..., 0] > array[..., 1]
            tank2mask = array[..., 2] > array[..., 3]
            
            # Create channels from locations of maximum charge pulses
            q1 = np.where(tank1mask, array[..., 0], array[..., 1])
            q2 = np.where(tank2mask, array[..., 2], array[..., 3])
            t1 = np.where(tank1mask, array[..., 4], array[..., 5])
            t2 = np.where(tank1mask, array[..., 6], array[..., 7])
            
            # Return new array with merged tank layers
            return np.stack((q1, q2, t1, t2), axis=-1)

        else:
            # Return every other layer excluding soft local coincidences
            return array[..., ::2]
        
    def merge_qt_layers(array: np.ndarray) -> np.ndarray:
        '''Merges charge and time values. Options are: None, False, "mean", "sum", "product", "min", or "max"'''

        ignore_values = {'mean':np.nan, 'min':np.inf, 'max':-np.inf}
        for layer_type, merge_type in zip(layer_slices.keys(), (prep['q'], prep['t'])):
            # Options for combining: do nothing, remove layers, mean, sum, product, min, max
            assert merge_type in (
                None, False, 'mean', 'sum', 'product', 'min', 'max'
            ), f'Unexpected argument for {layer_type} layer merge type, received {merge_type}.'

            # Do nothing
            if merge_type == None:
                continue
            # Remove layers
            elif merge_type == False:
                array = np.delete(array, layer_slices[layer_type], axis=-1)
                continue

            # Converts zeros to special values on charge or time layers for mean/min/max functions
            if merge_type in ignore_values.keys():
                array[..., layer_slices[layer_type]][
                    array[..., layer_slices[layer_type]] == 0
                ] = ignore_values[merge_type]

            # Mean requires special conversion back (since np.nan != np.nan)
            if merge_type == 'mean':
                # Special warning is given when all values in a mean calculation are np.nan
                # Returns np.nan, so we can ignore - math works out
                with catch_warnings():
                    filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
                    merged_layer =  np.nanmean(array[..., layer_slices[layer_type]], axis=-1, keepdims=True)
                merged_layer[merged_layer != merged_layer] = 0 # Remember np.nan != np.nan
            elif merge_type == 'sum':
                merged_layer = np.sum(array[..., layer_slices[layer_type]], axis=-1, keepdims=True)
            elif merge_type == 'product':
                merged_layer = np.prod(array[..., layer_slices[layer_type]], axis=-1, keepdims=True)
            elif merge_type == 'min':
                merged_layer = np.amin(array[..., layer_slices[layer_type]], axis=-1, keepdims=True)
            elif merge_type == 'max':
                merged_layer = np.amax(array[..., layer_slices[layer_type]], axis=-1, keepdims=True)

            # Convert zeros back
            if merge_type in ['min','max']:
                merged_layer[merged_layer == ignore_values[merge_type]] = 0

            # Maintain channel order
            if layer_type == 'charge':
                array = np.concatenate((merged_layer, np.delete(array, layer_slices[layer_type], -1)), axis=-1)
            elif layer_type == 'time':
                array = np.concatenate((np.delete(array, layer_slices[layer_type], -1), merged_layer), axis=-1)

        return array

    def update_layer_slices(layer_slices: dict[str, np.s_]) -> dict[str, np.s_]:
        '''Returns updated layer slices after the potential merging of charge/time layers'''

        if prep['q']:
            layer_slices['charge'] = np.s_[:1]
        if prep['t']:
            layer_slices['time'] = np.s_[-1:]

        return layer_slices

    def log_charge(array: np.ndarray) -> np.ndarray:
        '''Takes modified logarithm of charge layers and returns array'''

        if prep['q'] != False:
            mod_ilog(array[..., layer_slices['charge']])

        return array

    def clip_time(detector_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        '''Clips the top t_clip% of times to not exceed the time at the (100 - t_clip)th percentile'''

        if prep['t'] == False:
            return detector_inputs
            
        assert 0 <= prep['t_clip'] < 100, f"Invalid value for t_clip: {prep['t_clip']}. Choose value in range [0, 100)."

        # Get list of all nonzero time values for each detector-based input.
        nonzero_time_values = [detector_input[..., layer_slices['time']][
            np.nonzero(detector_input[..., layer_slices['time']])
        ] for detector_input in detector_inputs.values()]

        # Flatten list
        nonzero_time_values = [nonzero_time for nonzero_times in nonzero_time_values for nonzero_time in nonzero_times]

        # Find time value that corresponds to top t_clip% of nonzero time data
        clip_time = np.percentile(nonzero_time_values, 100 - prep['t_clip'])

        # Clip all time values that are greater than the clip_time corresponding to the t_clip% of data
        for detector_name, detector_input in detector_inputs.items():
            detector_inputs[detector_name][..., layer_slices['time']] = np.clip(
                detector_input[..., layer_slices['time']],
                None, clip_time
            )

        return detector_inputs
    
    def shift_time(detector_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        '''Shifts all time values such that the smallest nonzero time is shifted to zero'''

        if prep['t'] == False or not prep['t_shift']:
            return detector_inputs

        # Get nonzero minimum time from all nonzero times across all time layers and inputs
        smallest_nonzero_time_value = np.min([
            np.min(
                detector_input[..., layer_slices['time']][np.nonzero(
                    detector_input[..., layer_slices['time']]
                )]
            ) for detector_input in detector_inputs.values()
        ])

        # Subtract nonzero mimimum time from all nonzero times across all time layers and inputs
        for detector_name, detector_input in detector_inputs.items():
            detector_inputs[detector_name][..., layer_slices['time']][np.nonzero(
                detector_input[..., layer_slices['time']]
            )] -= smallest_nonzero_time_value

        return detector_inputs

    def normalize(detector_inputs: dict[str, np.ndarray]) -> np.ndarray:
        '''Normalizes charge/time data to be within the range [0,1]'''

        if not prep['normed']:
            return detector_inputs

        # Normalize charge and time layers
        for layer_type, merge_type in zip(layer_slices.keys(), (prep['q'], prep['t'])):
            if merge_type == False:
                continue

            # Get max value across all events for all detector inputs
            max_value = np.max([
                np.max(detector_input[..., layer_slices[layer_type]]) for detector_input in detector_inputs.values()
            ])

            # Normalize values between the range [0,1] by dividing by the maximum value
            for detector_name in detector_inputs.keys():
                detector_inputs[detector_name][..., layer_slices[layer_type]] /= max_value

        return detector_inputs

    # Remove infill if specified in model prep arguments
    if not prep['infill'] and 'infill' in detector_inputs:
        detector_inputs.pop('infill')

    # Merge tank layers based on whether we are analyzing soft local coincidences
    for detector_name, detector_input in detector_inputs.items():
        detector_inputs[detector_name] = merge_tank_layers(detector_input)

    # Merge charge and time layers with options specified in CONFIG
    for detector_name, detector_input in detector_inputs.items():
        detector_inputs[detector_name] = merge_qt_layers(detector_input)

    # Update respective layer slices if either charge/time layers are merged
    layer_slices = update_layer_slices(layer_slices)

    # Take modified logarithm of charge values
    for detector_name, detector_input in detector_inputs.items():
        detector_inputs[detector_name] = log_charge(detector_input)

    # Clip time values to not exceed some maximum value
    detector_inputs = clip_time(detector_inputs)

    # Shift time values such that minimum falls at zero
    detector_inputs = shift_time(detector_inputs)

    # Normalize detector data
    detector_inputs = normalize(detector_inputs)
    
    return detector_inputs


########################################################################################################################
#
# FUNCTIONS ASSOCIATED WITH CUTTING EVENTS FROM THE OVERALL DATASET
#
########################################################################################################################

def get_training_assessment_cut(event_parameters: dict[str, np.ndarray], mode: str | None, prep: dict[str, Any]) -> np.ndarray:
    '''Returns a cut based on whether the user is training or assessing a model'''

    # Make sure user selects a valid mode
    assert mode in ['training', 'assessment', None], f'Invalid mode choice, received {mode}'

    # Ensure prep dictionary contains all keys required for cutting data
    required_keys = ['sta5', 'reco']
    assert all(key in prep for key in required_keys), 'Error: one or more required keys missing from prep arguments.'

    # Get number of events from an arbitrary key
    num_events = event_parameters['file_info'].shape[0]

    # Used for testing/debugging/statistics
    if mode is None:
        return np.full(num_events, True)

    # Create the same randomized cut each time
    # Seed is arbitrary - what's important is that it's set and never changed
    np.random.seed(1148)

    # Create training cut
    cut = np.random.rand(num_events) > ASSESSMENT_SPLIT
    # Assessment cut is the complement of the training cut
    if mode == 'assessment':
        cut = np.logical_not(cut)

    # Optionally remove events that do not pass the STA5 filter
    if prep['sta5']:
        cut *= event_parameters['passed_STA5']
        
    # Filter NaNs from directional reconstruction data if used
    if prep['reco']:
        assert prep['reco'] in ['plane', 'laputop', 'small'], f"Invalid reco choice, received {prep['reco']}"
        cut *= ~np.isnan(event_parameters[f"{prep['reco']}_dir"].transpose()[0].astype('float64'))

    return cut
    

# Get cut to display on plots
def get_cuts(data_cut, event_parameters, cut_str):
    '''Returns a cut for reconstructed parameter and a cut for true parameter'''
    available_cuts = {'No Cut', 'Uncontained Cut', 'Quality Cut'}
    assert cut_str in available_cuts, f'Bad cut! Options are {available_cuts}'

    # No Cut
    if cut_str == 'No Cut':
        return np.full(np.sum(data_cut), True), data_cut
    # Uncontained Cut
    if cut_str == 'Uncontained Cut':
        return event_parameters['uncontained_cut'][data_cut], data_cut * event_parameters['uncontained_cut']
    # Quality Cut
    if cut_str == 'Quality Cut':
        return event_parameters['quality_cut'][data_cut], data_cut * event_parameters['quality_cut']