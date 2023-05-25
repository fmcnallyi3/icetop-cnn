import os
from collections import defaultdict
from glob import glob
from warnings import catch_warnings, filterwarnings

import numpy as np

ASSESSMENT_SPLIT = 0.1


def mod_log(n):
    '''Modified logarithm convenient for dealing with zeros and maintaining sign'''
    return np.sign(n) * np.log10(np.abs(n) + 1)




def r_log(m):
    '''Modified antilogarithm convenient for dealing with zeros and maintaining sign'''
    return np.sign(m) * (10**np.abs(m) - 1)




def load_preprocessed(filepath, composition='pf'):
    '''Loads files from disk into RAM specified by composition'''

    def get_data_files(filename):
        '''Returns a list of all matching filenames found in the specified filepath'''
        return sorted(glob(f'{filepath}/{filename}_*.npy'))


    def get_alias_list():
        '''Returns a list of nuclei converted to I3 dataset aliases'''
        return [composition_aliases[nucleus] for nucleus in composition]


    def get_composition_files(filenames):
        '''Returns a list of all filenames which contain the desired composition(s)'''
        return [filename for filename in filenames if any([alias in filename for alias in get_alias_list()])]


    def load_detector_array(composition_files, detector_array_shape):
        '''Returns detector array data as a numpy array in the correct shape'''

        # Load data from disk into list
        detector_array = []
        for composition_file in composition_files:
            composition_data = np.load(composition_file)
            detector_array += [r for r in composition_data]

        # Reshape data into numpy array as desired
        return np.asarray(detector_array).reshape((-1,) + detector_array_shape + (8,))


    def load_event_parameters(composition_files):
        '''Returns event parameters as a dictionary mapping strings to numpy arrays'''

        # Load data from disk into a default-dictionary
        event_parameters = defaultdict(list)
        for composition_file in composition_files:
            composition_data = np.load(composition_file, allow_pickle=True).item()
            for key, key_data in composition_data.items():
                event_parameters[key] += [r for r in key_data]

        # Convert lists to numpy arrays
        for key, key_data in event_parameters.items():
            event_parameters[key] = np.asarray(key_data)

        return event_parameters
    
    
    # Ensure that a directory is actually found
    assert os.path.exists(filepath), f'Could not find simulation data folder specified: {filepath}'

    # Mapping of nuclei to IceCube Simulation IDs
    composition_aliases = {'p':'12360','h':'12630','o':'12631','f':'12362'}
    detector_inputs = {}

    # Prepare IceTop files/data
    icetop_files = get_composition_files(get_data_files('icetop'))
    icetop = load_detector_array(icetop_files, (10,10))
    detector_inputs['icetop'] = icetop

    # Prepare Infill files/data
    infill_files = get_composition_files(get_data_files('infill'))
    infill = load_detector_array(infill_files, (3,))
    detector_inputs['infill'] = infill

    # Prepare event parameter files/data
    event_files = get_composition_files(get_data_files('event_parameters'))
    event_parameters = load_event_parameters(event_files)
    
    return detector_inputs, event_parameters




def data_prep(detector_inputs, infill=False, clc=True, sta5=False, q=None, t=None, t_shift=False, t_clip=0, normed=False, reco=None):
    '''Augments and prepares raw data for model input'''

    def merge_tank_layers(array):
        '''Merges tank layers depending on whether soft local coincidences are being analyzed'''

        # Data is in the form q1h, q1s, q2h, q2s, t1h, t1s, t2h, t2s (charge/time, tank1/tank2, hard/soft local coincidences)
        # Convenient for switching between including soft local coincidences
        if clc:
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


    def merge_qt_layers(array):
        '''Merges charge and time values. Options are: None, False, "mean", "sum", "product", "min", or "max"'''

        ignore_values = {'mean':np.nan, 'min':np.inf, 'max':-np.inf}
        for layer_type, merge_type in zip(layer_slices.keys(), (q, t)):
            # Options for combining: mean, sum, product, min, max
            assert merge_type in (None, False, 'mean', 'sum', 'product', 'min', 'max'), f'Unexpected argument for {layer_type} layer merge type, received {merge_type}.'

            # Do nothing
            if merge_type == None:
                continue
            # Remove layers
            elif merge_type == False:
                array = np.delete(array, layer_slices[layer_type], axis=-1)
                continue

            # Converts zeros to special values on charge or time layers for mean/min/max functions
            if merge_type in ignore_values.keys():
                array[..., layer_slices[layer_type]][array[..., layer_slices[layer_type]] == 0] = ignore_values[merge_type]

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


    def update_layer_slices(layer_slices):
        '''Returns updated layer slices that reflect the state of the data after the potential merging of charge/time layers'''

        if q:
            layer_slices['charge'] = np.s_[:1]
        if t:
            layer_slices['time'] = np.s_[-1:]

        return layer_slices


    def log_charge(array):
        '''Takes modified logarithm of charge layers and returns array'''

        if q != False:
            array[..., layer_slices['charge']] = mod_log(array[..., layer_slices['charge']])

        return array


    def clip_time(detector_inputs):
        '''Clips the top t_clip% of times to not exceed the time at the (100 - t_clip)th percentile'''

        if t == False or t_clip <= 0 or t_clip > 100:
            return detector_inputs

        # Get list of all nonzero time values for each detector-based input.
        nonzero_time_values = [detector_input[..., layer_slices['time']][np.nonzero(detector_input[..., layer_slices['time']])] for detector_input in detector_inputs.values()]

        # Flatten list
        nonzero_time_values = [nonzero_time for nonzero_times in nonzero_time_values for nonzero_time in nonzero_times]

        # Find time value that corresponds to top t_clip% of nonzero time data
        clip_time = np.percentile(nonzero_time_values, 100 - t_clip)

        # Clip all time values that are greater than the clip_time corresponding to the t_clip% of data
        for detector_name, detector_input in detector_inputs.items():
            detector_inputs[detector_name][..., layer_slices['time']] = np.clip(detector_input[..., layer_slices['time']], None, clip_time)

        return detector_inputs
    

    def shift_time(detector_inputs):
        '''Shifts all time values such that the smallest nonzero time is shifted to zero'''

        if t == False or not t_shift:
            return detector_inputs

        # Get nonzero minimum time from all nonzero times across all time layers and inputs
        smallest_nonzero_time_value = np.min([np.min(detector_input[..., layer_slices['time']][np.nonzero(detector_input[..., layer_slices['time']])]) for detector_input in detector_inputs.values()])

        # Subtract nonzero mimimum time from all nonzero times across all time layers and inputs
        for detector_name, detector_input in detector_inputs.items():
            detector_inputs[detector_name][..., layer_slices['time']][np.nonzero(detector_input[..., layer_slices['time']])] -= smallest_nonzero_time_value

        return detector_inputs


    def normalize(detector_inputs):
        '''Normalizes charge/time data to be within the range [0,1]'''

        if not normed:
            return detector_inputs

        # Normalize charge and time layers
        for layer_type, merge_type in zip(layer_slices.keys(), (q, t)):
            if merge_type == False:
                continue

            # Get max value across all events for all detector inputs
            max_value = np.max([np.max(detector_input[..., layer_slices[layer_type]]) for detector_input in detector_inputs.values()])

            # Normalize values between the range [0,1] by dividing by the maximum value
            for detector_name in detector_inputs.keys():
                detector_inputs[detector_name][..., layer_slices[layer_type]] /= max_value

        return detector_inputs

    # Create a shallow copy of the detector inputs - merge_tank_layers should create deep copies of the data
    detector_inputs = dict(detector_inputs)

    # Remove infill if specified in model prep arguments
    if not infill:
        detector_inputs.pop('infill')

    # Merge tank layers based on whether we are analyzing soft local coincidences
    detector_inputs = {detector_name: merge_tank_layers(detector_input) for detector_name, detector_input in detector_inputs.items()}

    # Create slices to keep track of which channels are charge and time
    layer_slices = {'charge': np.s_[:2], 'time': np.s_[-2:]}

    # Merge charge and time layers with options specified in CONFIG
    detector_inputs = {detector_name: merge_qt_layers(detector_input) for detector_name, detector_input in detector_inputs.items()}

    # Update respective layer slices if either charge/time layers are merged
    layer_slices = update_layer_slices(layer_slices)

    # Take modified logarithm of charge values
    detector_inputs = {detector_name: log_charge(detector_input) for detector_name, detector_input in detector_inputs.items()}

    # Clip time values to not exceed some maximum value
    detector_inputs = clip_time(detector_inputs)

    # Shift time values such that minimum falls at zero
    detector_inputs = shift_time(detector_inputs)

    # Normalize detector data
    detector_inputs = normalize(detector_inputs)
    
    return detector_inputs




def get_training_assessment_cut(event_parameters, mode, sta5):
    '''Returns a cut based on whether the user is training or assessing a model'''

    # Make sure user selects a valid mode
    assert mode in ['training', 'assessment', None], f'Invalid mode choice, received {mode}'

    # Get number of events from an arbitrary key
    num_events = len(event_parameters['energy'])

    # Used for testing/debugging/statistics
    if not mode:
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
    if sta5:
        cut *= event_parameters['passed_STA5']

    return cut




def add_reco(model_inputs, cut_events, event_parameters, reco, normed):
    '''Adds directional input and modifies cut to remove NaNs from reconstruction data if zenith/azimuth is used'''
    assert reco in ['plane', 'laputop', 'small', None], f'Invalid reco choice, received {reco}'
    
    if reco:
        # Get and compute the zenith angle
        zenith = np.pi - event_parameters[f'{reco}_dir'].transpose()[0].astype('float64')

        # Normalize if specified
        if normed:
            zenith /= np.nanmax(zenith)

        # Add zenith to model inputs
        model_inputs[reco] = zenith

        # Filter NaNs from reconstruction data
        cut_events *= ~np.isnan(zenith)

    return model_inputs, cut_events




# Get cut to display on plots
def get_cuts(data_cut, event_parameters, cut_str):
    '''Returns a cut for reconstructed energy and a cut for true energy'''
    assert cut_str in ('No Cut', 'Quality Cut'), 'Bad cut! Options are {No Cut, Quality Cut}'

    # No Cut
    if cut_str == 'No Cut':
        return np.full(np.sum(data_cut), True), data_cut

    # Quality Cut
    if cut_str == 'Quality Cut':
        return event_parameters['quality_cut'][data_cut], event_parameters['quality_cut'] * data_cut
