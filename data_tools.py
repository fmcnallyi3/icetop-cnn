from ast import literal_eval
from collections import defaultdict
from glob import glob
from sys import version_info
from warnings import catch_warnings, filterwarnings
import numpy as np


# Modified logarithm convenient for dealing with zeros and maintaining sign
def mod_log(n):
    return np.sign(n) * np.log10(np.abs(n) + 1)


# Reverse the above process
def r_log(m):
    return np.sign(m) * (10**np.abs(m) - 1)


# Hard-coded string to pix stolen from Mirco (deepLearning/hexagon/hexConv.py)
def hex_square():
    sta_to_hex = {'01':(-4,-1), '02':(-4,0), '03':(-4,1), '04':(-4,2), '05':(-4,3), '06':(-4,4), '07':(-3,-2), '08':(-3,-1), '09':(-3,0), '10':(-3,1), '11':(-3,2), '12':(-3,3), '13':(-3,4), '14':(-2,-3), '15':(-2,-2), '16':(-2,-1), '17':(-2,0), '18':(-2,1), '19':(-2,2), '20':(-2,3), '21':(-2,4), '22':(-1,-4), '23':(-1,-3), '24':(-1,-2), '25':(-1,-1), '26':(-1,0), '27':(-1,1), '28':(-1,2), '29':(-1,3), '30':(-1,4), '31':(0,-5), '32':(0,-4), '33':(0,-3), '34':(0,-2), '35':(0,-1), '36':(0,0), '37':(0,1), '38':(0,2), '39':(0,3), '40':(0,4), '41':(1,-5), '42':(1,-4), '43':(1,-3), '44':(1,-2), '45':(1,-1), '46':(1,0), '47':(1,1), '48':(1,2), '49':(1,3), '50':(1,4), '51':(2,-5), '52':(2,-4), '53':(2,-3), '54':(2,-2), '55':(2,-1), '56':(2,0), '57':(2,1), '58':(2,2), '59':(2,3), '60':(3,-5), '61':(3,-4), '62':(3,-3), '63':(3,-2), '64':(3,-1), '65':(3,0), '66':(3,1), '67':(3,2), '68':(4,-5), '69':(4,-4), '70':(4,-3), '71':(4,-2), '72':(4,-1), '73':(4,0), '74':(4,1), '75':(5,-5), '76':(5,-4), '77':(5,-3), '78':(5,-2)}
    # Inverse of above
    hex_to_sta = {v:k for k, v in sta_to_hex.items()}

    return sta_to_hex, hex_to_sta


# Hard-coded edge stations on pixelized array (see backend notebook for more)
def edge_idxs():
    idxs = [[0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [1, 3], [1, 9], [2, 2], [2, 9], [3, 1], [3, 9], [4, 0], [4, 9], [5, 0], [5, 9], [6, 0], [6, 8], [7, 0], [7, 7], [8, 0], [8, 4], [8, 5], [8, 6], [9, 0], [9, 1], [9, 2], [9, 3]]
    return np.asarray(idxs)


# Get cut to display on plots
def get_cut(cut_str, y, reco, recoE, data_cut):

    # No Cut
    if cut_str == 'No Cut':
        return np.array([True for _ in recoE]), y['energy'][get_data_cut(reco, y, data_cut)]
    # Quality Cut
    if cut_str == 'Quality Cut':
        return y['quality_cut'][get_data_cut(reco, y, data_cut)], y['energy'][y['quality_cut'] * get_data_cut(reco, y, data_cut)]

    raise Exception('Bad cut!')


def load_data(file_list, infill=False):

    d = defaultdict(list)
    single_keys = ['gain','position'] # Keys not tied to events
    dom_keys = ['charge_HLC', 'charge_SLC', 'time_HLC', 'time_SLC'] # Keys tied to DOMs

    # Concatenate all information tied to events
    for _, f in enumerate(file_list):
        if version_info.major == 3:
            d_i = np.load(f, allow_pickle=True)
        else:
            d_i = np.load(f)
        d_i = d_i.item()
        for key in d_i.keys():
            if key not in single_keys:
                d[key] += d_i[key]

    # Only copy in the detector information once
    for key in single_keys:
        d[key] = d_i[key]

    # Remove infill array
    if not infill:
        for key in dom_keys:
            d[key] = [{k:v for k, v in dom_series.items() if int(k[:2])<79}
                     for dom_series in d[key]]

    return d


""" Convert dictionary format to multi-layered array for use with CNN """
def dict_to_mat(d):

    sta_to_hex, _ = hex_square()
    station_list = sorted(sta_to_hex.keys())
    squareY, squareX = np.transpose([sta_to_hex[s] for s in station_list])
    tank1s = ['61','62']
    tank2s = ['63','64']

    nevents = len(d['charge_HLC'])
    nx = squareX.max() - squareX.min() + 1
    ny = squareY.max() - squareY.min() + 1
    depth = 8   # q1h, q1s, q2h, q2s, t1h, t1s, t2h, t2s
    data = np.zeros((nevents, ny, nx, depth))

    # Layers #1 and 3: HLC charge at each tank
    for i, charges in enumerate(d['charge_HLC']):
        for dom, charge in charges.items():
            y_i = sta_to_hex[dom[:2]][0] - squareY.min()
            x_i = sta_to_hex[dom[:2]][1] - squareX.min()
            if dom[2:] in tank1s:
                data[i][y_i][x_i][0] += mod_log(charge)
            if dom[2:] in tank2s:
                data[i][y_i][x_i][2] += mod_log(charge)

    # Layers #2 and 4: SLC charge at each tank
    for i, charges in enumerate(d['charge_SLC']):
        for dom, charge in charges.items():
            y_i = sta_to_hex[dom[:2]][0] - squareY.min()
            x_i = sta_to_hex[dom[:2]][1] - squareX.min()
            if dom[2:] in tank1s:
                data[i][y_i][x_i][1] += mod_log(charge)
            if dom[2:] in tank2s:
                data[i][y_i][x_i][3] += mod_log(charge)

    # Layers #5 and 7: HLC time at each tank
    for i, times in enumerate(d['time_HLC']):
        for dom, time in times.items():
            y_i = sta_to_hex[dom[:2]][0] - squareY.min()
            x_i = sta_to_hex[dom[:2]][1] - squareX.min()
            if dom[2:] in tank1s:
                data[i][y_i][x_i][4] += time
            if dom[2:] in tank2s:
                data[i][y_i][x_i][6] += time

    # Layers #6 and 8: SLC time at each tank
    for i, times in enumerate(d['time_SLC']):
        for dom, time in times.items():
            y_i = sta_to_hex[dom[:2]][0] - squareY.min()
            x_i = sta_to_hex[dom[:2]][1] - squareX.min()
            if dom[2:] in tank1s:
                data[i][y_i][x_i][5] += time
            if dom[2:] in tank2s:
                data[i][y_i][x_i][7] += time

    return data


def load_preprocessed(file_path, comp=['p','f']):
    
    # Load valid x and y files
    x_files = sorted(glob('%s/x_*.npy' % file_path))
    y_files = sorted(glob('%s/y_*.npy' % file_path))

    # Load only files specified in comp (default proton and iron)
    comp_dict = {'p':'12360','h':'12630','o':'12631','f':'12362'}
    sim_list = [comp_dict[c] for c in comp]
        
    x_files = [f for f in x_files if any([sim in f for sim in sim_list])]
    y_files = [f for f in y_files if any([sim in f for sim in sim_list])]

    x = []
    y = defaultdict(list)
    for i, xf in enumerate(x_files):
        x_i = np.load(xf)
        x += [r for r in x_i]
        # Loading code specific to python version
        if version_info.major == 3:
            y_i = np.load(y_files[i], allow_pickle=True, encoding='latin1')
        else:
            y_i = np.load(y_files[i])
        y_i = y_i.item()
        for key in y_i.keys():
            y[key] += [r for r in y_i[key]]

    x = np.asarray(x).reshape(-1,10,10,8)

    for key in y.keys():
        y[key] = np.asarray(y[key])

    return x, y


""" Catch-all function designed to prepare data in a variety of common ways. 
 - Can combine charge or time layers to one value per station
 - Can normalize charge, time, or both
"""
def data_prep(x_raw, y, mode, clc=True, sta5=False, q=None, t=None, t_shift=False, t_clip=0, normed=False, reco=None, cosz=False):

    if q == False:
        raise Exception('Why train the model without charge? It is not worth it, promise.')
    if t == False and reco == None:
        raise Exception('Why train the model on charge alone? It is not worth it, promise.')

    # Make sure user selects a valid mode
    if mode not in ['assessment', 'train', None]:
        print('Invalid mode choice')
        raise

    if clc:
        q1h, q1s, q2h, q2s, t1h, t1s, t2h, t2s = x_raw.transpose(3,0,1,2)
        x = np.array([q1h+q1s, q2h+q2s, t1h+t1s, t2h+t2s]).transpose(1,2,3,0)
    else:
        x = x_raw[...,::2]

    qt_dict = {'q':q, 't':t}
    new_dim = 4
    for k, merge_type in qt_dict.items():
        # False = remove the layer
        if merge_type == False:
            new_dim -= 2
        # None indicates do nothing to the layers, everything else merges
        elif merge_type != None:
            new_dim -= 1

    out_shape = (x.shape[0], x.shape[1], x.shape[2], new_dim)
    out_array = np.zeros(out_shape)

    ## Suggestions for alternate methods of merging layers
    ## - use charge associated with earliest arrival time 
    ## - use time associated with maximum charge

    # Setup for merging tanks to stations in common ways
    for k, merge_type in qt_dict.items():

        if merge_type == False:
            continue

        if k == 'q':
            if merge_type == None:
                out_array[...,:2] = x[...,:2]
                continue
            a1, a2 = x[...,:2].transpose(3,0,1,2)
            out_idx = 0
            
        elif k == 't':
            if merge_type == None:
                out_array[...,-2:] = x[...,-2:]
                continue
            a1, a2 = x[...,-2:].transpose(3,0,1,2)
            out_idx = -1            

        # Need to convert zeros for mean/min/max functions
        # Note: if there are np.nan's in data and you run the mean function,
        # they will be converted to zeros upon exit!
        ignore_values = {'mean':np.nan, 'min':np.inf, 'max':-np.inf}
        if merge_type in ignore_values.keys():
            a1[a1==0] = ignore_values[merge_type]
            a2[a2==0] = ignore_values[merge_type]

        # Options for combining: mean, sum, product, min, max
        if merge_type not in ['mean','sum','product','min','max']:
            print('Unexpected argument for %s' % k)
            raise

        # Mean requires special conversion back (since np.nan!=np.nan)
        if merge_type == 'mean':
            with catch_warnings():
                filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
                if k == 'q':
                    new_out = mod_log(np.nanmean([r_log(a1), r_log(a2)], axis=0))
                elif k == 't':
                    new_out = np.nanmean([a1,a2], axis=0)
            new_out[new_out != new_out] = 0
        elif merge_type == 'sum':
            if k == 'q':
                new_out = mod_log(r_log(a1) + r_log(a2))
            elif k == 't':
                new_out = a1 + a2
        elif merge_type == 'product':
            if k == 'q':
                new_out = a1 + a2   # sum of logs = product
            elif k == 't':
                new_out = a1 * a2
        elif merge_type == 'min':
            new_out = np.min([a1,a2], axis=0)
        elif merge_type == 'max':
            new_out = np.max([a1,a2], axis=0)

        # Convert zeros back
        if merge_type in ['min','max']:
            new_out[new_out == ignore_values[merge_type]] = 0

        out_array[...,out_idx] = new_out

    # Check if charge layers are merged, index is where charge layers end and time layers begin
    idx = 2 if q == None else 1

    if t != False:
        # Prepare all nonzero time values and sort them in ascending order
        nonzero_time_values = np.sort(out_array[...,idx:][np.nonzero(out_array[...,idx:])])
        # Time Clip
        if 0 < t_clip <= 100:
            # Find index of time value that corresponds to the % of data
            clip_idx = int((len(nonzero_time_values) - 1) * (1 - t_clip*.01))
            # Clip all time values that are greater than the time value corresponding to the % of data
            out_array[...,idx:] = np.clip(out_array[...,idx:], None, nonzero_time_values[clip_idx])
        # Time Shift
        if t_shift:
            # Subtract nonzero mimimum from all nonzero values across all time layers
            out_array[...,idx:][np.nonzero(out_array[...,idx:])] -= nonzero_time_values[0]
        

    # Normalization
    if normed:
        # Normalize charge
        out_array[...,:idx] /= out_array[...,:idx].max(keepdims=True)
        if t != False:
            # Normalize time
            out_array[...,idx:] /= out_array[...,idx:].max(keepdims=True)


    # Goal: Isolate a set of events to be used for assessment.
    # No model should ever be trained on these events.
    # Temporary approach, just isolate 10% of total sample

    # Create the same randomized array each time
    np.random.seed(1148)

    cut = (np.random.uniform(size=out_array.shape[0]) > 0.1).astype(bool)
    if mode == 'assessment':
        cut = np.logical_not(cut)
    if sta5:
        cut *= y['passed_STA5']

    # Keep NaN's in with reconstruction so we can tell which events to ignore
    if reco != None:
        th = y['{}_dir'.format(reco)].transpose()[0].astype('float')
        th = np.pi - th
        if cosz:
            th = np.cos(th)
        elif normed:
            th /= np.nanmax(th)
        out_array = [out_array, th]
    
    return out_array, idx, cut


""" Filter NaNs from reconstruction, if any """
def get_data_cut(reco, y, cut):
    return cut if reco == None else ~np.isnan(y['{}_dir'.format(reco)].transpose()[0].astype('float')) * cut


def load_mc(file_path):

    mc_files = sorted(glob('%s/sim_*_mc.npy' % file_path))

    mc = defaultdict(list)
    for _, f in enumerate(mc_files):
        if version_info.major == 3:
            mc_i = np.load(f, allow_pickle=True)
        else:
            mc_i = np.load(f)
        mc_i = mc_i.item()
        for key in mc_i.keys():
            mc[key] += mc_i[key]

    for key in mc.keys():
        mc[key] = np.asarray(mc[key])

    return mc


def name_model(prep, prefix=''):
    out_str = prefix
    for key, value in prep.items():
        out_str = '{}_{}.{}'.format(out_str, key, value)
    return out_str


def name_to_prep(name):
    prep = {}
    args = [a.split('.') for a in name.split('_') if len(a.split('.'))==2]
    for key, value in args:
        try: prep[key] = literal_eval(value)
        except ValueError:
            prep[key] = value
    return prep
