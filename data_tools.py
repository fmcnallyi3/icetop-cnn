#!/usr/bin/env python

from glob import glob
import numpy as np
from collections import defaultdict
import sys
import ast


# Modified logarithm convenient for dealing with zeros and maintaining sign
def modLog(n):
    return np.sign(n) * np.log10(np.abs(n) + 1)

# Reverse the above process
def rLog(m):
    return np.sign(m) * (10**np.abs(m) - 1)


# Hard-coded string to pix stolen from Mirco (deepLearning/hexagon/hexConv.py)
def hexSquare():
    sta2hex = {'01':(-4,-1), '02':(-4,0), '03':(-4,1), '04':(-4,2), '05':(-4,3), '06':(-4,4), '07':(-3,-2), '08':(-3,-1), '09':(-3,0), '10':(-3,1), '11':(-3,2), '12':(-3,3), '13':(-3,4), '14':(-2,-3), '15':(-2,-2), '16':(-2,-1), '17':(-2,0), '18':(-2,1), '19':(-2,2), '20':(-2,3), '21':(-2,4), '22':(-1,-4), '23':(-1,-3), '24':(-1,-2), '25':(-1,-1), '26':(-1,0), '27':(-1,1), '28':(-1,2), '29':(-1,3), '30':(-1,4), '31':(0,-5), '32':(0,-4), '33':(0,-3), '34':(0,-2), '35':(0,-1), '36':(0,0), '37':(0,1), '38':(0,2), '39':(0,3), '40':(0,4), '41':(1,-5), '42':(1,-4), '43':(1,-3), '44':(1,-2), '45':(1,-1), '46':(1,0), '47':(1,1), '48':(1,2), '49':(1,3), '50':(1,4), '51':(2,-5), '52':(2,-4), '53':(2,-3), '54':(2,-2), '55':(2,-1), '56':(2,0), '57':(2,1), '58':(2,2), '59':(2,3), '60':(3,-5), '61':(3,-4), '62':(3,-3), '63':(3,-2), '64':(3,-1), '65':(3,0), '66':(3,1), '67':(3,2), '68':(4,-5), '69':(4,-4), '70':(4,-3), '71':(4,-2), '72':(4,-1), '73':(4,0), '74':(4,1), '75':(5,-5), '76':(5,-4), '77':(5,-3), '78':(5,-2)}

    # Inverse of above
    hex2sta = {v:k for k, v in sta2hex.items()}

    return sta2hex, hex2sta


# Hard-coded edge stations on pixelized array (see backend notebook for more)
def edgeIdxs():
    idxs = [[0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [1, 3], [1, 9], [2, 2], [2, 9], [3, 1], [3, 9], [4, 0], [4, 9], [5, 0], [5, 9], [6, 0], [6, 8], [7, 0], [7, 7], [8, 0], [8, 4], [8, 5], [8, 6], [9, 0], [9, 1], [9, 2], [9, 3]]
    return np.asarray(idxs)


# Standard IceTop quality cut
def qualityCut(x, y, qmax=6, edgeCut=True, reco='plane', zmax=40,
               verbose=False):

    # Ensure it's being run on unmerged data
    if x.shape[-1] != 4:
        print('Quality cut designed to be run on full (Q1, Q2, t1, t2) x data')
        raise

    cut = np.ones(x.shape[0], dtype=bool)
    out = {'Starting events':cut.sum()}

    if qmax != None:
        max_charges = x[...,0:2].max(axis=(1,2,3))
        cut *= rLog(max_charges) > qmax
        out['Events after charge cut'] = cut.sum()

    if zmax != None:
        r = '{}_dir'.format(reco)
        if reco == None:
            r = 'plane_dir'
        zenith, _ = np.transpose(y[r]).astype('float')
        zenith = np.pi - zenith
        cut *= zenith <= zmax * np.pi/180
        out['Events after zenith cut'] = cut.sum()

    if edgeCut:
        cutArray = np.zeros((10,10), dtype=bool)
        edgeList = edgeIdxs()
        for i, j in edgeList:
            cutArray[i,j] = 1
        cutArray = np.where(cutArray.reshape(-1) == 1)[0]
        x_temp = dataPrep(x, y, q='max', normed=False)
        max_idxs = x_temp[...,0].reshape(x_temp.shape[0],-1).argmax(axis=1)
        for i in cutArray:
            cut *= np.logical_not(max_idxs == i)
        out['Events after edge cut:'] = cut.sum()

    if verbose:
        for key, n in out.items():
            print('{}: {}'.format(key, n))

    return cut


def load_data(fileList, infill=False):

    d = defaultdict(list)
    singleKeys = ['gain','position'] # Keys not tied to events
    domKeys = ['charge','time'] # Keys tied to DOMs

    # Concatenate all information tied to events
    for i, f in enumerate(fileList):
        if sys.version_info.major == 3:
            d_i = np.load(f, allow_pickle=True)
        else:
            d_i = np.load(f)
        d_i = d_i.item()
        for key in d_i.keys():
            if key not in singleKeys:
                d[key] += d_i[key]

    # Only copy in the detector information once
    for key in singleKeys:
        d[key] = d_i[key]

    # Remove infill array
    if not infill:
        for key in domKeys:
            d[key] = [{k:v for k, v in domSeries.items() if int(k[:2])<79}
                     for domSeries in d[key]]

    return d


""" Convert dictionary format to multi-layered array for use with CNN """
def dict2mat(d):

    sta2hex, hex2sta = hexSquare()
    stationList = sorted(sta2hex.keys())
    squareY, squareX = np.transpose([sta2hex[s] for s in stationList])
    tank1s = ['61','62']
    tank2s = ['63','64']

    nevents = len(d['charge'])
    nx = squareX.max() - squareX.min() + 1
    ny = squareY.max() - squareY.min() + 1
    depth = 4   # Q1, Q2, T1, T2
    data = np.zeros((nevents, ny, nx, depth))

    # Layers #1 and 2: charge at each tank
    for i, charges in enumerate(d['charge']):
        for dom, charge in charges.items():
            y_i = sta2hex[dom[:2]][0] - squareY.min()
            x_i = sta2hex[dom[:2]][1] - squareX.min()
            if dom[2:] in tank1s:
                data[i][y_i][x_i][0] += modLog(charge)
            if dom[2:] in tank2s:
                data[i][y_i][x_i][1] += modLog(charge)

    # Layers #3 and 4: time at each tank
    for i, times in enumerate(d['time']):
        for dom, time in times.items():
            y_i = sta2hex[dom[:2]][0] - squareY.min()
            x_i = sta2hex[dom[:2]][1] - squareX.min()
            if dom[2:] in tank1s:
                data[i][y_i][x_i][2] += time
            if dom[2:] in tank2s:
                data[i][y_i][x_i][3] += time

    return data


def load_preprocessed(filePath, mode, nanCut=True, comp=['p','f']):

    if mode not in ['assessment', 'train', None]:
        print('Invalid mode choice')
        raise

    xfiles = sorted(glob('%s/x_*.npy' % filePath))
    yfiles = sorted(glob('%s/y_*.npy' % filePath))

    compDict = {'p':'12360','h':'12630','o':'12631','f':'12362'}
    simList = [compDict[c] for c in comp]

    xfiles = [f for f in xfiles if any([sim in f for sim in simList])]
    yfiles = [f for f in yfiles if any([sim in f for sim in simList])]
    #xfiles = [f for f in xfiles if '12630' not in f and '12631' not in f]
    #yfiles = [f for f in yfiles if '12630' not in f and '12631' not in f]

    x = []
    y = defaultdict(list)
    for i, xf in enumerate(xfiles):
        x_i = np.load(xf)
        x += [r for r in x_i]
        # Loading code specific to python version
        if sys.version_info.major == 3:
            y_i = np.load(yfiles[i], allow_pickle=True, encoding='latin1')
        else:
            y_i = np.load(yfiles[i])
        y_i = y_i.item()
        for key in y_i.keys():
            y[key] += [r for r in y_i[key]]

    x = np.asarray(x).reshape(-1,10,10,4)
    for key in y.keys():
        y[key] = np.asarray(y[key])

    # Remove events with NaN's (reconsider later?)
    if nanCut:
        tempValue = x.sum(axis=(1,2,3))
        nan = (tempValue == tempValue)
        x = x[nan]
        for key in y.keys():
            y[key] = y[key][nan]
        loss = (len(nan)-len(x)) / len(nan) * 100
        print("Percentage of events with a NaN: %.02f" % loss)
    else:
        print("Warning: NaNs included in data!")

    # Goal: Isolate a set of events to be used for assessment.
    # No model should ever be trained on these events.
    # Temporary approach, just isolate 10% of total sample
    if mode == None:
        return x, y

    # Create the same randomized array each time
    np.random.seed(1148)
    cut_idxs = []

    cut = (np.random.uniform(size=x.shape[0]) > 0.1).astype(bool)
    if mode == 'assessment':
        cut = np.logical_not(cut)

    return x[cut], {key:y[key][cut] for key in y.keys()}


""" Catch-all function designed to prepare data in a variety of common ways. 
 - Can combine charge or time layers to one value per station
 - Can normalize charge, time, or both
"""
def dataPrep(x, y, q=None, t=None, normed=False, reco=None, cosz=False):
             #recocut=False):

    qtDict = {'q':q, 't':t}
    new_dim = 4
    for k, mergeType in qtDict.items():
        # False = remove the layer
        if mergeType == False:
            new_dim -= 2
        # None indicates do nothing to the layers, everything else merges
        elif mergeType != None:
            new_dim -= 1

    out_shape = (x.shape[0], x.shape[1], x.shape[2], new_dim)
    out_array = np.zeros(out_shape)

    ## Suggestions for alternate methods of merging layers
    ## - use charge associated with earliest arrival time 
    ## - use time associated with maximum charge

    # Setup for merging tanks to stations in common ways
    for k, mergeType in qtDict.items():

        if mergeType == False:
            continue

        if k == 'q':
            out_idx = 0
            a1, a2 = x[...,:2].transpose(3,0,1,2)
            if mergeType == None:
                out_array[...,:2] = x[...,:2]
                continue
        if k == 't':
            out_idx = -1
            a1, a2 = x[...,-2:].transpose(3,0,1,2)
            if mergeType == None:
                out_array[...,-2:] = x[...,-2:]
                continue

        # Need to convert zeros for mean/min/max functions
        # Note: if there are np.nan's in data and you run the mean function,
        # they will be converted to zeros upon exit!
        ignoreValues = {'mean':np.nan, 'min':np.inf, 'max':-np.inf}
        if mergeType in ignoreValues.keys():
            a1[a1==0] = ignoreValues[mergeType]
            a2[a2==0] = ignoreValues[mergeType]

        # Options for combining: mean, sum, product, min, max
        if mergeType not in ['mean','sum','product','min','max']:
            print('Unexpected argument for %s' % k)
            raise

        # Mean requires special conversion back (since np.nan!=np.nan)
        if mergeType == 'mean':
            new_out = np.nanmean([a1,a2], axis=0)
            a1[a1!=a1] = 0
            a2[a2!=a2] = 0
            new_out[new_out != new_out] = 0
        if mergeType == 'sum':
            new_out = modLog(rLog(a1) + rLog(a2))
        if mergeType == 'product':
            new_out = a1 + a2   # sum of logs = product
        if mergeType == 'min':
            new_out = np.min([a1,a2], axis=0)
        if mergeType == 'max':
            new_out = np.max([a1,a2], axis=0)

        # Convert zeros back
        if mergeType in ['min','max']:
            a1[a1==ignoreValues[mergeType]] = 0
            a2[a2==ignoreValues[mergeType]] = 0
            new_out[new_out == ignoreValues[mergeType]] = 0

        out_array[...,out_idx] = new_out

    # Normalization
    if normed:
        # Find the maximum value for each layer
        ## NOTE: separate layers for charge/time will be normed separately!
        maxValues = out_array.max(axis=(0,1,2), keepdims=True)
        # Avoid instances where no data is in a layer for a given event
        maxValues[maxValues==0] = 1
        # Normalize
        out_array /= maxValues

    # Keep NaN's in with reconstruction so we can tell which events to ignore
    if reco != None:
        th, _ = y['{}_dir'.format(reco)].transpose()
        th = th.astype('float')
        th = np.pi - th
        if cosz:
            th = np.cos(th)
        if normed and not cosz:
            th /= np.nanmax(th)
        out_array = [out_array, th]

    return out_array


def load_mc(filePath):

    mcfiles = sorted(glob('%s/sim_*_mc.npy' % filePath))

    mc = defaultdict(list)
    for i, f in enumerate(mcfiles):
        if sys.version_info.major == 3:
            mc_i = np.load(f, allow_pickle=True)
        else:
            mc_i = np.load(f)
        mc_i = mc_i.item()
        for key in mc_i.keys():
            mc[key] += mc_i[key]

    for key in mc.keys():
        mc[key] = np.asarray(mc[key])

    return mc


def nameModel(prep, prefix=''):
    outstr = prefix
    for key, value in prep.items():
        outstr = '{}_{}.{}'.format(outstr, key, value)
    return outstr


def name2prep(name):
    prep = {}
    args = [a.split('.') for a in name.split('_') if len(a.split('.'))==2]
    for key, value in args:
        try: prep[key] = ast.literal_eval(value)
        except ValueError:
            prep[key] = value
    return prep








