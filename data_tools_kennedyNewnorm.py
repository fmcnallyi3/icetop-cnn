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
    # no rotations
    alpha= {'01':(-4,-1), '02':(-4,0), '03':(-4,1), '04':(-4,2), '05':(-4,3), '06':(-4,4), '07':(-3,-2), '08':(-3,-1), '09':(-3,0), '10':(-3,1), '11':(-3,2), '12':(-3,3), '13':(-3,4), '14':(-2,-3), '15':(-2,-2), '16':(-2,-1), '17':(-2,0), '18':(-2,1), '19':(-2,2), '20':(-2,3), '21':(-2,4), '22':(-1,-4), '23':(-1,-3), '24':(-1,-2), '25':(-1,-1), '26':(-1,0), '27':(-1,1), '28':(-1,2), '29':(-1,3), '30':(-1,4), '31':(0,-5), '32':(0,-4), '33':(0,-3), '34':(0,-2), '35':(0,-1), '36':(0,0), '37':(0,1), '38':(0,2), '39':(0,3), '40':(0,4), '41':(1,-5), '42':(1,-4), '43':(1,-3), '44':(1,-2), '45':(1,-1), '46':(1,0), '47':(1,1), '48':(1,2), '49':(1,3), '50':(1,4), '51':(2,-5), '52':(2,-4), '53':(2,-3), '54':(2,-2), '55':(2,-1), '56':(2,0), '57':(2,1), '58':(2,2), '59':(2,3), '60':(3,-5), '61':(3,-4), '62':(3,-3), '63':(3,-2), '64':(3,-1), '65':(3,0), '66':(3,1), '67':(3,2), '68':(4,-5), '69':(4,-4), '70':(4,-3), '71':(4,-2), '72':(4,-1), '73':(4,0), '74':(4,1), '75':(5,-5), '76':(5,-4), '77':(5,-3), '78':(5,-2)}

    #2 rotations
    x,y=0,4
    w,z=-1,4
    a,b=-2,4
    c,d=-3,4
    e,f=-4,4
    g,h=-5,4
    j,k=-5,3
    m,n=-5,2
    o,p=-5,1
    r,s=-5,0
    beta = {'01':(-4,-1), '02':(-4,0), '03':(-4,1), '04':(-4,2), '05':(-4,3), '06':(-4,4), '07':(-3,-2), '08':(-3,-1), '09':(-3,0), '10':(-3,1), '11':(-3,2), '12':(-3,3), '13':(-3,4), '14':(-2,-3), '15':(-2,-2), '16':(-2,-1), '17':(-2,0), '18':(-2,1), '19':(-2,2), '20':(-2,3), '21':(-2,4), '22':(-1,-4), '23':(-1,-3), '24':(-1,-2), '25':(-1,-1), '26':(-1,0), '27':(-1,1), '28':(-1,2), '29':(-1,3), '30':(-1,4), '31':(0,-5), '32':(0,-4), '33':(0,-3), '34':(0,-2), '35':(0,-1), '36':(0,0), '37':(0,1), '38':(0,2), '39':(0,3), '40':(0,4), '41':(1,-5), '42':(1,-4), '43':(1,-3), '44':(1,-2), '45':(1,-1), '46':(1,0), '47':(1,1), '48':(1,2), '49':(1,3), '50':(1,4), '51':(2,-5), '52':(2,-4), '53':(2,-3), '54':(2,-2), '55':(2,-1), '56':(2,0), '57':(2,1), '58':(2,2), '59':(2,3), '60':(3,-5), '61':(3,-4), '62':(3,-3), '63':(3,-2), '64':(3,-1), '65':(3,0), '66':(3,1), '67':(3,2), '68':(4,-5), '69':(4,-4), '70':(4,-3), '71':(4,-2), '72':(4,-1), '73':(4,0), '74':(4,1), '75':(5,-5), '76':(5,-4), '77':(5,-3), '78':(5,-2)}
    for key in alpha:
        if int(key)<=6:
            beta[key]=(x, y)
            x+=1
            y-=1
        if int(key)>6 and int(key)<=13:
            beta[key]=(w, z)
            w+=1
            z-=1
        if int(key)>13 and int(key)<=21:
            beta[key]=(a, b)
            a+=1
            b-=1
        if int(key)> 21 and int(key)<=30:
            beta[key]=(c, d)
            c+=1
            d-=1
        if int(key)>30 and int(key)<=40:
            beta[key]=(e, f)
            e+=1
            f-=1
        if int(key)>40 and int(key)<=50:
            beta[key]=(g, h)
            g+=1
            h-=1
        if int(key)>50 and int(key)<=59:
            beta[key]=(j, k)
            j+=1
            k-=1
        if int(key)>59 and int(key)<=67:
            beta[key]=(m, n)
            m+=1
            n-=1
        if int(key)>67 and int(key)<=74:
            beta[key]=(o, p)
            o+=1
            p-=1
        if int(key)>74 and int(key)<=78:
            beta[key]=(r, s)
            r+=1
            s-=1 

    # 3 rotations      
    gamma = {'01':(-4,-1), '02':(-4,0), '03':(-4,1), '04':(-4,2), '05':(-4,3), '06':(-4,4), '07':(-3,-2), '08':(-3,-1), '09':(-3,0), '10':(-3,1), '11':(-3,2), '12':(-3,3), '13':(-3,4), '14':(-2,-3), '15':(-2,-2), '16':(-2,-1), '17':(-2,0), '18':(-2,1), '19':(-2,2), '20':(-2,3), '21':(-2,4), '22':(-1,-4), '23':(-1,-3), '24':(-1,-2), '25':(-1,-1), '26':(-1,0), '27':(-1,1), '28':(-1,2), '29':(-1,3), '30':(-1,4), '31':(0,-5), '32':(0,-4), '33':(0,-3), '34':(0,-2), '35':(0,-1), '36':(0,0), '37':(0,1), '38':(0,2), '39':(0,3), '40':(0,4), '41':(1,-5), '42':(1,-4), '43':(1,-3), '44':(1,-2), '45':(1,-1), '46':(1,0), '47':(1,1), '48':(1,2), '49':(1,3), '50':(1,4), '51':(2,-5), '52':(2,-4), '53':(2,-3), '54':(2,-2), '55':(2,-1), '56':(2,0), '57':(2,1), '58':(2,2), '59':(2,3), '60':(3,-5), '61':(3,-4), '62':(3,-3), '63':(3,-2), '64':(3,-1), '65':(3,0), '66':(3,1), '67':(3,2), '68':(4,-5), '69':(4,-4), '70':(4,-3), '71':(4,-2), '72':(4,-1), '73':(4,0), '74':(4,1), '75':(5,-5), '76':(5,-4), '77':(5,-3), '78':(5,-2)}
    for key in gamma:
        x = (gamma[key][0])*-1
        y = (gamma[key][1])*-1
        gamma[key]=(x, y) 

    sta2hex = [alpha, beta, gamma]
    # Inverse of above
    hex2sta = [{v:k for k, v in i.items()} for i in sta2hex]

    return sta2hex, hex2sta


# Hard-coded edge stations on pixelized array (see backend notebook for more)
def edgeIdxs():
    idxs = [[0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [1, 3], [1, 9], [2, 2], [2, 9], [3, 1], [3, 9], [4, 0], [4, 9], [5, 0], [5, 9], [6, 0], [6, 8], [7, 0], [7, 7], [8, 0], [8, 4], [8, 5], [8, 6], [9, 0], [9, 1], [9, 2], [9, 3]]
    return np.asarray(idxs)


# Get cut to display on plots
def getCut(cut_str, x, y, p, recoE, key):
    if cut_str == 'No Cut':
        cut = np.array([True for _ in x])
        if len(x) != len(recoE[key]):
            temp_cut=np.zeros(x.shape[0])
            ind=0
            for event in cut:
                for num in range(4):
                    temp_cut=event
                    ind+=1
            cut=temp_cut
    elif cut_str == 'Quality Cut':
        if len(x) == len(recoE[key]):
            cut = qualityCut(x, y, reco=p[key]['reco'])
        if len(x) != len(recoE[key]):
            cut = qualityCut(x, y, reco=p[key]['reco'], mult=True)
    else:
        print('Bad cut!')
    cut *= ~np.isnan(recoE[key])
    return cut

# Standard IceTop quality cut
def qualityCut(x, y, qmax=6, edgeCut=True, reco='plane', zmax=40,
               verbose=False, mult=False):

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

    if mult:
        temp_cut=np.ones(x.shape[0]*4, dtype=bool)
        ind=0
        for event in cut:
            for num in range(4):
                temp_cut[ind]=event
                ind+=1
        cut=temp_cut
    
    if zmax != None:
        r = '{}_dir'.format(reco)
        #if reco == None:
        #    r = 'plane_dir'
        zenith, _ = np.transpose(y[r])
        zenith = zenith.astype('float')
        zenith = np.pi - zenith
        cut *= zenith <= zmax * np.pi/180
        out['Events after zenith cut'] = cut.sum()

    if verbose:
        for key, n in out.items():
            print('{}: {}'.format(key, n))
            
    return cut


def load_data(fileList, infill=False):

    d = defaultdict(list)
    singleKeys = ['gain','position'] # Keys not tied to events
    domKeys = ['charge','time'] # Keys tied to DOMs

    # Concatenate all information tied to events
    for f in fileList:
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
    stationList = sorted(sta2hex[0].keys()) #get the keys
    squareY, squareX = np.transpose([sta2hex[0][s] for s in stationList]) #separate the coords into lists
    tank1s = ['61','62']
    tank2s = ['63','64']
    nevents = len(d['charge'])
    nx = squareX.max() - squareX.min() + 1 #length of side x
    ny = squareY.max() - squareY.min() + 1 #length of side y
    depth = 4   # Q1, Q2, T1, T2
    data = np.zeros((nevents, ny, nx, depth)) #create an empty array with the needed dimensions

    # record the activated doms into the empty array            
    for i in range(nevents):
        randHex=sta2hex[np.random.randint(3)]
        for dom in d['charge'][i]:
            y_i=randHex[dom[:2]][0] - squareY.min()
            x_i=randHex[dom[:2]][1] - squareX.min()
            if y_i<ny and x_i<nx: #filters rotated doms that don't fall within 10x10 structure (first filter)
                charge=d['charge'][i][dom]
                if dom[2:] in tank1s:
                    data[i][y_i][x_i][0] += modLog(charge)
                if dom[2:] in tank2s:
                    data[i][y_i][x_i][1] +=modLog(charge)
        for dom in d['time'][i]:
            y_i = randHex[dom[:2]][0] - squareY.min()
            x_i = randHex[dom[:2]][1] - squareX.min()
            if y_i<ny and x_i<nx:
                time=d['time'][i][dom]
                if dom[2:] in tank1s:
                    data[i][y_i][x_i][2] += time
                if dom[2:] in tank2s:
                    data[i][y_i][x_i][3] += time
    
    return np.asarray(data).reshape(-1,10,10,4)


def load_preprocessed(filePath, mode, nanCut=True, comp=['p','f']):

    # Make sure user selects a valid mode
    if mode not in ['assessment', 'train', None]:
        print('Invalid mode choice')
        raise
    
    # Load valid x and y files
    xfiles = sorted(glob('%s/x_*.npy' % filePath))
    yfiles = sorted(glob('%s/y_*.npy' % filePath))

    # Load only proton and iron files
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
        tempValue = x.sum(axis=(1,2,3)) # sum all data in each event
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

    cut = (np.random.uniform(size=x.shape[0]) > 0.1).astype(bool)
    if mode == 'assessment':
        cut = np.logical_not(cut)

    return x[cut], {key:y[key][cut] for key in y.keys()}


""" Catch-all function designed to prepare data in a variety of common ways. 
 - Can combine charge or time layers to one value per station
 - Can normalize charge, time, or both
"""
def dataPrep(x, y, q=None, t=None, normed=False, reco=None, cosz=False, rot=False, rotx=False, t_shift=False):
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
        #maxValues = out_array.max(axis=(0,1,2), keepdims=True)
        # Avoid instances where no data is in a layer for a given event
        #maxValues[maxValues==0] = 1
        # Normalize
        #out_array /= maxValues

        if q != False: 
            maxValuesQ1 = out_array[...,0].max(axis=(0,1,2), keepdims=True)
            maxValuesQ2 = out_array[...,1].max(axis=(0,1,2), keepdims=True)
        #choose the highest maximum from one of the two layers       
            if maxValuesQ2 > maxValuesQ1:
                maxValue1 = maxValuesQ2 
            else: 
                maxValue1 = maxValuesQ1
        #Avoid instances where no data is in a layer for a given event        
            maxValue1[maxValue1==0] = 1  
        # Normalize
            out_array[...,:2]/=maxValue1    
        
        
        if t != False:
        #shift time to eliminate gap 
            if t_shift != False: 
                timelayer1 = np.array(out_array[...,2])
                timelayer2 = np.array(out_array[...,3])
                bothtimelayers = np.array(out_array[...,-2:])
            
                minval1 = np.min(timelayer1[np.nonzero(timelayer1)])
                minval2 = np.min(timelayer2[np.nonzero(timelayer2)])

                if minval1 > minval2:
                    minval = minval1
                else: 
                    minval = minval2
    
                bothtimelayers[bothtimelayers==0] = minval 
                shiftedtime = bothtimelayers - minval
                out_array[...,-2:] = shiftedtime 
            

            #take the shifted time and normalize     
            maxValuesT1 = out_array[...,2].max(axis=(0,1,2), keepdims=True) 
            maxValuesT2 = out_array[...,3].max(axis=(0,1,2), keepdims=True)

            #choose the highest maximum from one of the two layers 
            if maxValuesT1 > maxValuesT2:
                maxValue2 = maxValuesT1
            else: 
                maxValue2 = maxValuesT2
            #Avoid instances where no data is in a layer for a given event
            maxValue2[maxValue2==0] = 1
            # Normalize
            out_array[...,-2:]/= maxValue2
    
    # Randomly rotate each event by 0,90,180, or 270 degrees
    if rot:
        #0 degrees
        alpha={(0, 4): (0, 4), (0, 5): (0, 5), (0, 6): (0, 6), (0, 7): (0, 7), (0, 8): (0, 8), (0, 9): (0, 9), (1, 3): (1, 3), (1, 4): (1, 4), (1, 5): (1, 5), (1, 6): (1, 6), (1, 7): (1, 7), (1, 8): (1, 8), (1, 9): (1, 9), (2, 2): (2, 2), (2, 3): (2, 3), (2, 4): (2, 4), (2, 5): (2, 5), (2, 6): (2, 6), (2, 7): (2, 7), (2, 8): (2, 8), (2, 9): (2, 9), (3, 1): (3, 1), (3, 2): (3, 2), (3, 3): (3, 3), (3, 4): (3, 4), (3, 5): (3, 5), (3, 6): (3, 6), (3, 7): (3, 7), (3, 8): (3, 8), (3, 9): (3, 9), (4, 0): (4, 0), (4, 1): (4, 1), (4, 2): (4, 2), (4, 3): (4, 3), (4, 4): (4, 4), (4, 5): (4, 5), (4, 6): (4, 6), (4, 7): (4, 7), (4, 8): (4, 8), (4, 9): (4, 9), (5, 0): (5, 0), (5, 1): (5, 1), (5, 2): (5, 2), (5, 3): (5, 3), (5, 4): (5, 4), (5, 5): (5, 5), (5, 6): (5, 6), (5, 7): (5, 7), (5, 8): (5, 8), (5, 9): (5, 9), (6, 0): (6, 0), (6, 1): (6, 1), (6, 2): (6, 2), (6, 3): (6, 3), (6, 4): (6, 4), (6, 5): (6, 5), (6, 6): (6, 6), (6, 7): (6, 7), (6, 8): (6, 8), (7, 0): (7, 0), (7, 1): (7, 1), (7, 2): (7, 2), (7, 3): (7, 3), (7, 4): (7, 4), (7, 5): (7, 5), (7, 6): (7, 6), (7, 7): (7, 7), (8, 0): (8, 0), (8, 1): (8, 1), (8, 2): (8, 2), (8, 3): (8, 3), (8, 4): (8, 4), (8, 5): (8, 5), (8, 6): (8, 6), (9, 0): (9, 0), (9, 1): (9, 1), (9, 2): (9, 2), (9, 3): (9, 3)}
        #60 degrees
        beta={(0, 4): (0, 9), (0, 5): (1, 9), (0, 6): (2, 9), (0, 7): (3, 9), (0, 8): (4, 9), (0, 9): (5, 9), (1, 3): (0, 8), (1, 4): (1, 8), (1, 5): (2, 8), (1, 6): (3, 8), (1, 7): (4, 8), (1, 8): (5, 8), (1, 9): (6, 8), (2, 2): (0, 7), (2, 3): (1, 7), (2, 4): (2, 7), (2, 5): (3, 7), (2, 6): (4, 7), (2, 7): (5, 7), (2, 8): (6, 7), (2, 9): (7, 7), (3, 1): (0, 6), (3, 2): (1, 6), (3, 3): (2, 6), (3, 4): (3, 6), (3, 5): (4, 6), (3, 6): (5, 6), (3, 7): (6, 6), (3, 8): (7, 6), (3, 9): (8, 6), (4, 0): (0, 5), (4, 1): (1, 5), (4, 2): (2, 5), (4, 3): (3, 5), (4, 4): (4, 5), (4, 5): (5, 5), (4, 6): (6, 5), (4, 7): (7, 5), (4, 8): (8, 5), (4, 9): (9, 5), (5, 0): (0, 4), (5, 1): (1, 4), (5, 2): (2, 4), (5, 3): (3, 4), (5, 4): (4, 4), (5, 5): (5, 4), (5, 6): (6, 4), (5, 7): (7, 4), (5, 8): (8, 4), (5, 9): (9, 4), (6, 0): (1, 3), (6, 1): (2, 3), (6, 2): (3, 3), (6, 3): (4, 3), (6, 4): (5, 3), (6, 5): (6, 3), (6, 6): (7, 3), (6, 7): (8, 3), (6, 8): (9, 3), (7, 0): (2, 2), (7, 1): (3, 2), (7, 2): (4, 2), (7, 3): (5, 2), (7, 4): (6, 2), (7, 5): (7, 2), (7, 6): (8, 2), (7, 7): (9, 2), (8, 0): (3, 1), (8, 1): (4, 1), (8, 2): (5, 1), (8, 3): (6, 1), (8, 4): (7, 1), (8, 5): (8, 1), (8, 6): (9, 1), (9, 0): (4, 0), (9, 1): (5, 0), (9, 2): (6, 0), (9, 3): (7, 0)}
        #120 degrees
        gamma={(0, 4): (5, 9), (0, 5): (6, 8), (0, 6): (7, 7), (0, 7): (8, 6), (0, 8): (9, 5), (0, 9): (10, 4), (1, 3): (4, 9), (1, 4): (5, 8), (1, 5): (6, 7), (1, 6): (7, 6), (1, 7): (8, 5), (1, 8): (9, 4), (1, 9): (10, 3), (2, 2): (3, 9), (2, 3): (4, 8), (2, 4): (5, 7), (2, 5): (6, 6), (2, 6): (7, 5), (2, 7): (8, 4), (2, 8): (9, 3), (2, 9): (10, 2), (3, 1): (2, 9), (3, 2): (3, 8), (3, 3): (4, 7), (3, 4): (5, 6), (3, 5): (6, 5), (3, 6): (7, 4), (3, 7): (8, 3), (3, 8): (9, 2), (3, 9): (10, 1), (4, 0): (1, 9), (4, 1): (2, 8), (4, 2): (3, 7), (4, 3): (4, 6), (4, 4): (5, 5), (4, 5): (6, 4), (4, 6): (7, 3), (4, 7): (8, 2), (4, 8): (9, 1), (4, 9): (10, 0), (5, 0): (0, 9), (5, 1): (1, 8), (5, 2): (2, 7), (5, 3): (3, 6), (5, 4): (4, 5), (5, 5): (5, 4), (5, 6): (6, 3), (5, 7): (7, 2), (5, 8): (8, 1), (5, 9): (9, 0), (6, 0): (0, 8), (6, 1): (1, 7), (6, 2): (2, 6), (6, 3): (3, 5), (6, 4): (4, 4), (6, 5): (5, 3), (6, 6): (6, 2), (6, 7): (7, 1), (6, 8): (8, 0), (7, 0): (0, 7), (7, 1): (1, 6), (7, 2): (2, 5), (7, 3): (3, 4), (7, 4): (4, 3), (7, 5): (5, 2), (7, 6): (6, 1), (7, 7): (7, 0), (8, 0): (0, 6), (8, 1): (1, 5), (8, 2): (2, 4), (8, 3): (3, 3), (8, 4): (4, 2), (8, 5): (5, 1), (8, 6): (6, 0), (9, 0): (0, 5), (9, 1): (1, 4), (9, 2): (2, 3), (9, 3): (3, 2)}
        #180 degrees
        delta={(0, 4): (8, 6), (0, 5): (8, 5), (0, 6): (8, 4), (0, 7): (8, 3), (0, 8): (8, 2), (0, 9): (8, 1), (1, 3): (7, 7), (1, 4): (7, 6), (1, 5): (7, 5), (1, 6): (7, 4), (1, 7): (7, 3), (1, 8): (7, 2), (1, 9): (7, 1), (2, 2): (6, 8), (2, 3): (6, 7), (2, 4): (6, 6), (2, 5): (6, 5), (2, 6): (6, 4), (2, 7): (6, 3), (2, 8): (6, 2), (2, 9): (6, 1), (3, 1): (5, 9), (3, 2): (5, 8), (3, 3): (5, 7), (3, 4): (5, 6), (3, 5): (5, 5), (3, 6): (5, 4), (3, 7): (5, 3), (3, 8): (5, 2), (3, 9): (5, 1), (4, 0): (4, 10), (4, 1): (4, 9), (4, 2): (4, 8), (4, 3): (4, 7), (4, 4): (4, 6), (4, 5): (4, 5), (4, 6): (4, 4), (4, 7): (4, 3), (4, 8): (4, 2), (4, 9): (4, 1), (5, 0): (3, 10), (5, 1): (3, 9), (5, 2): (3, 8), (5, 3): (3, 7), (5, 4): (3, 6), (5, 5): (3, 5), (5, 6): (3, 4), (5, 7): (3, 3), (5, 8): (3, 2), (5, 9): (3, 1), (6, 0): (2, 10), (6, 1): (2, 9), (6, 2): (2, 8), (6, 3): (2, 7), (6, 4): (2, 6), (6, 5): (2, 5), (6, 6): (2, 4), (6, 7): (2, 3), (6, 8): (2, 2), (7, 0): (1, 10), (7, 1): (1, 9), (7, 2): (1, 8), (7, 3): (1, 7), (7, 4): (1, 6), (7, 5): (1, 5), (7, 6): (1, 4), (7, 7): (1, 3), (8, 0): (0, 10), (8, 1): (0, 9), (8, 2): (0, 8), (8, 3): (0, 7), (8, 4): (0, 6), (8, 5): (0, 5), (8, 6): (0, 4), (9, 0): (-1, 10), (9, 1): (-1, 9), (9, 2): (-1, 8), (9, 3): (-1, 7)}

        for int in range(out_array.shape[0]):
            temp=np.zeros((10,10,out_array.shape[-1]))
            rots=np.random.choice([alpha,beta,gamma,delta])
            for i,c in np.ndenumerate(out_array[int]):
                if c>0:
                    new_coords=(rots[i[:2]][0],rots[i[:2]][1],i[-1])
                    if new_coords[:2] in alpha.values():
                        temp[new_coords]=c
            out_array[int]=temp
    # Create rot variations of every event
    if rot=='x':
        #0 degrees
        alpha={(0, 4): (0, 4), (0, 5): (0, 5), (0, 6): (0, 6), (0, 7): (0, 7), (0, 8): (0, 8), (0, 9): (0, 9), (1, 3): (1, 3), (1, 4): (1, 4), (1, 5): (1, 5), (1, 6): (1, 6), (1, 7): (1, 7), (1, 8): (1, 8), (1, 9): (1, 9), (2, 2): (2, 2), (2, 3): (2, 3), (2, 4): (2, 4), (2, 5): (2, 5), (2, 6): (2, 6), (2, 7): (2, 7), (2, 8): (2, 8), (2, 9): (2, 9), (3, 1): (3, 1), (3, 2): (3, 2), (3, 3): (3, 3), (3, 4): (3, 4), (3, 5): (3, 5), (3, 6): (3, 6), (3, 7): (3, 7), (3, 8): (3, 8), (3, 9): (3, 9), (4, 0): (4, 0), (4, 1): (4, 1), (4, 2): (4, 2), (4, 3): (4, 3), (4, 4): (4, 4), (4, 5): (4, 5), (4, 6): (4, 6), (4, 7): (4, 7), (4, 8): (4, 8), (4, 9): (4, 9), (5, 0): (5, 0), (5, 1): (5, 1), (5, 2): (5, 2), (5, 3): (5, 3), (5, 4): (5, 4), (5, 5): (5, 5), (5, 6): (5, 6), (5, 7): (5, 7), (5, 8): (5, 8), (5, 9): (5, 9), (6, 0): (6, 0), (6, 1): (6, 1), (6, 2): (6, 2), (6, 3): (6, 3), (6, 4): (6, 4), (6, 5): (6, 5), (6, 6): (6, 6), (6, 7): (6, 7), (6, 8): (6, 8), (7, 0): (7, 0), (7, 1): (7, 1), (7, 2): (7, 2), (7, 3): (7, 3), (7, 4): (7, 4), (7, 5): (7, 5), (7, 6): (7, 6), (7, 7): (7, 7), (8, 0): (8, 0), (8, 1): (8, 1), (8, 2): (8, 2), (8, 3): (8, 3), (8, 4): (8, 4), (8, 5): (8, 5), (8, 6): (8, 6), (9, 0): (9, 0), (9, 1): (9, 1), (9, 2): (9, 2), (9, 3): (9, 3)}
        #60 degrees
        beta={(0, 4): (0, 9), (0, 5): (1, 9), (0, 6): (2, 9), (0, 7): (3, 9), (0, 8): (4, 9), (0, 9): (5, 9), (1, 3): (0, 8), (1, 4): (1, 8), (1, 5): (2, 8), (1, 6): (3, 8), (1, 7): (4, 8), (1, 8): (5, 8), (1, 9): (6, 8), (2, 2): (0, 7), (2, 3): (1, 7), (2, 4): (2, 7), (2, 5): (3, 7), (2, 6): (4, 7), (2, 7): (5, 7), (2, 8): (6, 7), (2, 9): (7, 7), (3, 1): (0, 6), (3, 2): (1, 6), (3, 3): (2, 6), (3, 4): (3, 6), (3, 5): (4, 6), (3, 6): (5, 6), (3, 7): (6, 6), (3, 8): (7, 6), (3, 9): (8, 6), (4, 0): (0, 5), (4, 1): (1, 5), (4, 2): (2, 5), (4, 3): (3, 5), (4, 4): (4, 5), (4, 5): (5, 5), (4, 6): (6, 5), (4, 7): (7, 5), (4, 8): (8, 5), (4, 9): (9, 5), (5, 0): (0, 4), (5, 1): (1, 4), (5, 2): (2, 4), (5, 3): (3, 4), (5, 4): (4, 4), (5, 5): (5, 4), (5, 6): (6, 4), (5, 7): (7, 4), (5, 8): (8, 4), (5, 9): (9, 4), (6, 0): (1, 3), (6, 1): (2, 3), (6, 2): (3, 3), (6, 3): (4, 3), (6, 4): (5, 3), (6, 5): (6, 3), (6, 6): (7, 3), (6, 7): (8, 3), (6, 8): (9, 3), (7, 0): (2, 2), (7, 1): (3, 2), (7, 2): (4, 2), (7, 3): (5, 2), (7, 4): (6, 2), (7, 5): (7, 2), (7, 6): (8, 2), (7, 7): (9, 2), (8, 0): (3, 1), (8, 1): (4, 1), (8, 2): (5, 1), (8, 3): (6, 1), (8, 4): (7, 1), (8, 5): (8, 1), (8, 6): (9, 1), (9, 0): (4, 0), (9, 1): (5, 0), (9, 2): (6, 0), (9, 3): (7, 0)}

        #120 degrees
        gamma={(0, 4): (4, 8), (0, 5): (5, 7), (0, 6): (6, 6), (0, 7): (7, 5), (0, 8): (8, 4), (0, 9): (9, 3), (1, 3): (3, 8), (1, 4): (4, 7), (1, 5): (5, 6), (1, 6): (6, 5), (1, 7): (7, 4), (1, 8): (8, 3), (1, 9): (9, 2), (2, 2): (2, 8), (2, 3): (3, 7), (2, 4): (4, 6), (2, 5): (5, 5), (2, 6): (6, 4), (2, 7): (7, 3), (2, 8): (8, 2), (2, 9): (9, 1), (3, 1): (1, 8), (3, 2): (2, 7), (3, 3): (3, 6), (3, 4): (4, 5), (3, 5): (5, 4), (3, 6): (6, 3), (3, 7): (7, 2), (3, 8): (8, 1), (3, 9): (9, 0), (4, 0): (0, 8), (4, 1): (1, 7), (4, 2): (2, 6), (4, 3): (3, 5), (4, 4): (4, 4), (4, 5): (5, 3), (4, 6): (6, 2), (4, 7): (7, 1), (4, 8): (8, 0), (4, 9): (9, -1), (5, 0): (-1, 8), (5, 1): (0, 7), (5, 2): (1, 6), (5, 3): (2, 5), (5, 4): (3, 4), (5, 5): (4, 3), (5, 6): (5, 2), (5, 7): (6, 1), (5, 8): (7, 0), (5, 9): (8, -1), (6, 0): (-1, 7), (6, 1): (0, 6), (6, 2): (1, 5), (6, 3): (2, 4), (6, 4): (3, 3), (6, 5): (4, 2), (6, 6): (5, 1), (6, 7): (6, 0), (6, 8): (7, -1), (7, 0): (-1, 6), (7, 1): (0, 5), (7, 2): (1, 4), (7, 3): (2, 3), (7, 4): (3, 2), (7, 5): (4, 1), (7, 6): (5, 0), (7, 7): (6, -1), (8, 0): (-1, 5), (8, 1): (0, 4), (8, 2): (1, 3), (8, 3): (2, 2), (8, 4): (3, 1), (8, 5): (4, 0), (8, 6): (5, -1), (9, 0): (-1, 4), (9, 1): (0, 3), (9, 2): (1, 2), (9, 3): (2, 1)}

        #180 degrees
        delta={(4, 0): (4, 8), (5, 0): (3, 8), (6, 0): (2, 8), (7, 0): (1, 8), (8, 0): (0, 8), (9, 0): (-1, 8), (3, 1): (5, 7), (4, 1): (4, 7), (5, 1): (3, 7), (6, 1): (2, 7), (7, 1): (1, 7), (8, 1): (0, 7), (9, 1): (-1, 7), (2, 2): (6, 6), (3, 2): (5, 6), (4, 2): (4, 6), (5, 2): (3, 6), (6, 2): (2, 6), (7, 2): (1, 6), (8, 2): (0, 6), (9, 2): (-1, 6), (1, 3): (7, 5), (2, 3): (6, 5), (3, 3): (5, 5), (4, 3): (4, 5), (5, 3): (3, 5), (6, 3): (2, 5), (7, 3): (1, 5), (8, 3): (0, 5), (9, 3): (-1, 5), (0, 4): (8, 4), (1, 4): (7, 4), (2, 4): (6, 4), (3, 4): (5, 4), (4, 4): (4, 4), (5, 4): (3, 4), (6, 4): (2, 4), (7, 4): (1, 4), (8, 4): (0, 4), (9, 4): (-1, 4), (0, 5): (8, 3), (1, 5): (7, 3), (2, 5): (6, 3), (3, 5): (5, 3), (4, 5): (4, 3), (5, 5): (3, 3), (6, 5): (2, 3), (7, 5): (1, 3), (8, 5): (0, 3), (9, 5): (-1, 3), (0, 6): (8, 2), (1, 6): (7, 2), (2, 6): (6, 2), (3, 6): (5, 2), (4, 6): (4, 2), (5, 6): (3, 2), (6, 6): (2, 2), (7, 6): (1, 2), (8, 6): (0, 2), (0, 7): (8, 1), (1, 7): (7, 1), (2, 7): (6, 1), (3, 7): (5, 1), (4, 7): (4, 1), (5, 7): (3, 1), (6, 7): (2, 1), (7, 7): (1, 1), (0, 8): (8, 0), (1, 8): (7, 0), (2, 8): (6, 0), (3, 8): (5, 0), (4, 8): (4, 0), (5, 8): (3, 0), (6, 8): (2, 0), (0, 9): (8, -1), (1, 9): (7, -1), (2, 9): (6, -1), (3, 9): (5, -1)}

        #240 degrees
        epsilon={(0, 4): (9, 0), (0, 5): (8, 0), (0, 6): (7, 0), (0, 7): (6, 0), (0, 8): (5, 0), (0, 9): (4, 0), (1, 3): (9, 1), (1, 4): (8, 1), (1, 5): (7, 1), (1, 6): (6, 1), (1, 7): (5, 1), (1, 8): (4, 1), (1, 9): (3, 1), (2, 2): (9, 2), (2, 3): (8, 2), (2, 4): (7, 2), (2, 5): (6, 2), (2, 6): (5, 2), (2, 7): (4, 2), (2, 8): (3, 2), (2, 9): (2, 2), (3, 1): (9, 3), (3, 2): (8, 3), (3, 3): (7, 3), (3, 4): (6, 3), (3, 5): (5, 3), (3, 6): (4, 3), (3, 7): (3, 3), (3, 8): (2, 3), (3, 9): (1, 3), (4, 0): (8, 4), (4, 1): (7, 4), (4, 2): (6, 4), (4, 3): (5, 4), (4, 4): (4, 4), (4, 5): (3, 4), (4, 6): (2, 4), (4, 7): (1, 4), (4, 8): (0, 4), (4, 9): (-1, 4), (5, 0): (8, 5), (5, 1): (7, 5), (5, 2): (6, 5), (5, 3): (5, 5), (5, 4): (4, 5), (5, 5): (3, 5), (5, 6): (2, 5), (5, 7): (1, 5), (5, 8): (0, 5), (5, 9): (-1, 5), (6, 0): (8, 6), (6, 1): (7, 6), (6, 2): (6, 6), (6, 3): (5, 6), (6, 4): (4, 6), (6, 5): (3, 6), (6, 6): (2, 6), (6, 7): (1, 6), (6, 8): (0, 6), (7, 0): (7, 7), (7, 1): (6, 7), (7, 2): (5, 7), (7, 3): (4, 7), (7, 4): (3, 7), (7, 5): (2, 7), (7, 6): (1, 7), (7, 7): (0, 7), (8, 0): (6, 8), (8, 1): (5, 8), (8, 2): (4, 8), (8, 3): (3, 8), (8, 4): (2, 8), (8, 5): (1, 8), (8, 6): (0, 8), (9, 0): (5, 9), (9, 1): (4, 9), (9, 2): (3, 9), (9, 3): (2, 9)}

        #300 degrees
        zeta={(0, 4): (4, 0), (0, 5): (3, 1), (0, 6): (2, 2), (0, 7): (1, 3), (0, 8): (0, 4), (0, 9): (-1, 5), (1, 3): (5, 0), (1, 4): (4, 1), (1, 5): (3, 2), (1, 6): (2, 3), (1, 7): (1, 4), (1, 8): (0, 5), (1, 9): (-1, 6), (2, 2): (6, 0), (2, 3): (5, 1), (2, 4): (4, 2), (2, 5): (3, 3), (2, 6): (2, 4), (2, 7): (1, 5), (2, 8): (0, 6), (2, 9): (-1, 7), (3, 1): (7, 0), (3, 2): (6, 1), (3, 3): (5, 2), (3, 4): (4, 3), (3, 5): (3, 4), (3, 6): (2, 5), (3, 7): (1, 6), (3, 8): (0, 7), (3, 9): (-1, 8), (4, 0): (8, 0), (4, 1): (7, 1), (4, 2): (6, 2), (4, 3): (5, 3), (4, 4): (4, 4), (4, 5): (3, 5), (4, 6): (2, 6), (4, 7): (1, 7), (4, 8): (0, 8), (4, 9): (-1, 9), (5, 0): (9, 0), (5, 1): (8, 1), (5, 2): (7, 2), (5, 3): (6, 3), (5, 4): (5, 4), (5, 5): (4, 5), (5, 6): (3, 6), (5, 7): (2, 7), (5, 8): (1, 8), (5, 9): (0, 9), (6, 0): (9, 1), (6, 1): (8, 2), (6, 2): (7, 3), (6, 3): (6, 4), (6, 4): (5, 5), (6, 5): (4, 6), (6, 6): (3, 7), (6, 7): (2, 8), (6, 8): (1, 9), (7, 0): (9, 2), (7, 1): (8, 3), (7, 2): (7, 4), (7, 3): (6, 5), (7, 4): (5, 6), (7, 5): (4, 7), (7, 6): (3, 8), (7, 7): (2, 9), (8, 0): (9, 3), (8, 1): (8, 4), (8, 2): (7, 5), (8, 3): (6, 6), (8, 4): (5, 7), (8, 5): (4, 8), (8, 6): (3, 9), (9, 0): (8, 5), (9, 1): (7, 6), (9, 2): (6, 7), (9, 3): (5, 8)}

        nevents=out_array.shape[0]
        temp_array=np.zeros((nevents*4,10,10,out_array.shape[-1]))
        nind=0
        for ind in range(nevents):
            for rots in [alpha,beta,gamma,delta]:
                temp=np.zeros((10,10,out_array.shape[-1]))
                for i,c in np.ndenumerate(out_array[ind]):
                    if c>0:
                        new_coords=(rots[i[:2]][0],rots[i[:2]][1],i[-1])
                        if new_coords[:2] in alpha.values():
                            temp[new_coords]=c
                temp_array[nind]=temp
                nind+=1
        out_array=temp_array
        # Update y, too
        temp_y={'comp':np.empty(nevents*4), 'energy':np.empty(nevents*4), 'dir':np.empty((nevents*4,2)), 'plane_dir':np.empty((nevents*4,2)), 'laputop_dir':np.empty((nevents*4,2)), 'small_dir':np.empty((nevents*4,2))}
        for key in y:
            nind=0
            for ind in range(nevents):
                for num in range(4):
                    temp_y[key][nind]=y[key][ind]
                    nind+=1
            y[key]=temp_y[key]


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

""" Filter NaNs from reconstruction """
def filterReco(prep, y, x_i):
    if prep['reco'] != None:
        th, _ = y['{}_dir'.format(prep['reco'])].transpose()
        th = th.astype('float') 
        th = np.pi - th 
        if prep['cosz']:
            th = np.cos(th)
        if prep['normed'] and not prep['cosz']: 
            th /=np.nanmax(th)

        nanCut = ~np.isnan(th)
        x_i[1] = x_i[1][nanCut] 
        x_i[0] = x_i[0][nanCut]

        for key in y.keys():
            y[key] = y[key][nanCut]


def load_mc(filePath):

    mcfiles = sorted(glob('%s/sim_*_mc.npy' % filePath))

    mc = defaultdict(list)
    for _, f in enumerate(mcfiles):
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