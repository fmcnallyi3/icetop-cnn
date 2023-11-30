from collections import defaultdict
from sys import version_info

import numpy as np

def load_data(f, infill=False):

    d = defaultdict(list)
    single_keys = ['gain','position'] # Keys not tied to events
    dom_keys = ['charge_HLC', 'charge_SLC', 'time_HLC', 'time_SLC'] # Keys tied to DOMs

    # Concatenate all information tied to events
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

    # Remove infill array if not desired
    if not infill:
        for key in dom_keys:
            d[key] = [{k:v for k, v in dom_series.items() if int(k[:2])<79} for dom_series in d[key]]

    return d


""" Hard-coded string to pix stolen from Mirco (deepLearning/hexagon/hexConv.py) """
def hex_square():
    
    return {'01': (-4,-1), '02': (-4,0),  '03': (-4,1),  '04': (-4,2),  '05': (-4,3),  '06': (-4,4),  '07': (-3,-2), '08': (-3,-1), '09': (-3,0), '10': (-3,1),
            '11': (-3,2),  '12': (-3,3),  '13': (-3,4),  '14': (-2,-3), '15': (-2,-2), '16': (-2,-1), '17': (-2,0),  '18': (-2,1),  '19': (-2,2), '20': (-2,3),
            '21': (-2,4),  '22': (-1,-4), '23': (-1,-3), '24': (-1,-2), '25': (-1,-1), '26': (-1,0),  '27': (-1,1),  '28': (-1,2),  '29': (-1,3), '30': (-1,4),
            '31': (0,-5),  '32': (0,-4),  '33': (0,-3),  '34': (0,-2),  '35': (0,-1),  '36': (0,0),   '37': (0,1),   '38': (0,2),   '39': (0,3),  '40': (0,4),
            '41': (1,-5),  '42': (1,-4),  '43': (1,-3),  '44': (1,-2),  '45': (1,-1),  '46': (1,0),   '47': (1,1),   '48': (1,2),   '49': (1,3),  '50': (1,4),
            '51': (2,-5),  '52': (2,-4),  '53': (2,-3),  '54': (2,-2),  '55': (2,-1),  '56': (2,0),   '57': (2,1),   '58': (2,2),   '59': (2,3),  '60': (3,-5),
            '61': (3,-4),  '62': (3,-3),  '63': (3,-2),  '64': (3,-1),  '65': (3,0),   '66': (3,1),   '67': (3,2),   '68': (4,-5),  '69': (4,-4), '70': (4,-3),
            '71': (4,-2),  '72': (4,-1),  '73': (4,0),   '74': (4,1),   '75': (5,-5),  '76': (5,-4),  '77': (5,-3),  '78': (5,-2)}


""" Convert dictionary format to multi-layered array for use with CNN """
def dict_to_mat(d):

    sta_to_hex = hex_square()
    station_list = sorted(sta_to_hex.keys())
    squareY, squareX = np.transpose([sta_to_hex[s] for s in station_list])

    nevents = len(d['charge_HLC'])
    nx = squareX.max() - squareX.min() + 1
    ny = squareY.max() - squareY.min() + 1
    depth = 8   # q1h, q1s, q2h, q2s, t1h, t1s, t2h, t2s

    icetop = np.zeros((nevents, ny, nx, depth))
    infill = np.zeros((nevents, 3, depth))

    # Layers #1 and #3: HLC charge at each tank
    # Layers #2 and #4: SLC charge at each tank
    # Layers #5 and #7: HLC time at each tank
    # Layers #6 and #8: SLC time at each tank

    values = ['charge_HLC', 'charge_SLC', 'time_HLC', 'time_SLC']
    indices = [(0, 2), (1, 3), (4, 6), (5, 7)]
    for value, (i1, i2) in zip(values, indices):
        for i, v in enumerate(d[value]):
            for dom, v_i in v.items():
                if int(dom[:2]) < 79:
                    y_i = sta_to_hex[dom[:2]][0] - squareY.min()
                    x_i = sta_to_hex[dom[:2]][1] - squareX.min()
                    if dom[2:] in ['61','62']:
                        icetop[i][y_i][x_i][i1] += v_i
                    else:
                        icetop[i][y_i][x_i][i2] += v_i
                else:
                    if dom[2:] in ['61','62']:
                        infill[i][int(dom[:2]) - 79][i1] += v_i
                    else:
                        infill[i][int(dom[:2]) - 79][i2] += v_i

    return icetop, infill
