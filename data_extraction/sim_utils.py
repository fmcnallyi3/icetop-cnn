from collections import defaultdict
from sys import version_info

import numpy as np

def load_data(f, infill=False):

    d = defaultdict(list)
    single_keys = ['gain', 'position'] # Keys not tied to events

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


def _build_grid_meta():
    sta_to_hex = hex_square()
    station_list = sorted(sta_to_hex.keys())
    squareY, squareX = np.transpose([sta_to_hex[s] for s in station_list])
    min_y = int(squareY.min())
    min_x = int(squareX.min())
    nx = int(squareX.max()) - min_x + 1
    ny = int(squareY.max()) - min_y + 1

    # Precompute dom -> (is_infill, idx1, idx2, tank_offset) for IceTop & infill DOMs
    dom_lut = {}
    for s, (yy, xx) in sta_to_hex.items():
        y_i = int(yy) - min_y
        x_i = int(xx) - min_x
        for om in ('61', '62', '63', '64'):
            dom_lut[s + om] = (False, y_i, x_i, 0 if om in ('61', '62') else 2)
    for s in range(79, 82):
        s_str = f'{s:02d}'
        for om in ('61', '62', '63', '64'):
            dom_lut[s_str + om] = (True, s - 79, 0, 0 if om in ('61', '62') else 2)

    return sta_to_hex, min_y, min_x, ny, nx, dom_lut


_STA_TO_HEX, _MIN_Y, _MIN_X, _NY, _NX, _DOM_LUT = _build_grid_meta()


""" Convert dictionary format to multi-layered array for use with CNN """
def dict_to_mat(d, infill=True):

    nevents = len(d['charge_HLC'])
    depth = 8   # q1h, q1s, q2h, q2s, t1h, t1s, t2h, t2s

    icetop = np.zeros((nevents, _NY, _NX, depth), dtype=np.float32)
    infill_arr = np.zeros((nevents, 3, depth), dtype=np.float32) if infill else None

    # Layers #1 and #3: HLC charge at each tank
    # Layers #2 and #4: SLC charge at each tank
    # Layers #5 and #7: HLC time at each tank
    # Layers #6 and #8: SLC time at each tank

    dom_lut = _DOM_LUT
    lut_get = dom_lut.get
    chunk_size = 100000

    for i1, value in ((0, 'charge_HLC'), (1, 'charge_SLC'), (4, 'time_HLC'), (5, 'time_SLC')):
        series = d[value]
        for start in range(0, nevents, chunk_size):
            ev_idx, y_idx, x_idx, c_idx, vals = [], [], [], [], []
            inf_ev, inf_s, inf_c, inf_vals = [], [], [], []
            for i, v in enumerate(series[start:start + chunk_size], start):
                for dom, v_i in v.items():
                    entry = lut_get(dom)
                    if entry is None:
                        sta = int(dom[:2])
                        tank_offset = 0 if dom[2:] in ('61', '62') else 2
                        if sta < 79:
                            yy, xx = _STA_TO_HEX[dom[:2]]
                            entry = (False, yy - _MIN_Y, xx - _MIN_X, tank_offset)
                        else:
                            entry = (True, sta - 79, 0, tank_offset)
                        dom_lut[dom] = entry
                    is_inf, a, b, off = entry
                    if not is_inf:
                        ev_idx.append(i)
                        y_idx.append(a)
                        x_idx.append(b)
                        c_idx.append(i1 + off)
                        vals.append(v_i)
                    elif infill:
                        inf_ev.append(i)
                        inf_s.append(a)
                        inf_c.append(i1 + off)
                        inf_vals.append(v_i)
            if vals:
                np.add.at(
                    icetop,
                    (
                        np.asarray(ev_idx, dtype=np.intp),
                        np.asarray(y_idx, dtype=np.intp),
                        np.asarray(x_idx, dtype=np.intp),
                        np.asarray(c_idx, dtype=np.intp),
                    ),
                    np.asarray(vals, dtype=np.float32),
                )
            if infill and inf_vals:
                np.add.at(
                    infill_arr,
                    (
                        np.asarray(inf_ev, dtype=np.intp),
                        np.asarray(inf_s, dtype=np.intp),
                        np.asarray(inf_c, dtype=np.intp),
                    ),
                    np.asarray(inf_vals, dtype=np.float32),
                )

    return icetop, infill_arr
