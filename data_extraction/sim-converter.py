#!/usr/bin/env python3

import argparse
import os
from glob import glob

import numpy as np

from sim_utils import load_data, dict_to_mat

def main(args):

    # Verify source folder exists
    source = f'/data/user/{os.getlogin()}/sim'
    if not os.path.exists(source):
        raise Exception('Source folder containing extracted simulation data was not found.')
    # Verify destination folder exists, create otherwise.
    dest = f'/data/user/{os.getlogin()}/preprocessed'
    if not os.path.exists(dest):
        os.mkdir(dest)

    # Loop over files and convert array and parameter data
    files = sorted(glob(f'{source}/sim_*.npy'))
    for f in files:
        array_out = f.replace('/sim/sim_', '/preprocessed/icetop_')
        param_out = f.replace('/sim/sim_', '/preprocessed/event_parameters_')

        # Skip files already created if not overwriting
        if not args.overwrite and any([
            args.output == 'both'  and os.path.isfile(array_out) and os.path.isfile(param_out),
            args.output == 'array' and os.path.isfile(array_out),
            args.output == 'param' and os.path.isfile(param_out)
        ]):
            continue

        print(f'Converting {f}...')
        d = load_data(f, infill=args.infill)

        if args.output in ['array', 'both']:
            array_data, infill_data = dict_to_mat(d)
            np.save(array_out, array_data)
            if args.infill:
                np.save(array_out.replace('icetop_', 'infill_'), infill_data)

        
        comp = {'PPlus':1, 'He4Nucleus':4, 'O16Nucleus':16, 'Fe56Nucleus':56}
        if args.output in ['y','both']:
            param_data = {
                # Event info
                'file_info':       d['file_info'],
                'energy':          np.log10(d['energy']),
                'comp':            np.array([comp[c] for c in d['comp']]),
                'dir':             d['dir'],
                'plane_dir':       d['plane_dir'],
                'laputop_dir':     d['laputop_dir'],
                'small_dir':       d['small_dir'],

                """ CODE GOES HERE """

                # Cuts
                'passed_STA5':     d['passed_STA5'],
                'uncontained_cut': d['uncontained_cut'],
                'quality_cut':     d['quality_cut']
            }
            np.save(param_out, param_data)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
            description='Converts numpy dictionaries of events into matrices')
    p.add_argument('-i', '--infill', dest='infill',
            default=False, action='store_true',
            help='Option to include infill array')
    p.add_argument('-o', '--output', dest='output',
            default='both', choices=['array','param','both'],
            help='Output detector reaction (x), primary info (y), or both')
    p.add_argument('--overwrite', dest='overwrite',
            default=False, action='store_true',
            help='Option to overwrite existing matrix files')
    main(p.parse_args())
