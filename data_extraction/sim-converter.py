#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/icetray-start
#METAPROJECT icetray/v1.17.0

import argparse
import os
from glob import glob

import numpy as np

from sim_utils import load_data, dict_to_mat

def main(args):
    
    
    # Verify source folder exists
    source = args.source;
    sim_name = source[source.lstrip().rfind('/')+1:]
    if not os.path.exists(source):
        raise Exception('Source folder containing extracted simulation data was not found.')
    # Verify destination folder exists, create otherwise.
    dest = os.path.join(os.environ['ICETOP_CNN_DATA_DIR'], 'simdata', sim_name)
    print(dest)
    for sub in ('icetop', 'event_parameters'):
        os.makedirs(os.path.join(dest, sub), exist_ok=True)

    # Loop over files and convert array and parameter data
    files = sorted(glob(f'{source}/sim_*.npy'))
    comp = {'PPlus':1, 'He4Nucleus':4, 'O16Nucleus':16, 'Fe56Nucleus':56}
    comp_aliases = {1:'12360', 4:'12630', 16:'12631', 56:'12362'}
    for f in files:
        print(f)
        d = load_data(f, infill=args.infill)
        batch = f.rsplit('_', 1)[-1]
        alias = comp_aliases[comp[d['comp'][0]]]
        array_out = os.path.join(dest, 'icetop', f'icetop_{alias}_{sim_name}_{batch}')
        param_out = os.path.join(dest, 'event_parameters', f'event_parameters_{alias}_{sim_name}_{batch}')

        # Skip files already created if not overwriting
        if not args.overwrite and any([
            args.output == 'both'  and os.path.isfile(array_out) and os.path.isfile(param_out),
            args.output == 'array' and os.path.isfile(array_out),
            args.output == 'param' and os.path.isfile(param_out)
        ]):
            continue

        print(f'Converting {f}...')

        if args.output in ['array', 'both']:
            array_data, infill_data = dict_to_mat(d)
            np.save(array_out, array_data)
            if args.infill:
                np.save(array_out.replace('icetop_', 'infill_'), infill_data)

        
        comp = {'PPlus':1, 'He4Nucleus':4, 'O16Nucleus':16, 'Fe56Nucleus':56}
        if args.output in ['param','both']:
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
    p.add_argument('-s', '--source', dest='source',
            help='Source folder of simulation data')
    p.add_argument('-i', '--infill', dest='infill',
            default=False, action='store_true',
            help='Option to include infill array')
    p.add_argument('-o', '--output', dest='output',
            default='both', choices=['array','param','both'],
            help='Output detector reaction (array), primary info (param), or both')
    p.add_argument('--overwrite', dest='overwrite',
            default=False, action='store_true',
            help='Option to overwrite existing matrix files')
    main(p.parse_args())
