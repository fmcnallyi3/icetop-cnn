#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/icetray-start
#METAPROJECT icetray/v1.17.0

import argparse
import os
import getpass
from glob import glob

import numpy as np

from sim_utils import load_data, dict_to_mat


def _to_float32_dir(rows):
    cleaned = [[np.nan, np.nan] if (r is None or r[0] is None or r[1] is None) else r for r in rows]
    return np.asarray(cleaned, dtype=np.float32)


def main(args):
    
    
    # Verify source folder exists
    source = args.source;
    sim_name = source[source.lstrip().rfind('/')+1:]
    if not os.path.exists(source):
        raise Exception('Source folder containing extracted simulation data was not found.')
    # Verify destination folder exists, create otherwise.
    data_dir = os.environ.get('ICETOP_CNN_DATA_DIR', f'/data/user/{getpass.getuser()}/icetop-cnn')
    dest = os.path.join(data_dir, 'simdata', sim_name)
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
        if not d['comp']:
            print(f'Skipping empty file {f}')
            continue
        batch = f.rsplit('_', 1)[-1]
        unique_comps = set(d['comp'])
        if len(unique_comps) > 1:
            print(f'Warning: mixed compositions {sorted(unique_comps)} in {f}; using first.')
        first_comp = d['comp'][0]
        alias = comp_aliases.get(comp.get(first_comp), str(first_comp))
        # simset ID in the name: batch numbers restart at 000 per simset, so
        # alias+batch alone collides across simsets and skip-if-exists drops data
        simset = f.split('_')[-2]
        array_out = os.path.join(dest, 'icetop', f'icetop_{alias}_{simset}_{sim_name}_{batch}')
        infill_out = os.path.join(dest, 'icetop', f'infill_{alias}_{simset}_{sim_name}_{batch}')
        param_out = os.path.join(dest, 'event_parameters', f'event_parameters_{alias}_{simset}_{sim_name}_{batch}')

        # Skip files already created if not overwriting
        array_done = os.path.isfile(array_out) and (not args.infill or os.path.isfile(infill_out))
        param_done = os.path.isfile(param_out)
        if not args.overwrite and any([
            args.output == 'both'  and array_done and param_done,
            args.output == 'array' and array_done,
            args.output == 'param' and param_done
        ]):
            continue

        print(f'Converting {f}...')

        if args.output in ['array', 'both']:
            array_data, infill_data = dict_to_mat(d, infill=args.infill)
            np.save(array_out, array_data)
            if args.infill:
                np.save(infill_out, infill_data)

        if args.output in ['param','both']:
            param_data = {
                # Event info
                'file_info':       d['file_info'],
                'energy':          np.log10(d['energy']).astype(np.float32),
                'comp':            np.array([comp[c] for c in d['comp']]),
                'dir':             _to_float32_dir(d['dir']),
                'plane_dir':       _to_float32_dir(d['plane_dir']),
                'laputop_dir':     _to_float32_dir(d['laputop_dir']),
                'small_dir':       _to_float32_dir(d['small_dir']),

                # CODE GOES HERE

                # Cuts
                'passed_STA5':     np.asarray(d['passed_STA5'], dtype=np.bool_),
                'uncontained_cut': np.asarray(d['uncontained_cut'], dtype=np.bool_),
                'quality_cut':     np.asarray(d['quality_cut'], dtype=np.bool_)
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
