#!/usr/bin/env python3

import sys
import argparse
import os
from glob import glob
import subprocess
import stat, random
from pathlib import Path

from pysubmit import pysubmit

def main(args):
    gcd_path = Path(args.gcd_path)
    file_path = Path(args.inp_path.lstrip())

    #Create the output folder and parse the sim name
    Path(args.out_dir.lstrip()).mkdir(parents=True, exist_ok=True)
    sim_name = args.out_dir.lstrip()[args.out_dir.lstrip().rfind('/')+1:]
    #Path to extractor
    exe = './sim-extractor.py'
    #Path to data 
    l3sim = '/data/ana/CosmicRay/IceTop_level3/sim/IC86.2012'

    comp = {'proton': [12360], 'helium': [12630], 'oxygen': [12631], 'iron': [12362], 'all': [12360, 12630, 12631, 12362]}
    c = comp[args.composition]    
    count = 0
    for ID in c:
        #gcd = f'{l3sim}/GCD/Level3_{ID}_GCD.i3.gz'
        #files = glob(f'{l3sim}/oldstructure/{ID}/Level3_IC86.2012_{ID}_Run??????.i3.gz')
        gcd = args.gcd_path
        files = list(file_path.iterdir())

        if args.mc:
            mc_prefix = '/data/sim/IceTop/2012/generated/CORSIKA-ice-top'
            files = glob(f'{mc_prefix}/{ID}/topsimulator/*/*.i3.gz')
        files.sort()
        
        if not(".i3" in str(files[0])):
            files = files[1:]

        batch_list = [[str(f) for f in files[i:i+args.n]] for i in range(0, len(files), args.n)]

        for i, batch in enumerate(batch_list):
            if args.test:
                batch = batch[:5]
            batched_files = ' '.join(batch)            
            out_file = f'{args.out_dir}/sim_{sim_name}_{i:03}.npy'

            if args.mc:
                exe = exe.replace('extractor', 'mc_extractor')
                out_file = out_file.replace('.npy', '_mc.npy')
            if args.test:
                out_file = out_file.replace('.npy','_test.npy')

            exelines = f'{exe} -g {gcd} -f {batched_files} -o {out_file}'
            
            pysubmit(exelines, outdir=f'./condor', test=args.test)

            if count == 0:
                break
            count += 1
            #print(count)
        break


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Runs extractor.py on the cluster')
    p.add_argument('-c', '--composition', dest='composition', type=str,
        default='proton', choices=['proton','helium','oxygen','iron','all'],
        help='Composition you want to run over')
    p.add_argument('-n', '--n', dest='n', type=int,
        default=1000,
        help='Number of files to run per batch')
    p.add_argument('-o', '--out', dest='out_dir', type=str,
        default=f'/data/user/{os.getlogin()}/sim',
        help='Output directory')
    p.add_argument('--mc', dest='mc',
        default=False, action='store_true',
        help='Extract energy information from generated sim files instead')
    p.add_argument('--test', dest='test',
        default=False, action='store_true',
        help='Option for running test off cluster')
    #Arguments added by yours truly!
    p.add_argument('-g', '--gcdPath', dest='gcd_path', type=str,
        help='Path to the GCD file')
    p.add_argument('-i', '--inputPath', dest='inp_path', type=str,
        help='Path to the input files')

    main(p.parse_args())