#!/usr/bin/env python3

import argparse
import os
from glob import glob

from npx4.pysubmit import pysubmit


def main(args):

    """ CHANGE IF NEEDED """
    exe = f'/home/{os.getlogin()}/sim-extractor.py'

    # Verify simulation extraction script exists
    if not os.path.exists(exe):
        raise Exception('The simulation extraction script (default: "sim-extractor.py") was not found.')
    # Verify output folder exists, create otherwise
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    l3sim = '/data/ana/CosmicRay/IceTop_level3/sim/IC86.2012'
    comp = {'proton':12360, 'helium':12630, 'oxygen':12631, 'iron':12362}
    ID = comp[args.composition]

    gcd = f'{l3sim}/GCD/Level3_{ID}_GCD.i3.gz'
    files = glob(f'{l3sim}/oldstructure/{ID}/Level3_IC86.2012_{ID}_Run??????.i3.gz')
    if args.mc:
        mc_prefix = '/data/sim/IceTop/2012/generated/CORSIKA-ice-top'
        files = glob(f'{mc_prefix}/{ID}/topsimulator/*/*.i3.gz')
    files.sort()

    batch_list = [files[i:i+args.n] for i in range(0, len(files), args.n)]
    for i, batch in enumerate(batch_list):
        if args.test:
            batch = batch[:5]
        batched_files = ' '.join(batch)
        out_file = f'{args.out_dir}/sim_{ID}_{i:03}.npy'
        if args.mc:
            exe = exe.replace('extractor', 'mc_extractor')
            out_file = out_file.replace('.npy', '_mc.npy')
        if args.test:
            out_file = out_file.replace('.npy','_test.npy')

        exelines = f'{exe} -g {gcd} -f {batched_files} -o {out_file}'
        pysubmit(exelines, outdir=f'/home/{os.getlogin()}/npx4', test=args.test)

        if args.test:
            break


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Runs extractor.py on the cluster')
    p.add_argument('-c', '--composition', dest='composition', type=str,
        default='proton', choices=['proton','helium','oxygen','iron'],
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

    main(p.parse_args())
