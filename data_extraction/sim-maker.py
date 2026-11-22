#!/usr/bin/env python3

import sys
import argparse
import os
import getpass
from glob import glob
import subprocess
import stat, random
from pathlib import Path

from pysubmit import pysubmit

def simset_of(path):
    # simset dir: '12360_v1' in the regular trees, bare '20144' in NewL3_MC
    return next((pt.split('_')[0] for pt in reversed(Path(path).parts)
                 if pt.split('_')[0].isdigit()), None)

def main(args):
    gcd_path = Path(args.gcd_path) if args.gcd_path else None
    file_path = Path(args.inp_path.lstrip())

    #Create the output folder and parse the sim name
    Path(args.out_dir.lstrip()).mkdir(parents=True, exist_ok=True)
    sim_name = args.out_dir.lstrip()[args.out_dir.lstrip().rfind('/')+1:]
    
    #Path to extractor
    exe = f'/home/{getpass.getuser()}/icetop-cnn/data_extraction/sim-extractor.py'
    #Path to data 
    l3sim = '/data/ana/CosmicRay/IceTop_level3/sim/IC86.2012'

    comp = {'proton': [12360], 'helium': [12630], 'oxygen': [12631], 'iron': [12362], 'all': [12360, 12630, 12631, 12362]}
    c = comp[args.composition]    

    # per-ID loop re-globbed the SAME files once per composition (4x jobs AND
    # 4x duplicate data with -c all). One pass for real data; per-ID loop is
    # an --mc-only concern. Drop this whole block if --mc dies.
    for ID in (c if args.mc else [c[0]]):
        if args.mc:
            mc_prefix = '/data/sim/IceTop/2012/generated/CORSIKA-ice-top'
            files = glob(f'{mc_prefix}/{ID}/topsimulator/*/*.i3.gz')
        else:
            # -i may be a comp root spanning many simsets; per-simset GCD is
            # resolved per BATCH below (each sim set has its own GCD).
            files = sorted(f for f in file_path.rglob('*.i3*') if f.is_file() and not f.is_symlink() and not any(x in f.parts for x in ('_ITonly', 'incomplete_obsolete')))
        files.sort()
        
        # empty dir guard: version dirs / filtered dirs can hold zero extractable files
        if not files:
            print(f'No extractable .i3 files under {file_path}; skipping.')
            return
        if not(".i3" in str(files[0])):
            files = files[1:]

        # Group by simset dir, chunk each group; jobs = ceil(files_in_simset/n).
        by_simset = {}
        for f in files:
            simset = simset_of(f)
            if(not simset):
                print(f'No simset dir in {f}; skipping file.')
                continue
            by_simset.setdefault(simset, []).append(str(f))
        batch_list = []
        for group in by_simset.values():
            batch_list += [[str(x) for x in group[i:i+args.n]] for i in range(0, len(group), args.n)]

        for i, batch in enumerate(batch_list):
            # per-batch GCD: files in one batch can still span energy-bin dirs of
            # one simset, and batches are sequential over the sorted list, so a
            # batch boundary can split a simset. Resolve from batch[0]'s path.
            # simset from the batch's first file drives out_file naming
            simset = simset_of(batch[0])
            # -g wins outright; else walk up: tree depth differs by production
            # year (2012 nests energy bins, 2018 does not), so probe each
            # ancestor for a GCD dir with this simset's file, else a lone
            # year-wide GeoCalib
            gcd = str(gcd_path) if gcd_path else None
            if gcd is None:
                for anc in list(Path(batch[0]).parents)[:6]:
                    cands = [anc / 'GCD' / f'Level3_{simset}_GCD.i3.gz']
                    if cands[0].exists():
                        gcd = str(cands[0]); break
                    geo = sorted(glob(str(anc / 'GCD' / '*.i3.gz')))
                    if len(geo) == 1:
                        gcd = geo[0]; break
                if not gcd:
                    print(f'No GCD found for simset {simset} (batch {i}); skipping batch.')
                    continue
            if args.test:
                batch = batch[:5]
            batched_files = ' '.join(batch)            
            # Weak spot fix: include the sim set ID so batches from different
            # primaries in one extraction cannot overwrite each other.
            out_file = f'{args.out_dir}/sim_{sim_name}_{simset}_{i:03}.npy'

            if args.mc:
                exe = exe.replace('extractor', 'mc_extractor')
                out_file = out_file.replace('.npy', '_mc.npy')
            if args.test:
                out_file = out_file.replace('.npy','_test.npy')

            exelines = f'{exe} -g {gcd} -f {batched_files} -o {out_file}'
            
            pysubmit(exelines, outdir=f'./condor', test=args.test,
                      sublines=['request_memory = 12 GB'])

            if args.test:
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
        default=f'/data/user/{getpass.getuser()}/sim',
        help='Output directory')
    p.add_argument('--mc', dest='mc',
        default=False, action='store_true',
        help='Extract energy information from generated sim files instead')
    p.add_argument('--test', dest='test',
        default=False, action='store_true',
        help='Option for running test off cluster')
    #Arguments added by yours truly!
    p.add_argument('-g', '--gcdPath', dest='gcd_path', type=str, default=None,
        help='Path to the GCD file')
    p.add_argument('-i', '--inputPath', dest='inp_path', type=str,
        help='Path to the input files')

    main(p.parse_args())