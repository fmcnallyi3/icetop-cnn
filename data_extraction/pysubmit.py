#!/usr/bin/env python

#############################################################################
## A python wrapper for submitting executables. To use, import pysubmit    ##
## and call in a python script.                                            ##
##                                                                         ##
## Arguments:                                                              ##
## - executable - string with the executable argument                      ##
##    - ex: '/home/user/test.py -a [argument]'                             ##
## - jobID - title for the job in condor and exec/out/log/error files      ##
## - outdir - location for exec/out/log/error files                        ##
## - test - run executable off cluster as a test                           ##
## - sublines - location for additional submission options                 ##
##    - replace eventually with actual options (like 'universe')           ##
#############################################################################

import subprocess
import stat, random
from pathlib import Path
import os


def pysubmit(executable, jobID=None, outdir=None,
             test=False, universe='vanilla',
             header=['#!/bin/bash'],
             notification='never', sublines=None):

    # Default storage for exec/out/log/error files is in this directory
    if outdir == None:
        outdir = Path(__file__).parent.resolve()
    outdir = Path(outdir)

    # Default naming for jobIDs if not specified
    if jobID == None:
        jobID = f'npx4-{random.randint(0,999999):06d}' 
        # Only print ID if actually submitting, or debug info if testing
        if not test:
            print(jobID)

    # Ensure output directories exist
    outdir.mkdir(parents=True, exist_ok=True)
    for condor_out in ['execs','logs','out','error']:
        condor_dir = outdir / f'npx4-{condor_out}'
        condor_dir.mkdir(exist_ok=True)

    # Create execution script (The Wrapper)
    # We create this even in test mode to ensure we test exactly what we ship
    exelines = header + [
        'date',
        'hostname',
        '',
        f'{executable}',
        '',
        'date',
        'echo "Fin"'
    ]
    exelines = [l+'\n' for l in exelines]

    exe_out = Path(f'{outdir}/npx4-execs/{jobID}.sh')
    with open(exe_out, 'w') as f:
        f.writelines(exelines)

    # Make file executable
    mode = exe_out.stat().st_mode
    exe_out.chmod(mode | stat.S_IEXEC)

    # --- TEST MODE START ---
    if test:
        print(f"--- DEBUG MODE: Running Locally on {os.uname().nodename} ---")
        print(f"Wrapper Script: {exe_out}")
        print("-" * 40)
        
        try:
            # We run the wrapper script directly.
            # check=True raises an error if the script fails.
            # We do NOT capture output so you can see it stream to your terminal.
            subprocess.run([str(exe_out)], check=True)
            print("-" * 40)
            print("--- Local execution successful ---")
        except subprocess.CalledProcessError as e:
            print("-" * 40)
            print(f"--- Local execution FAILED (Exit Code: {e.returncode}) ---")
        
        # Exit the function here so we don't try to submit to Condor
        return
    # --- TEST MODE END ---

    # Condor submission script
    lines = [
        f'universe = {universe}',
        'getenv = True',
        f'executable = {outdir}/npx4-execs/{jobID}.sh',
        f'log = {outdir}/npx4-logs/{jobID}.log',
        f'output = {outdir}/npx4-out/{jobID}.out',
        f'error = {outdir}/npx4-error/{jobID}.error',
        f'notification = {notification}',
        'queue'
    ]
    lines = [l+'\n' for l in lines]

    # Option for additional lines to submission script
    if sublines != None:
        for l in sublines:
            lines.insert(-1, f'{l}\n')

    # Submission script for condor
    condor_script = f'{outdir}/2sub.sub'
    with open(condor_script, 'w') as f:
        f.writelines(lines)

    # Submit to HTCondor
    env = os.environ.copy()
    try:
        result = subprocess.run(
            ['condor_submit', condor_script],
            capture_output=True,  
            text=True,            
            check=True,           
            env=env
        )
        print("Job submitted successfully:")
        print(result.stdout)
    
    except FileNotFoundError:
        print("Error: 'condor_submit' command not found. Is HTCondor installed and in your PATH?")
        sys.exit(1)
    
    except subprocess.CalledProcessError as e:
        print(f"Error executing condor_submit (Exit code {e.returncode}):")
        print(e.stderr)
        sys.exit(1)