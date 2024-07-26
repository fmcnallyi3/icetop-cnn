#!/usr/bin/env python3

import argparse
import os
import sys
from glob import glob

ERROR_ENVIRONMENT_NOT_ACTIVATED = 'Virtual environment not activated. Activate with the command "icetop-cnn"'
ERROR_ARCHITECTURE_NOT_FOUND = f'Model architecture not found in "architectures" folder. Please check for typos.'
ERROR_MODEL_ALREADY_FOUND = 'Model folder already found with this name, but it is incomplete.\n'\
    '  This indicates a model with that name may be currently training.\n'\
    '  You can check your active jobs with "condor_q" when logged on to the submitter.\n'\
    '  At your own risk, re-run with the "--overwrite" flag to overwrite existing data.'

# Check that the environment has been activated
# Not necessary to check if ran on the cluster
# Should check first before the rest of the code is parsed
ICETOP_CNN_DIR = os.getenv('ICETOP_CNN_DIR', '')
venv_path = os.path.join(ICETOP_CNN_DIR, '.venv')
assert os.getenv('VIRTUAL_ENV') == venv_path, ERROR_ENVIRONMENT_NOT_ACTIVATED

ICETOP_CNN_DATA_DIR = os.getenv('ICETOP_CNN_DATA_DIR')
ICETOP_CNN_SCRATCH_DIR = os.getenv('ICETOP_CNN_SCRATCH_DIR')
LOGS_DIR = os.path.join(ICETOP_CNN_SCRATCH_DIR, 'condor', 'logs')

def main(args):
    '''Submission script for training models on the cluster'''

    # None: something went wrong, terminate program and display error
    # False: restore a model and continue training
    # True: overwrite whatever data may exist
    new_model: bool | None = get_model_origin(args)
    if new_model is None: return

    # Check if the user is okay overwriting files if a new model is being trained
    if new_model and not ok_to_overwrite_files(args): return

    assert os.path.exists(f'architectures/{args.model_design}.py'), ERROR_ARCHITECTURE_NOT_FOUND

    # Create model folder
    os.makedirs(os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name), exist_ok=True)
    
    # Optionally run the model off the cluster with a custom command prompt
    if args.test:
        if not args.epochs or args.epochs > 10:
            print('Defaulting to 10 epochs...')
            args.epochs = 10
        command = f'python trainer.py -c {args.composition} -p {" ".join(args.predict)} -e {args.epochs} {("-r", "-o")[new_model]} -t -n {args.model_name} -m {args.model_design}'
        print('Starting training...')
        return os.system(command)
    
    # Create condor directories
    create_required_directories()
    
    lines = [
        f'executable = {os.path.join(ICETOP_CNN_DIR, "trainer.py")}',
        f'arguments = "{" ".join(sys.argv[1:])}"',
        f'environment = "ICETOP_CNN_DIR={ICETOP_CNN_DIR} ICETOP_CNN_DATA_DIR={ICETOP_CNN_DATA_DIR} ICETOP_CNN_SCRATCH_DIR={ICETOP_CNN_SCRATCH_DIR}"',
        'transfer_input_files = config.py,utils.py,model.py,loss_grapher.py',
        'getenv = True',
        '',
        f'log = {os.path.join(LOGS_DIR, f"{args.model_name}.log")}',
        f'output = {os.path.join("condor", "output", f"{args.model_name}.out")}',
        f'error = {os.path.join("condor", "errors", f"{args.model_name }.err")}',
        'notification = never',
        '',
        f'+SingularityImage = "{os.path.join(os.sep, "data", "user", "fmcnally", "icetop-cnn", "tf.sif")}"',
        'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT',
        f'initialdir = {ICETOP_CNN_DIR}',
        '',
        'request_memory = 16G',
        'request_gpus = 1',
        '',
        'requirements = HasSingularity && CudaCapability',
        '',
        'queue',
        ''
    ]

    submission_filepath = os.path.join(ICETOP_CNN_DIR, 'condor', 'submissions', f'{args.model_name}.sub')
    with open(submission_filepath, 'w') as f:
        f.write('\n'.join(lines))

    return os.system(f'condor_submit {submission_filepath}')

def get_model_origin(args):
    '''Returns True if a new model should be created, False if and old model should be restored, and None if there is an error.'''
    if args.overwrite: return True
    
    model_path = os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name)
    # Check to see if the model folder exists. This could indicate that the model already exists or is actively training
    if not os.path.exists(model_path): return True

    # The model folder exists -> check for model weights and configuration files
    assert all(os.path.exists(os.path.join(model_path, f'{args.model_name}.{ext}')) for ext in ('keras', 'json')), ERROR_MODEL_ALREADY_FOUND

    # Model was found, and the user indicated that they would like to restore it
    if args.restore:
        return False

    # Model was found, but the user did not indicate whether they wanted to restore or overwrite the model
    print('Model already found with this name.\n'
          '  Choose a different name or run with EITHER the "--overwrite" or "--restore" flags enabled.')
    return

def ok_to_overwrite_files(args):
    '''Prompts the user and returns a boolean indicating whether they are sure they want to overwrite a model'''
    files = sorted(
        # Data files
        glob(os.path.join('*', '*', f'{args.model_name}.*'), root_dir=ICETOP_CNN_DATA_DIR) +
        # Submission files
        glob(os.path.join('condor', '*', f'{args.model_name}.*')) + 
        # Log file
        glob(os.path.join(LOGS_DIR, f'{args.model_name}.log'))
    )
    if files:
        print('WARNING: The following files will be overwritten:')
        for file in files: print(' '*9 + f'- {file}')
        if input('Would you like to continue? [y/n]: ').lower() not in ('y', 'ye', 'yes'):
            print('Exiting...')
            return False
    return True

def create_required_directories():
    ''' Create all required directories needed for job submission '''
    required_directories = [
        # Submission scripts
        os.path.join(ICETOP_CNN_DIR, 'condor', 'submissions'),
        # Standard output
        os.path.join(ICETOP_CNN_DIR, 'condor', 'output'),
        # Standard error
        os.path.join(ICETOP_CNN_DIR, 'condor', 'errors'),
        # Log files
        LOGS_DIR,
    ]
    for required_directory in required_directories:
        os.makedirs(required_directory, exist_ok=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Submission script to train ML models with TensorFlow on the cluster using GPUs')
    p.add_argument(
        '-n', '--name', dest='model_name', type=str,
        required=True,
        help='Name of the model to train')
    p.add_argument(
        '-c', '--composition', type=str,
        default='phof',
        help='Composition of datasets to load and train the model on.')
    p.add_argument(
        '-e', '--epochs', type=int,
        help='The number of epochs that the model should train for')
    p.add_argument(
        '-m', '--model', dest='model_design', type=str,
        choices=[os.path.splitext(arch)[0] for arch in glob('*.py', root_dir='architectures')],
        default='mini0',
        help='Desired model architecture')
    p.add_argument(
        '-p', '--predict', nargs='+', type=str,
        choices=['comp', 'energy'],
        required=True,
        help='A list of one or more desired model outputs')
    p.add_argument(
        '-t', '--test', action='store_true',
        help='Run the script off the cluster on a limited dataset for a maximum of 10 epochs')
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        '-o', '--overwrite', action='store_true',
        help='Overwrite a model with the same name if it exists. Can not be used with restore')
    g.add_argument(
        '-r', '--restore', action='store_true',
        help='Attempt to restore and continue training a model if it exists. Can not be used with overwrite')
    args = p.parse_args()
    
    '''
    MANUAL ARGUMENT VALIDATION
    All arguments get passed directly to trainer.py, so only arguments that
        will crash a program need to be validated in submit.py (so the time
        spent idling isn't wasted).
    '''
    # Ensure that there are no unrecognized characters in the composition string
    if not all(c in 'phof' for c in args.composition):
        p.error('Unrecognized composition dataset combination requested')
     
    # Ensure epochs are a valid number
    if args.epochs and args.epochs <= 0:
        p.error('Epochs must be a positive value')
    
    main(args)
