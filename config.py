# Configuration file for the trainer script

VALIDATION_SPLIT = 0.15
BATCH_SIZE = 32

PREP = {
    'infill': False,
    'clc': True,
    'sta5': False,
    'q': None,
    't': None,
    't_shift': True,
    't_clip': 0.0,
    'normed': True,
    'reco': 'plane',
}

MAX_EPOCHS = 100

MIN_ES_DELTA = 1e-4
ES_PATIENCE = 10
RESTORE_BEST_WEIGHTS = True

INITIAL_LR = 1e-3
LR_FACTOR = 0.4
LR_PATIENCE = 5
MIN_LR_DELTA = 1e-4
LR_COOLDOWN = 3
MIN_LR = 1e-6
