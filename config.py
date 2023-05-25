import os

GPU_ID = '0'

SIMDATA_FOLDER_PATH = os.path.join(os.getcwd(), 'simdata')
NUCLEI = 'phof'

VALIDATION_SPLIT = 0.15
BATCH_SIZE = 256

MODELS_FOLDER_PATH = os.path.join(os.getcwd(), 'models')
MODEL_NAME = 'placeholder0'
PREP = {
    'infill': False,
    'clc': True,
    'sta5': False,
    'q': None,
    't': None,
    't_shift': True,
    't_clip': 0,
    'normed': True,
    'reco': None,
}

LOSS_FUNCTION = 'huber_loss'
METRICS = ['mae','mse']
NUM_EPOCHS = 100

MIN_ES_DELTA = 1e-4
ES_PATIENCE = 15
RESTORE_BEST_WEIGHTS = True

LR_FACTOR = 0.4
LR_PATIENCE = 5
MIN_LR_DELTA = 1e-4
LR_COOLDOWN = 3
MIN_LR = 1e-6