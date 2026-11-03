# ASSESSMENT FILE
# Loads or generates reconstructions for every (model, dataset) pair.
# Called by the assessment notebooks; cache-checking happens here.

import os
import subprocess

import numpy as np

from utils import data_prep, files, get_event_parameters, get_training_assessment_cut, load_detector_array

# Map the nuclei string representations to their atomic weights.
comp_conversion = {'p':1, 'h':4, 'o':16, 'f':56}

# Training container image. The venv's keras cannot load container-era .keras files,
# so each model is converted to .h5 once, inside the container.
SIF_PATH = '/data/user/fmcnally/icetop-cnn/tf.sif'


def ensure_h5(models_folder_path, model_name):
    '''Converts a container-era .keras model to .h5 so this venv can load it.'''
    keras_path = os.path.join(models_folder_path, model_name, model_name + '.keras')
    h5_path = os.path.join(models_folder_path, model_name, model_name + '.h5')
    if not os.path.exists(h5_path):
        code = 'import tensorflow as tf; m = tf.keras.models.load_model(%r); m.save(%r)' % (keras_path, h5_path)
        subprocess.run(['apptainer', 'exec', '--bind', '/data/user', SIF_PATH, 'python3', '-c', code], check=True)
    return h5_path


def predict(models_folder_path, model_parameters, model_name, dataset):
    '''Predicts a model over a dataset's assessment split. Same seeded cut for every model.
    Loads one batch file at a time so full-dataset detector arrays never sit in memory.'''
    import tensorflow as tf
    prep = model_parameters[model_name]
    model = tf.keras.models.load_model(ensure_h5(models_folder_path, model_name), compile=False)
    path = dataset if os.path.isabs(dataset) else os.path.join(os.getenv('ICETOP_CNN_DATA_DIR'), 'simdata', dataset)
    ep = get_event_parameters(path, composition='phof')
    cut = get_training_assessment_cut(ep, 'assessment', prep)
    # Match training-time recons: predict only on the model's assessment nuclei.
    cut = cut * np.isin(ep['comp'], [comp_conversion[n] for n in prep['assessment_nuclei']])
    batch_files = files(path, 'icetop', composition='phof')
    infill_files = None
    if prep['infill']:
        infill_files = files(path, 'infill', composition='phof')
    out, n = [], 0
    for i, bf in enumerate(batch_files):
        detector_inputs = {'icetop': load_detector_array([bf])}
        if infill_files is not None:
            detector_inputs['infill'] = load_detector_array([infill_files[i]])
        detector_inputs = data_prep(detector_inputs, prep)
        file_cut = cut[n:n + len(detector_inputs['icetop'])]
        if file_cut.any():
            inputs = [data[file_cut] for data in detector_inputs.values()]
            out.append(model.predict(inputs, verbose=0))
        n += len(detector_inputs['icetop'])
    assert n == len(ep['comp']), 'File/event count mismatch: %s on %s' % (model_name, dataset)
    return np.concatenate(out)


def assess_models(model_parameters, DATASETS, eps, models_folder_path, reconstructions_folder_path):
    '''Returns (reconstructions, cuts), each keyed by (model, dataset) pairs.
    Cache-first: existing cross predictions and training-time reconstructions are loaded,
    missing pairs are predicted and cached.'''
    reconstructions, cuts = {}, {}
    for model_name, dataset in [(m, d) for m in model_parameters for d in DATASETS]:
        model_prep = model_parameters[model_name]
        ep = eps[dataset]

        # Cached cross prediction, else the training-time reconstruction, else predict fresh.
        cache = os.path.join(reconstructions_folder_path, '%s__%s.npy' % (model_name, dataset))
        own = model_prep.get('simdata', '')
        own_file = os.path.join(reconstructions_folder_path, model_name + '.npy')
        if os.path.exists(cache):
            reconstruction = np.load(cache)
        elif os.path.basename(own) == dataset and os.path.exists(own_file):
            reconstruction = np.load(own_file)
        else:
            reconstruction = predict(models_folder_path, model_parameters, model_name, dataset)
            np.save(cache, reconstruction)

        # Get the model-specific assessment cut.
        model_cut = get_training_assessment_cut(ep, 'assessment', model_prep)

        # Apply the nuclei cut, then check alignment: recon rows must match cut events.
        nuclei_to_assess = [comp_conversion[nuclei] for nuclei in model_prep['assessment_nuclei']]
        model_cut = np.isin(ep['comp'], nuclei_to_assess) * model_cut
        assert len(reconstruction) == np.sum(model_cut), \
            'Reconstruction/event misalignment: %s on %s' % (model_name, dataset)

        # Flatten the predictions into a one-dimensional array and save them.
        reconstruction = reconstruction.flatten()
        reconstructions[(model_name, dataset)] = reconstruction

        # Save the model-specific events cut.
        cuts[(model_name, dataset)] = model_cut
    return reconstructions, cuts


if __name__ == '__main__':
    print('Imported by the assessment notebooks; nothing to run standalone.')
