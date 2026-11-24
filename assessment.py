# ASSESSMENT FILE
# Loads or generates reconstructions for every (model, dataset) pair.
# Called by the assessment notebooks; cache-checking happens here.

import os
import subprocess

import numpy as np
try:
    import plots  # sets the notebook-wide color cycle
except ImportError:
    pass

from utils import data_prep, files, get_event_parameters, get_training_assessment_cut, load_detector_array


def _iter_slabs(bf, slab_events):
    # 4 GB kernel cap: np.load(mmap) page-cache reads count against kernel RSS
    # (~10 GB on snow2018 batches); seek+fromfile reads the same bytes at ~4 GB.
    with open(bf, 'rb') as fh:
        ver = np.lib.format.read_magic(fh)
        shape, fortran, dtype = np.lib.format._read_array_header(fh, ver)
        off = fh.tell()
        rowbytes = int(np.prod(shape[1:])) * dtype.itemsize
        for s0 in range(0, shape[0], slab_events):
            n = min(slab_events, shape[0] - s0)
            fh.seek(off + s0 * rowbytes)
            arr = np.fromfile(fh, dtype=dtype, count=n * int(np.prod(shape[1:])))
            yield s0, arr.reshape(n, *shape[1:]).astype('float32')

# Map the nuclei string representations to their atomic weights.
comp_conversion = {'p':1, 'h':4, 'o':16, 'f':56}

# Training container image. The venv's keras cannot load container-era .keras files,
# so each model is converted to .h5 once, inside the container.
SIF_PATH = '/data/user/fmcnally/icetop-cnn/tf.sif'


def ensure_h5(models_folder_path, model_name):
    '''Converts a container-era .keras model to .h5 so this venv can load it.'''
    keras_path = os.path.join(models_folder_path, model_name, model_name + '.keras')
    h5_path = os.path.join(models_folder_path, model_name, model_name + '.h5')
    # shadow a fresh keras (burned us: b.h5 label-bug era, b.keras fixed era).
    if (not os.path.exists(h5_path)
            or os.path.getmtime(h5_path) < os.path.getmtime(keras_path)):
        code = 'import tensorflow as tf; m = tf.keras.models.load_model(%r); m.save(%r)' % (keras_path, h5_path)
        subprocess.run(['/usr/bin/apptainer', 'exec', '--bind', '/data/user', SIF_PATH, 'python3', '-c', code], check=True)
    return h5_path


def _ensure_onnx(models_folder_path, model_name):
    # Converts the model .h5 to .onnx once (via tf2onnx) so predict can run on
    # onnxruntime (2s import) instead of tensorflow (300s import on cold FS).
    onnx = os.path.join(models_folder_path, model_name, model_name + '.onnx')
    if os.path.exists(onnx):
        return onnx
    # Imported here rather than at module scope so that a run which already has
    # an .onnx file never pays the TensorFlow import.
    import tensorflow as tf
    import tf2onnx
    h5 = ensure_h5(models_folder_path, model_name)
    m = tf.keras.models.load_model(h5, compile=False)
    spec = (tf.TensorSpec((None, 10, 10, 4), tf.float32, name='icetop'),)
    tf2onnx.convert.from_keras(m, input_signature=spec, opset=13, output_path=onnx)
    return onnx


def _ort_session(models_folder_path, model_name):
    import onnxruntime as ort
    return ort.InferenceSession(_ensure_onnx(models_folder_path, model_name),
                                providers=['CPUExecutionProvider'])


def predict(models_folder_path, model_parameters, model_name, dataset, model=None):
    '''Predicts a model over a dataset's assessment split. Same seeded cut for every model.
    Loads one batch file at a time so full-dataset detector arrays never sit in memory.
    Pass model= (a loaded keras model) to skip the TF import + load on batch calls.'''
    is_ort = hasattr(model, 'run') if model is not None else False
    if model is None:
        import tensorflow as tf
    prep = model_parameters[model_name]
    if model is None:
        model = tf.keras.models.load_model(ensure_h5(models_folder_path, model_name), compile=False)
    path = dataset if os.path.isabs(dataset) else os.path.join(os.getenv('ICETOP_CNN_DATA_DIR') or '/data/user/jstowers/icetop-cnn', 'simdata', dataset)
    ep = get_event_parameters(path, composition='phof')
    cut = get_training_assessment_cut(ep, 'assessment', prep)
    # Match training-time recons: predict only on the model's assessment nuclei.
    nuclei = prep.get('assessment_nuclei') or list(dict.fromkeys(prep.get('training_nuclei', 'phof')))
    cut = cut * np.isin(ep['comp'], [comp_conversion[n] for n in nuclei])
    batch_files = files(path, 'icetop', composition='phof')
    infill_files = None
    if prep['infill']:
        infill_files = files(path, 'infill', composition='phof')
    out, n = [], 0
    print(f'Predicting {model_name} on {dataset}: {len(batch_files)} files...')
    for i, bf in enumerate(batch_files):
        # 4 GB kernel cap - never materialize a batch. mmap slice ->
        # prep -> predict -> free, 100k slabs (~1 GB transient). 50k if it still OOMs.
        with open(bf, 'rb') as fh:
            ver = np.lib.format.read_magic(fh)
            shape, _, _ = np.lib.format._read_array_header(fh, ver)
            total = shape[0]
        file_cut = cut[n:n + total]
        infill_slabs = _iter_slabs(infill_files[i], 100_000) if infill_files is not None else None
        k = 0
        for s0, icetop_slab in _iter_slabs(bf, 100_000):
            d = {'icetop': icetop_slab}
            if infill_slabs is not None:
                _, infill_slab = next(infill_slabs)
                d['infill'] = infill_slab
            d = data_prep(d, prep)
            scut = file_cut[s0:s0 + len(d['icetop'])]
            if scut.any():
                inputs = [data[scut] for data in d.values()]
                if is_ort:
                    onames = [i.name for i in model.get_inputs()]
                    out.append(model.run(None, dict(zip(onames, inputs)))[0])
                else:
                    names = list(model.get_config()['input_layers'].keys())
                    out.append(model.predict(dict(zip(names, inputs)), batch_size=4096, verbose=0))
            k += 1
        print(f'  {model_name} [{dataset}]: file {i+1}/{len(batch_files)} done', flush=True)
        n += total
    assert n == len(ep['comp']), 'File/event count mismatch: %s on %s' % (model_name, dataset)
    return np.concatenate(out)


def model_weights_exist(models_folder_path, model_name):
    '''True when saved weights (.h5 or .keras) exist for a model.'''
    for ext in ('.h5', '.keras'):
        if os.path.exists(os.path.join(models_folder_path, model_name, model_name + ext)):
            return True
    return False


def assess_models(model_parameters, DATASETS, eps, models_folder_path, reconstructions_folder_path):
    '''Returns (reconstructions, cuts), each keyed by (model, dataset) pairs.
    Cache-first: existing cross predictions and training-time reconstructions are loaded,
    missing pairs are predicted and cached.'''
    reconstructions, cuts = {}, {}
    cut_cache = {}
    # ponytail: 1-model LRU - 18 sessions x ~0.2 GB OOMs the 4 GB kernel;
    # reload per model ~1-2 s. functools.lru_cache needs hashables, dict+del is simpler.
    loaded_models = {}
    for model_name, dataset in [(m, d) for m in model_parameters for d in DATASETS]:
        # Skip models with no saved weights (e.g. still training) instead of
        # crashing the whole assessment inside the apptainer conversion step.
        if not model_weights_exist(models_folder_path, model_name):
            print(f'WARNING: No saved weights for {model_name} (training in progress?); skipping model.')
            continue
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
            if model_name not in loaded_models:
                loaded_models.clear()  # keep <=1 session resident
                try:
                    loaded_models[model_name] = _ort_session(models_folder_path, model_name)
                except Exception:
                    import tensorflow as tf
                    loaded_models[model_name] = tf.keras.models.load_model(
                        ensure_h5(models_folder_path, model_name), compile=False)
            reconstruction = predict(models_folder_path, model_parameters, model_name, dataset,
                                     model=loaded_models[model_name])
            np.save(cache, reconstruction)

        # Seeds of one model share prep, so key the cut on (dataset, prep)
        # and compute once per unique prep instead of once per pair.
        cut_key = (dataset, repr(sorted(model_prep.items(), key=lambda kv: str(kv[0]))))
        if cut_key not in cut_cache:
            cut_cache[cut_key] = get_training_assessment_cut(ep, 'assessment', model_prep)
        model_cut = cut_cache[cut_key]

        # Apply the nuclei cut, then check alignment: recon rows must match cut events.
        nuclei_to_assess = [comp_conversion[nuclei] for nuclei in
                            (model_prep.get('assessment_nuclei')
                             or list(dict.fromkeys(model_prep.get('training_nuclei', 'phof'))))]
        model_cut = np.isin(ep['comp'], nuclei_to_assess) * model_cut
        assert len(reconstruction) == np.sum(model_cut), \
            'Reconstruction/event misalignment: %s on %s' % (model_name, dataset)

        # Flatten the predictions into a one-dimensional array and save them.
        reconstruction = reconstruction.flatten()
        reconstructions[(model_name, dataset)] = reconstruction

        # Save the model-specific events cut.
        cuts[(model_name, dataset)] = model_cut
    return reconstructions, cuts


def prebuild_onnx(models_folder_path):
    '''Converts every model to .onnx in one pass. Run once, offline: after this the
    notebooks only import onnxruntime (~2 s) and never touch TensorFlow.'''
    for model_name in sorted(os.listdir(models_folder_path)):
        if not model_weights_exist(models_folder_path, model_name):
            continue
        print('%s -> %s' % (model_name, _ensure_onnx(models_folder_path, model_name)), flush=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'prebuild':
        prebuild_onnx(sys.argv[2] if len(sys.argv) > 2 else
                      os.path.join(os.getenv('ICETOP_CNN_DATA_DIR') or '/data/user/jstowers/icetop-cnn', 'models'))
    else:
        print('Imported by the assessment notebooks; nothing to run standalone.')
