# ASSESSMENT FILE
# Loads or generates reconstructions for every (model, dataset) pair.
# Called by the assessment notebooks; cache-checking happens here.

import getpass
import os
import subprocess

import numpy as np
try:
    import plots  # sets the notebook-wide color cycle
except ImportError:
    pass

from utils import data_prep, files, get_event_parameters, get_training_assessment_cut, iter_slabs, load_detector_array


# Map the nuclei string representations to their atomic weights.
comp_conversion = {'p':1, 'h':4, 'o':16, 'f':56}

# Training container image. The venv's keras cannot load container-era .keras files,
# so each model is converted to .h5 once, inside the container.
SIF_PATH = '/data/user/fmcnally/icetop-cnn/tf.sif'
_DATA_DIR = os.getenv('ICETOP_CNN_DATA_DIR') or f'/data/user/{getpass.getuser()}/icetop-cnn'


def _is_fresh(target_path, source_path):
    return os.path.exists(target_path) and (
        not os.path.exists(source_path)
        or os.path.getmtime(target_path) >= os.path.getmtime(source_path)
    )


def ensure_h5(models_folder_path, model_name):
    '''Converts a container-era .keras model to .h5 so this venv can load it.'''
    keras_path = os.path.join(models_folder_path, model_name, model_name + '.keras')
    h5_path = os.path.join(models_folder_path, model_name, model_name + '.h5')
    # shadow a fresh keras (burned us: b.h5 label-bug era, b.keras fixed era).
    if _is_fresh(h5_path, keras_path):
        return h5_path
    if not os.path.exists(keras_path):
        raise FileNotFoundError(f'Missing model weights: neither {h5_path} nor {keras_path} exists.')
    if not os.path.exists('/usr/bin/apptainer'):
        raise RuntimeError(
            f'Missing or stale .h5 ({h5_path}) and /usr/bin/apptainer is unavailable '
            f'to convert {keras_path}. Prebuild .h5/.onnx on cobalt first.'
        )
    code = 'import tensorflow as tf; m = tf.keras.models.load_model(%r); m.save(%r)' % (keras_path, h5_path)
    subprocess.run(['/usr/bin/apptainer', 'exec', '--bind', '/data/user', SIF_PATH, 'python3', '-c', code], check=True)
    return h5_path


def _keras_input_signature(model, tf):
    '''Derives tf.TensorSpec input_signature dynamically from a Keras model.'''
    cfg_inputs = model.get_config().get('input_layers')
    if isinstance(cfg_inputs, dict):
        by_layer = {t.name.split(':')[0]: t for t in model.inputs}
        specs = {}
        for idx, (in_name, layer_ref) in enumerate(cfg_inputs.items()):
            layer_name = layer_ref[0] if isinstance(layer_ref, (list, tuple)) else str(layer_ref)
            t = by_layer.get(layer_name, model.inputs[idx])
            specs[in_name] = tf.TensorSpec((None, *t.shape[1:]), tf.as_dtype(t.dtype), name=in_name)
        return (specs,)
    if isinstance(cfg_inputs, list) and len(cfg_inputs) == len(model.inputs):
        return tuple(
            tf.TensorSpec((None, *t.shape[1:]), tf.as_dtype(t.dtype),
                          name=(ref[0] if isinstance(ref, (list, tuple)) else t.name.split(':')[0]))
            for ref, t in zip(cfg_inputs, model.inputs)
        )
    return tuple(
        tf.TensorSpec((None, *t.shape[1:]), tf.as_dtype(t.dtype), name=t.name.split(':')[0])
        for t in model.inputs
    )


def _ensure_onnx(models_folder_path, model_name):
    # Converts the model .h5 to .onnx once (via tf2onnx) so predict can run on
    # onnxruntime (2s import) instead of tensorflow (300s import on cold FS).
    keras_path = os.path.join(models_folder_path, model_name, model_name + '.keras')
    onnx = os.path.join(models_folder_path, model_name, model_name + '.onnx')
    if _is_fresh(onnx, keras_path):
        return onnx
    # Imported here rather than at module scope so that a run which already has
    # an .onnx file never pays the TensorFlow import.
    try:
        h5 = ensure_h5(models_folder_path, model_name)
    except Exception as err:
        err._from_ensure_h5 = True
        raise
    import tensorflow as tf
    import tf2onnx
    m = tf.keras.models.load_model(h5, compile=False)
    spec = _keras_input_signature(m, tf)
    tf2onnx.convert.from_keras(m, input_signature=spec, opset=13, output_path=onnx)
    return onnx


def _ort_session(models_folder_path, model_name):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    # The default is the host core count, which thrashes these small batches:
    # 60.8 s vs 7.0 s for the same work on a 64-core node.
    opts.intra_op_num_threads = max(1, min(4, len(os.sched_getaffinity(0))))
    opts.inter_op_num_threads = 1
    # Arenas hold ~3.5 GB of freed scratch across 8 sessions; turning them off
    # costs no measurable speed and keeps the notebook kernel under its cap.
    opts.enable_cpu_mem_arena = False
    return ort.InferenceSession(_ensure_onnx(models_folder_path, model_name), opts,
                                providers=['CPUExecutionProvider'])


def _load_model(models_folder_path, model_name):
    '''Loads an onnxruntime session, else a keras model. One session is ~13 MB.'''
    try:
        return _ort_session(models_folder_path, model_name)
    except Exception as err:
        if getattr(err, '_from_ensure_h5', False):
            raise
        h5_path = os.path.join(models_folder_path, model_name, model_name + '.h5')
        keras_path = os.path.join(models_folder_path, model_name, model_name + '.keras')
        if not _is_fresh(h5_path, keras_path):
            h5_path = ensure_h5(models_folder_path, model_name)
        import tensorflow as tf
        return tf.keras.models.load_model(h5_path, compile=False)


def _run_model(model, inputs, ort_batch_size=2048):
    '''Feeds one event batch to a session or a keras model and returns its output.'''
    if hasattr(model, 'run'):
        names = [i.name for i in model.get_inputs()]
        n_outs = len(model.get_outputs())
        n = len(inputs[0])
        if n == 0:
            empty = [np.empty((0,), dtype=np.float32) for _ in range(n_outs)]
            return empty[0] if n_outs == 1 else empty
        chunks = [[] for _ in range(n_outs)]
        for start in range(0, n, ort_batch_size):
            end = min(start + ort_batch_size, n)
            feed = {name: arr[start:end] for name, arr in zip(names, inputs)}
            res = model.run(None, feed)
            for idx, r in enumerate(res):
                chunks[idx].append(r)
        outs = [np.concatenate(c, axis=0) if len(c) > 1 else c[0] for c in chunks]
        return outs[0] if n_outs == 1 else outs
    cfg_inputs = model.get_config().get('input_layers')
    if isinstance(cfg_inputs, dict):
        feed = dict(zip(cfg_inputs.keys(), inputs))
    else:
        feed = inputs[0] if len(inputs) == 1 else inputs
    res = model.predict(feed, batch_size=4096, verbose=0)
    if isinstance(res, list) and len(res) == 1:
        return res[0]
    return res


def _dataset_path(dataset):
    if os.path.isabs(dataset):
        return dataset
    return os.path.join(_DATA_DIR,
                        'simdata', dataset)


def _model_cut(ep, prep):
    '''Seeded assessment cut for one prep, narrowed to the model's assessment nuclei.'''
    cut = get_training_assessment_cut(ep, 'assessment', prep)
    nuclei = (prep.get('assessment_nuclei')
              or list(dict.fromkeys(prep.get('training_nuclei', 'phof'))))
    return np.isin(ep['comp'], [comp_conversion[n] for n in nuclei]) * cut


def _reco_input(ep, prep):
    '''Extracts optional normalized zenith input (N, 1) float32 when prep["reco"] is set.'''
    reco = prep.get('reco')
    if not reco:
        return None
    zenith = np.pi - ep[f'{reco}_dir'].transpose()[0].astype('float32')
    if prep.get('normed'):
        zenith /= np.nanmax(zenith)
    return zenith[:, None]


# Prep fields that change the cut or the data source but not the prepared arrays.
# Models differing only in these still share one pass over the dataset.
_CUT_ONLY_KEYS = {'seed', 'simdata', 'training_nuclei', 'assessment_nuclei'}


def _prep_key(prep):
    '''Identifies the prepared arrays a prep produces, ignoring seed and source.'''
    return repr(sorted((k, v) for k, v in prep.items() if k not in _CUT_ONLY_KEYS))


def predict(models_folder_path, model_parameters, model_name, dataset, model=None):
    '''Predicts one model over a dataset's assessment split.
    Loads one batch file at a time so full-dataset detector arrays never sit in memory.
    Pass model= (a loaded model) to skip the TF import + load on batch calls.
    assess_models does not use this path; it streams every model over the data at once.'''
    prep = model_parameters[model_name]
    if model is None:
        model = _load_model(models_folder_path, model_name)
    path = _dataset_path(dataset)
    ep = get_event_parameters(path, composition='phof')
    cut = _model_cut(ep, prep)
    reco_arr = _reco_input(ep, prep)
    batch_files = files(path, 'icetop', composition='phof')
    infill_files = files(path, 'infill', composition='phof') if prep['infill'] else None
    out, n = [], 0
    print(f'Predicting {model_name} on {dataset}: {len(batch_files)} files...')
    for i, bf in enumerate(batch_files):
        with open(bf, 'rb') as fh:
            ver = np.lib.format.read_magic(fh)
            shape, _, _ = np.lib.format._read_array_header(fh, ver)
            total = shape[0]
        file_cut = cut[n:n + total]
        infill_slabs = iter_slabs(infill_files[i], 100_000) if infill_files is not None else None
        for s0, icetop_slab in iter_slabs(bf, 100_000):
            d = {'icetop': icetop_slab}
            if infill_slabs is not None:
                _, infill_slab = next(infill_slabs)
                d['infill'] = infill_slab
            d = data_prep(d, prep)
            m = len(d['icetop'])
            scut = file_cut[s0:s0 + m]
            if scut.any():
                inputs = [data[scut] for data in d.values()]
                if reco_arr is not None:
                    inputs.append(reco_arr[n + s0:n + s0 + m][scut])
                out.append(_run_model(model, inputs))
        print(f'  {model_name} [{dataset}]: file {i+1}/{len(batch_files)} done', flush=True)
        n += total
    assert n == len(ep['comp']), 'File/event count mismatch: %s on %s' % (model_name, dataset)
    if not out:
        n_outs = len(model.get_outputs()) if hasattr(model, 'get_outputs') else len(model.outputs)
        empty = [np.empty((0,), dtype=np.float32) for _ in range(n_outs)]
        return empty[0] if n_outs == 1 else empty
    if isinstance(out[0], list):
        return [np.concatenate([c[i] for c in out], axis=0) for i in range(len(out[0]))]
    return np.concatenate(out, axis=0)


def model_weights_exist(models_folder_path, model_name):
    '''True when saved weights (.h5 or .keras) exist for a model.'''
    for ext in ('.h5', '.keras'):
        if os.path.exists(os.path.join(models_folder_path, model_name, model_name + ext)):
            return True
    return False


def _predict_group(dataset_path, dataset, prep, group, models_folder_path, ep=None):
    '''Streams one dataset once and runs every model in the group on the same slabs.

    group is a list of (model_name, model_cut). Returns {model_name: recon} with each
    recon shaped (n, 1), matching the cache format. N models cost one read of the
    dataset instead of N reads, and models sharing an identical cut reuse sliced slabs.
    '''
    batch_files = files(dataset_path, 'icetop', composition='phof')
    infill_files = files(dataset_path, 'infill', composition='phof') if prep['infill'] else None
    reco_arr = None
    if prep.get('reco'):
        if ep is None:
            ep = get_event_parameters(dataset_path, composition='phof')
        reco_arr = _reco_input(ep, prep)
    models = {model_name: _load_model(models_folder_path, model_name) for model_name, _ in group}
    out = {model_name: [] for model_name, _ in group}

    print(f'Predicting {len(group)} model(s) on {dataset}: {len(batch_files)} files...')
    n = 0
    for i, bf in enumerate(batch_files):
        with open(bf, 'rb') as fh:
            ver = np.lib.format.read_magic(fh)
            shape, _, _ = np.lib.format._read_array_header(fh, ver)
            total = shape[0]
        infill_slabs = iter_slabs(infill_files[i], 100_000) if infill_files is not None else None
        for s0, icetop_slab in iter_slabs(bf, 100_000):
            d = {'icetop': icetop_slab}
            if infill_slabs is not None:
                _, infill_slab = next(infill_slabs)
                d['infill'] = infill_slab
            d = data_prep(d, prep)
            m = len(d['icetop'])
            sliced_cache = {}
            for model_name, cut in group:
                cut_id = id(cut)
                if cut_id not in sliced_cache:
                    scut = cut[n + s0:n + s0 + m]
                    if not scut.any():
                        sliced_cache[cut_id] = None
                    else:
                        inputs = [v[scut] for v in d.values()]
                        if reco_arr is not None:
                            inputs.append(reco_arr[n + s0:n + s0 + m][scut])
                        sliced_cache[cut_id] = inputs
                inputs = sliced_cache[cut_id]
                if inputs is None:
                    continue
                out[model_name].append(_run_model(models[model_name], inputs))
        print(f'  [{dataset}] file {i+1}/{len(batch_files)} done', flush=True)
        n += total

    # One row per cut event, same alignment check as the single-model path.
    for model_name, model_cut in group:
        chunks = out[model_name]
        if not chunks:
            out[model_name] = np.empty((0,), dtype=np.float32)
        elif isinstance(chunks[0], list):
            out[model_name] = [np.concatenate([c[idx] for c in chunks], axis=0) for idx in range(len(chunks[0]))]
        else:
            out[model_name] = np.concatenate(chunks, axis=0)
        recon_len = len(out[model_name][0]) if isinstance(out[model_name], list) else len(out[model_name])
        assert recon_len == np.sum(model_cut), \
            'Reconstruction/event misalignment: %s on %s' % (model_name, dataset)
    return out


def assess_models(model_parameters, DATASETS, eps, models_folder_path, reconstructions_folder_path):
    '''Returns (reconstructions, cuts), each keyed by (model, dataset) pairs.
    Cache-first: existing cross predictions and training-time reconstructions are loaded,
    missing pairs are predicted and cached. Each dataset is read once and every pending
    model runs on the same slabs, so N models cost one pass over the data.'''
    reconstructions, cuts = {}, {}
    cut_cache = {}

    for dataset in DATASETS:
        ep = eps[dataset]
        dataset_path = _dataset_path(dataset)

        # Resolve every model first: cache hit, training-time recon, or pending a pass.
        groups = {}
        for model_name, model_prep in model_parameters.items():
            # Skip models with no saved weights (e.g. still training) instead of
            # crashing the whole assessment inside the apptainer conversion step.
            if not model_weights_exist(models_folder_path, model_name):
                print(f'WARNING: No saved weights for {model_name} (training in progress?); skipping model.')
                continue

            # Seeds and identical-cut models share the exact same cut ndarray object.
            nuclei = (model_prep.get('assessment_nuclei')
                      or list(dict.fromkeys(model_prep.get('training_nuclei', 'phof'))))
            cut_key = (dataset, model_prep.get('sta5'), model_prep.get('reco'), tuple(nuclei))
            if cut_key not in cut_cache:
                cut_cache[cut_key] = _model_cut(ep, model_prep)
            model_cut = cut_cache[cut_key]
            cuts[(model_name, dataset)] = model_cut

            # Cached cross prediction, else the training-time reconstruction, else predict.
            cache = os.path.join(reconstructions_folder_path, '%s__%s.npy' % (model_name, dataset))
            own = model_prep.get('simdata', '')
            own_file = os.path.join(reconstructions_folder_path, model_name + '.npy')
            if os.path.exists(cache):
                reconstruction = np.load(cache)
            elif (os.path.basename(own.rstrip('/')) == dataset or own == dataset) and os.path.exists(own_file):
                reconstruction = np.load(own_file)
            else:
                # Same preparation means the same slabs, so batch these models into
                # one pass. Seeds change the cut only, not the prepared data.
                groups.setdefault(_prep_key(model_prep), []).append((model_name, model_cut))
                continue
            assert len(reconstruction) == np.sum(model_cut), \
                'Reconstruction/event misalignment: %s on %s' % (model_name, dataset)
            reconstructions[(model_name, dataset)] = reconstruction.flatten()

        # One streaming pass per distinct prep, all of its models sharing the slabs.
        for cut_key, group in groups.items():
            prep = model_parameters[group[0][0]]
            fresh = _predict_group(dataset_path, dataset, prep, group, models_folder_path, ep=ep)
            for model_name, _ in group:
                cache = os.path.join(reconstructions_folder_path, '%s__%s.npy' % (model_name, dataset))
                np.save(cache, fresh[model_name])
                reconstructions[(model_name, dataset)] = fresh[model_name].flatten()

    # Keep the model-outer order the notebooks iterate when printing stats.
    order = [(m, d) for m in model_parameters for d in DATASETS]
    return ({k: reconstructions[k] for k in order if k in reconstructions},
            {k: cuts[k] for k in order if k in cuts})


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
                      os.path.join(_DATA_DIR, 'models'))
    else:
        print('Imported by the assessment notebooks; nothing to run standalone.')
