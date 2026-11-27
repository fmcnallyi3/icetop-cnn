#!/usr/bin/env python3

import argparse
import gc
import getpass
import json
import os
import subprocess
import sys
from glob import glob

ERROR_ENVIRONMENT_NOT_ACTIVATED = 'Virtual environment not activated. Activate with the command "icetop-cnn"'

# Check that the environment has been activated
# Should only check when not run on the condor cluster
# Should check before importing external libraries
if not os.getenv('_CONDOR_SLOT'):
    assert os.getenv('VIRTUAL_ENV'), ERROR_ENVIRONMENT_NOT_ACTIVATED

# Supress debugging information
# Can remove safely, just makes for cleaner output
# Should be set before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Do not preallocate all GPU memory; big datasets OOM the copy to an
# eager constant otherwise (Dst tensor is not initialized).
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf

import config as cg
from utils import data_prep, files, get_event_parameters, get_training_assessment_cut, iter_slabs
from model import get_compiled_model

ICETOP_CNN_DIR = os.getenv('ICETOP_CNN_DIR') or os.path.dirname(os.path.abspath(__file__))
ICETOP_CNN_DATA_DIR = os.getenv('ICETOP_CNN_DATA_DIR') or f'/data/user/{os.environ.get("USER", getpass.getuser())}/icetop-cnn'

def main():
    '''Creates/restores and trains a model based on the settings located in "config"'''

    # Get datasets
    if args.test: print('Getting datasets...')
    (training_dataset, validation_dataset), input_shapes = get_training_datasets()

    # Get model
    if args.test: print('Fetching model...')
    model = create_or_restore_model(input_shapes)

    # Train the model
    if args.test: print('Training model...')
    model = train_model(model, training_dataset, validation_dataset)

    # The prepared detector cache (several GB) is only needed while training.
    del training_dataset, validation_dataset
    gc.collect()

    # Assess the model
    if args.test: print('Assessing model...')
    assess_model(model)
    

def get_training_datasets():
    """Prepares streaming tensorflow datasets used for training.

    The full detector array for multi-million event datasets does not fit in
    slot memory. Instead: event parameters load whole (small), and detector
    data streams one simdata batch file at a time. Cuts and data_prep are
    applied per file exactly as the whole-load path applies them, and a global
    index permutation preserves full-event shuffle, so training semantics are
    unchanged. Peak RAM is one batch file plus parameters.
    """

    simdata_folder_path = args.simdata

    # Event parameters: small, load whole. Cuts need global event order.
    event_parameters = get_event_parameters(simdata_folder_path, composition=args.composition, test=args.test)

    # One-hot encode the composition data if composition is to be predicted
    if 'comp' in args.predict:
        comp_lut = np.full((57, 4), -1, dtype=np.int64)
        comp_lut[1] = [1, 0, 0, 0]
        comp_lut[4] = [0, 1, 0, 0]
        comp_lut[16] = [0, 0, 1, 0]
        comp_lut[56] = [0, 0, 0, 1]
        comp_raw = np.asarray(event_parameters['comp'], dtype=np.int64)
        encoded = comp_lut[comp_raw]
        if (encoded < 0).any():
            raise ValueError(f'Unexpected composition values: {np.unique(comp_raw)}')
        event_parameters['comp'] = encoded

    # Get number of events from an arbitrary key
    num_events = event_parameters['file_info'].shape[0]

    # Use local RandomState(1148) matching get_training_assessment_cut so the
    # training/validation split is identical across seeds even if assessment cut
    # no longer mutates global np.random.
    training_assessment_cut = get_training_assessment_cut(event_parameters, 'training', cg.PREP)
    split_rng = np.random.RandomState(1148)
    split_rng.rand(num_events)
    training_cut = split_rng.rand(num_events) > cg.VALIDATION_SPLIT
    validation_cut = np.logical_not(training_cut)

    train_mask = training_cut & training_assessment_cut
    val_mask = validation_cut & training_assessment_cut
    train_rows = np.where(train_mask)[0]
    val_rows = np.where(val_mask)[0]

    # File layout: icetop and infill files share order and row counts
    batch_files = files(simdata_folder_path, 'icetop', composition=args.composition, test=args.test)
    infill_files = files(simdata_folder_path, 'infill', composition=args.composition, test=args.test) if cg.PREP['infill'] else None

    def _npy_info(filepath):
        with open(filepath, 'rb') as fh:
            version = np.lib.format.read_magic(fh)
            shape, fortran_order, dtype = np.lib.format._read_array_header(fh, version)
            assert not fortran_order, f'Fortran-order array not supported: {filepath}'
            return shape, dtype, fh.tell()

    def _read_slab(filepath, info, start, count):
        shape, file_dtype, offset = info
        row_elems = int(np.prod(shape[1:]))
        row_bytes = row_elems * file_dtype.itemsize
        with open(filepath, 'rb') as fh:
            fh.seek(offset + start * row_bytes)
            slab = np.fromfile(fh, dtype=file_dtype, count=count * row_elems)
        return slab.reshape(count, *shape[1:])

    # Row counts from .npy headers (avoids re-unpickling event_parameters)
    file_infos_icetop = [_npy_info(bf) for bf in batch_files]
    counts = [shape[0] for shape, _, _ in file_infos_icetop]
    assert sum(counts) == num_events, f'Row count mismatch: icetop {sum(counts)} vs event_parameters {num_events}'
    boundaries = np.cumsum([0] + counts)
    file_of_row = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    file_infos_infill = [_npy_info(inf) for inf in infill_files] if infill_files else None

    # Optional reco input (zenith) comes from parameters, already in RAM.
    # nanmax ignores failed-fit NaNs in uncut event_parameters.
    reco_inputs = {}
    if cg.PREP['reco']:
        zenith = np.pi - event_parameters[f"{cg.PREP['reco']}_dir"].transpose()[0].astype('float32')
        if cg.PREP['normed']:
            zenith /= np.nanmax(zenith)
        reco_inputs[cg.PREP['reco']] = zenith

    SUPER_BATCH = 8192
    SLAB_EVENTS = 50_000

    _kept_events = int(training_assessment_cut.sum())
    _footprint_gb = _kept_events * 800 / 1e9  # fp16 cache (kept rows only)
    _slot_gb = int(os.getenv('SLOT_GB', '0')) or None
    _ram_ok = bool(_slot_gb and _footprint_gb < 0.6 * _slot_gb)
    print(f'detector cache: {"RAM" if _ram_ok else "streaming"} '
          f'({_footprint_gb:.1f} GB prepared vs {_slot_gb or "?"} GB slot)')

    def prep_file(filepath, detector_name, mask=None):
        '''Prepares one batch file slab by slab in file dtype, keeping only
        rows passing mask (if given) so cut events never sit in RAM.'''
        slabs = []
        for s0, slab in iter_slabs(filepath, SLAB_EVENTS, dtype=None):
            prepped = data_prep({detector_name: slab}, cg.PREP)[detector_name].astype('float16')
            if mask is not None:
                prepped = prepped[mask[s0:s0 + len(prepped)]]
            if len(prepped):
                slabs.append(prepped)
        if slabs:
            return np.concatenate(slabs)
        shape, sample_dtype, _ = _npy_info(filepath)
        empty = data_prep({detector_name: np.empty((0,) + shape[1:], dtype=sample_dtype)}, cg.PREP)[detector_name]
        return np.empty((0,) + empty.shape[1:], dtype='float16')

    detector_names = ['icetop'] + (['infill'] if infill_files else [])

    if _ram_ok:
        kept_local_of_row = np.empty(num_events, dtype=np.int32)
        detector_mmaps = {name: [] for name in detector_names}
        for fi, bf in enumerate(batch_files):
            f_mask = training_assessment_cut[boundaries[fi]:boundaries[fi + 1]]
            kept_local_of_row[boundaries[fi]:boundaries[fi + 1]] = np.cumsum(f_mask, dtype=np.int32) - 1
            detector_mmaps['icetop'].append(prep_file(bf, 'icetop', mask=f_mask))
            if infill_files:
                detector_mmaps['infill'].append(prep_file(infill_files[fi], 'infill', mask=f_mask))

        def load_rows(rows):
            """Gather detector rows for global indices, writing each file block straight
            into its output rows: one pass, no concatenate or reorder temporaries."""
            file_ids = file_of_row[rows]
            by_file = np.argsort(file_ids, kind='stable')
            sorted_fids = file_ids[by_file]
            uniq_fids, split_idx = np.unique(sorted_fids, return_index=True)
            split_ends = np.append(split_idx[1:], len(sorted_fids))

            out = {name: np.empty((len(rows),) + detector_mmaps[name][0].shape[1:], dtype='float32')
                   for name in detector_names}
            for k, fi in enumerate(uniq_fids):
                pos = by_file[split_idx[k]:split_ends[k]]
                local = kept_local_of_row[rows[pos]]
                for name in detector_names:
                    out[name][pos] = detector_mmaps[name][fi][local]

            out.update({name: arr[rows] for name, arr in reco_inputs.items()})
            return out

        # Alignment self-check vs naive per-row gather
        if len(train_rows):
            _chk_n = min(512, len(train_rows))
            _chk_rows = np.random.default_rng(0).choice(train_rows, size=_chk_n, replace=False)
            _fast_chk = load_rows(_chk_rows)
            for name in detector_names:
                _naive = np.stack([detector_mmaps[name][file_of_row[r]][kept_local_of_row[r]].astype('float32') for r in _chk_rows])
                assert np.array_equal(_fast_chk[name], _naive), f'Alignment check failed for {name}'
            for name, arr in reco_inputs.items():
                assert np.array_equal(_fast_chk[name], arr[_chk_rows]), f'Alignment check failed for {name}'

        def training_gen():
            order = np.random.permutation(train_rows)
            for s0 in range(0, len(order), SUPER_BATCH):
                b = order[s0:s0 + SUPER_BATCH]
                inputs = load_rows(b)
                outputs = [event_parameters[output][b] for output in args.predict]
                yield (tuple(inputs.values()), tuple(outputs))

        def validation_gen():
            for s0 in range(0, len(val_rows), SUPER_BATCH):
                b = val_rows[s0:s0 + SUPER_BATCH]
                inputs = load_rows(b)
                outputs = [event_parameters[output][b] for output in args.predict]
                yield (tuple(inputs.values()), tuple(outputs))

        probe_rows = train_rows[:min(cg.BATCH_SIZE, len(train_rows))] if len(train_rows) else val_rows[:cg.BATCH_SIZE]
        probe = load_rows(probe_rows)
        input_shapes = {name: arr.shape[1:] for name, arr in probe.items()}
        sig = (
            tuple(tf.TensorSpec((None,) + probe[name].shape[1:], probe[name].dtype) for name in probe),
            tuple(tf.TensorSpec((None,) + event_parameters[output].shape[1:], event_parameters[output].dtype) for output in args.predict),
        )
        return (
            tf.data.Dataset.from_generator(training_gen, output_signature=sig)
            .unbatch().batch(cg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE),
            tf.data.Dataset.from_generator(validation_gen, output_signature=sig)
            .unbatch().batch(cg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE),
        ), input_shapes

    # Streaming fallback: iterate/shuffle at slab granularity so every file slab
    # is read and prepped at most once per epoch, never per 32-event batch.
    train_slabs = []
    val_slabs = []
    for fi in range(len(batch_files)):
        b0 = boundaries[fi]
        for start in range(0, counts[fi], SLAB_EVENTS):
            count = min(SLAB_EVENTS, counts[fi] - start)
            g0 = b0 + start
            t_local = np.where(train_mask[g0:g0 + count])[0]
            if len(t_local):
                train_slabs.append((fi, start, count, t_local, g0 + t_local))
            v_local = np.where(val_mask[g0:g0 + count])[0]
            if len(v_local):
                val_slabs.append((fi, start, count, v_local, g0 + v_local))

    def _prep_slab(fi, start, count, local_idx):
        raw = {'icetop': _read_slab(batch_files[fi], file_infos_icetop[fi], start, count)}
        if infill_files:
            raw['infill'] = _read_slab(infill_files[fi], file_infos_infill[fi], start, count)
        prepped = data_prep(raw, cg.PREP)
        return {name: prepped[name][local_idx].astype('float32') for name in detector_names}

    # Alignment self-check on first training slab
    if train_slabs:
        fi0, s0, c0, l0, g0_idx = train_slabs[0]
        assert np.array_equal(boundaries[fi0] + s0 + l0, g0_idx)
        assert train_mask[g0_idx].all()

    def training_gen():
        for si in np.random.permutation(len(train_slabs)):
            fi, start, count, local_idx, global_idx = train_slabs[si]
            prepped = _prep_slab(fi, start, count, local_idx)
            perm = np.random.permutation(len(local_idx))
            for c0 in range(0, len(perm), SUPER_BATCH):
                p_slice = perm[c0:c0 + SUPER_BATCH]
                g_slice = global_idx[p_slice]
                inp = [prepped[name][p_slice] for name in detector_names]
                for arr in reco_inputs.values():
                    inp.append(arr[g_slice])
                outs = [event_parameters[output][g_slice] for output in args.predict]
                yield (tuple(inp), tuple(outs))

    def validation_gen():
        for fi, start, count, local_idx, global_idx in val_slabs:
            prepped = _prep_slab(fi, start, count, local_idx)
            for c0 in range(0, len(local_idx), SUPER_BATCH):
                c1 = c0 + SUPER_BATCH
                g_slice = global_idx[c0:c1]
                inp = [prepped[name][c0:c1] for name in detector_names]
                for arr in reco_inputs.values():
                    inp.append(arr[g_slice])
                outs = [event_parameters[output][g_slice] for output in args.predict]
                yield (tuple(inp), tuple(outs))

    probe_slab = train_slabs[0] if train_slabs else val_slabs[0]
    p_fi, p_start, p_count, p_local, p_global = probe_slab
    p_take = min(cg.BATCH_SIZE, len(p_local))
    probe = _prep_slab(p_fi, p_start, int(p_local[p_take - 1]) + 1, p_local[:p_take])
    for name, arr in reco_inputs.items():
        probe[name] = arr[p_global[:p_take]]
    input_shapes = {name: arr.shape[1:] for name, arr in probe.items()}
    sig = (
        tuple(tf.TensorSpec((None,) + probe[name].shape[1:], probe[name].dtype) for name in probe),
        tuple(tf.TensorSpec((None,) + event_parameters[output].shape[1:], event_parameters[output].dtype) for output in args.predict),
    )
    shuffle_buf = min(32768, max(1, len(train_rows)))
    return (
        tf.data.Dataset.from_generator(training_gen, output_signature=sig)
        .unbatch().shuffle(shuffle_buf, reshuffle_each_iteration=False)
        .batch(cg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE),
        tf.data.Dataset.from_generator(validation_gen, output_signature=sig)
        .unbatch().batch(cg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE),
    ), input_shapes


def create_or_restore_model(input_shapes):
    # Seed all RNGs so runs are reproducible; different --seed values give
    # genuinely independent models on the same dataset (study requirement).
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    '''Creates or restores a model based on the command line arguments passed in by the user'''

    # User indicated that they would like to restore the model.
    # The submission script should have already confirmed its existence.
    if args.restore:
        return tf.keras.models.load_model(
            os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name, f'{args.model_name}.keras')
        )
    # Otherwise, create and return a new model
    return get_compiled_model(input_shapes, args.model_name, args.model_design, cg.PREP, args.predict)


def train_model(model, training_dataset, validation_dataset):
    '''Trains the model using the provided training/validation datasets and settings from "config"'''

    # Create models folder if it does not already exist
    if not os.path.exists(os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name)):
        os.makedirs(os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name))

    # Save model parameters and nuclei
    with open(os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name, f'{args.model_name}.json'), 'w') as f:
        json.dump({**cg.PREP, 'training_nuclei':args.composition, 'simdata':args.simdata, **({'seed':args.seed} if args.seed is not None else {})}, f, indent=4)

    # Let 'er rip
    model.fit(
        training_dataset,
        epochs=args.epochs,
        validation_data=validation_dataset,
        callbacks=[
            tf.keras.callbacks.CSVLogger(
                os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name, f'{args.model_name}.csv'),
                append=args.restore
            ),
            tf.keras.callbacks.EarlyStopping(
                min_delta=cg.MIN_ES_DELTA,
                patience=cg.ES_PATIENCE,
                mode='min',
                restore_best_weights=cg.RESTORE_BEST_WEIGHTS
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=cg.LR_FACTOR,
                patience=cg.LR_PATIENCE,
                mode='min',
                min_delta=cg.MIN_LR_DELTA,
                cooldown=cg.LR_COOLDOWN,
                min_lr=cg.MIN_LR
            ),
        ]
    )
    tf.keras.models.save_model(
        model,
        os.path.join(ICETOP_CNN_DATA_DIR, 'models', args.model_name, f'{args.model_name}.keras'),
        save_format='keras'
    )
    return model


def assess_model(model: tf.keras.Model):
    '''Assesses the model once it has finished training.

    Streams the dataset one slab at a time. Loading it whole needs ~45 GB of
    float64 detector data, which gets the job held for exceeding slot memory.
    '''
    import assessment

    reconstructions = assessment.predict(
        os.path.join(ICETOP_CNN_DATA_DIR, 'models'),
        {args.model_name: {**cg.PREP, 'training_nuclei': args.composition}},
        args.model_name, args.simdata, model=model
    )

    # Save the reconstruction(s)
    for i, prediction in enumerate(args.predict):
        prediction_folder = os.path.join(ICETOP_CNN_DATA_DIR, 'reconstructions', prediction)
        os.makedirs(prediction_folder, exist_ok=True)
        if isinstance(reconstructions, dict):
            pred_arr = reconstructions[prediction]
        elif len(args.predict) == 1:
            pred_arr = reconstructions[0] if isinstance(reconstructions, (list, tuple)) else reconstructions
        else:
            pred_arr = reconstructions[i]
        np.save(os.path.join(prediction_folder, f'{args.model_name}.npy'), pred_arr)

    # Create loss graphs
    loss_grapher_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'loss_grapher.py')
    if not os.path.exists(loss_grapher_path):
        loss_grapher_path = os.path.join(ICETOP_CNN_DIR, 'loss_grapher.py')
    subprocess.run([sys.executable, loss_grapher_path, args.model_name], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description = 'Trains ML models using TensorFlow')
    p.add_argument(
        '-n', '--name', dest='model_name', type=str,
        required=True,
        help='Name of the model to train')
    p.add_argument(
        '-c', '--composition', type=str,
        required=True,
        help='Composition of datasets to load and train the model on')
    p.add_argument(
        '-e', '--epochs', type=int,
        default=cg.MAX_EPOCHS,
        help='The maximum number of epochs that the model should train for')
    _arch_dir = os.path.join(ICETOP_CNN_DIR, 'architectures')
    _arch_choices = sorted(os.path.splitext(arch)[0] for arch in glob('*.py', root_dir=_arch_dir)) if os.path.isdir(_arch_dir) else []
    p.add_argument(
        '-m', '--model', dest='model_design', type=str,
        choices=_arch_choices or None,
        required=True,
        help='Desired model architecture')
    p.add_argument(
        '-p', '--predict', nargs='+', type=str,
        choices=['comp', 'energy'],
        required=True,
        help='A list of one or more desired model outputs')
    p.add_argument(
        '--seed', dest='seed', type=int, default=None,
        help='Random seed; different values give independent models on the same dataset')
    p.add_argument(
        '-t', '--test', action='store_true',
        help='Run the script off the cluster with a limited dataset')
    p.add_argument(
        '-d', '--dataset', '--simdata', dest='simdata', default='simdata',
        help='Dataset name under $ICETOP_CNN_DATA_DIR/simdata, or a full path')
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        '-o', '--overwrite', action='store_true',
        help='Overwrite a model with the same name if it exists. Can not be used with restore')
    g.add_argument(
        '-r', '--restore', action='store_true',
        help='Attempt to restore and continue training a model if it exists. Can not be used with overwrite')
    args = p.parse_args()

    # Accept a bare dataset name or a full path; submit.py passes the path.
    if not os.path.isdir(args.simdata):
        args.simdata = os.path.join(ICETOP_CNN_DATA_DIR, 'simdata', args.simdata)
    assert os.path.isdir(args.simdata), f'Simdata folder not found: {args.simdata}'

    # Ensure that there are no unrecognized characters in the composition string
    if not all(c in 'phof' for c in args.composition):
        p.error('Unrecognized composition dataset combination requested')
    # Sort the composition string into a predictable order
    args.composition = ''.join(sorted(args.composition, key=lambda c: list('phof').index(c)))

    # Ensure epochs are a valid number
    if args.epochs and args.epochs <= 0:
        p.error('Epochs must be a positive integer')

    # Transform to a set to remove duplicates
    # Sort to maintain a predictable order to process arguments
    args.predict = sorted(set(args.predict))

    main()
