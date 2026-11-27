"""condor_assess.py: assess cold (model, dataset) pairs on condor from the notebook.

Cell usage (before the assess_models call):
    from condor_assess import assess_on_condor
    assess_on_condor(model_parameters, DATASETS, eps,
                     MODELS_FOLDER_PATH, RECONSTRUCTIONS_FOLDER_PATH)
    reconstructions, cuts = assess_models(model_parameters, DATASETS, eps,
                                          MODELS_FOLDER_PATH, RECONSTRUCTIONS_FOLDER_PATH)

Submits one condor job per cold (dataset, prep) group (they run in parallel and
stream each dataset once for all models in that group), then BLOCKS on
condor_wait until every cluster terminates. The second call loads caches (fast).
"""
import getpass
import json
import os
import re
import subprocess

import numpy as np

SUBMIT_HOST = 'npx-submitter'
ROOT = os.getenv('ICETOP_CNN_DIR') or os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.getenv('ICETOP_CNN_DATA_DIR') or f'/data/user/{getpass.getuser()}/icetop-cnn'
_SCRATCH = os.getenv('ICETOP_CNN_SCRATCH_DIR') or f'/scratch/{getpass.getuser()}/icetop-cnn'
LOG_DIR = os.path.join(_SCRATCH, 'assess')

_SUB = """Universe   = container
Executable = {root}/condor_shard.py
Arguments  = {dataset} {models} {recon} {model_args}
Log        = {log}
Output     = condor/assess/{tag}.out
Error      = condor/assess/{tag}.err
container_image = /data/user/fmcnally/icetop-cnn/tf.sif
environment = "PATH=/usr/bin:/bin ICETOP_CNN_DATA_DIR={datadir}"
initialdir = {root}
transfer_input_files = {transfer_files}
request_memory = 8G
request_cpus = 2
requirements = HasSingularity
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
Queue 1
"""


def _ssh(cmd, timeout=120):
    r = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', SUBMIT_HOST, cmd],
        capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError('ssh %s failed: %s' % (SUBMIT_HOST, r.stderr.strip()))
    return r.stdout


def _dataset_tag(dataset):
    return os.path.basename(dataset.rstrip('/')) or 'dataset'


def assess_on_condor(model_parameters, DATASETS, eps, models_folder_path,
                     reconstructions_folder_path):
    from assessment import _dataset_path, _prep_key, model_weights_exist

    cold = [(m, d) for m in model_parameters for d in DATASETS
            if not os.path.exists(os.path.join(reconstructions_folder_path,
                                               '%s__%s.npy' % (m, d)))
            and model_weights_exist(models_folder_path, m)]
    if not cold:
        print('all pairs cached - nothing to do')
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    _rt = os.path.join(ROOT, 'condor', 'assess')
    os.makedirs(_rt, exist_ok=True)

    # Clean stale shard stdout/stderr files at staging so failures never show old logs.
    for fname in os.listdir(_rt):
        if fname.endswith(('.out', '.err')):
            os.remove(os.path.join(_rt, fname))

    # Group cold pairs by (dataset, _prep_key) so each shard streams the dataset once
    # across all matching models (e.g. multiple seeds).
    groups = {}
    for m, d in cold:
        groups.setdefault((d, _prep_key(model_parameters[m])), []).append(m)

    staged_files = []
    params_file = os.path.join(_rt, 'assess_params.json')
    params = {m: model_parameters[m] for m, _ in cold}
    with open(params_file, 'w') as f:
        json.dump(params, f)
    staged_files.append(params_file)

    # Avoid writing/transferring huge ep npz files when /data/user is directly readable
    # inside the container; stage ep npz only when the dataset is not on /data/user.
    ep_staged = {}
    for d in sorted({d for d, _ in groups}):
        ds_path = _dataset_path(d)
        on_data_user = os.path.abspath(ds_path).startswith('/data/user/') and os.path.isdir(
            os.path.join(ds_path, 'event_parameters'))
        if not on_data_user:
            ds_tag = _dataset_tag(d)
            npz_path = os.path.join(_rt, 'assess_%s_ep.npz' % ds_tag)
            np.savez(npz_path, **eps[d])
            ep_staged[d] = 'condor/assess/assess_%s_ep.npz' % ds_tag
            staged_files.append(npz_path)

    try:
        clusters = []
        for idx, ((d, _pkey), shard_models) in enumerate(groups.items()):
            tag = '%s_g%d' % (_dataset_tag(d), idx)
            log = '%s/%s.condor.log' % (LOG_DIR, tag)
            _ssh('mkdir -p %s && rm -f %s' % (LOG_DIR, log))
            transfer = ['assessment.py', 'utils.py', 'condor_shard.py', 'condor/assess/assess_params.json']
            if d in ep_staged:
                transfer.append(ep_staged[d])
            sub = os.path.join(_rt, 'assess_%s.sub' % tag)
            with open(sub, 'w') as f:
                f.write(_SUB.format(
                    root=ROOT,
                    dataset=d,
                    models=models_folder_path,
                    recon=reconstructions_folder_path,
                    model_args=' '.join(shard_models),
                    log=log,
                    tag=tag,
                    datadir=_DATA_DIR,
                    transfer_files=','.join(transfer),
                ))
            staged_files.append(sub)
            out = _ssh('condor_submit %s' % sub)
            m_cid = re.search(r'submitted to cluster (\d+)', out)
            if not m_cid:
                raise RuntimeError('Unexpected condor_submit output: %s' % out.strip())
            cid = m_cid.group(1)
            clusters.append((tag, d, shard_models, log, cid))
            print('  submitted %d model(s) on %s: cluster %s' % (len(shard_models), d, cid), flush=True)

        for tag, d, shard_models, log, cid in clusters:
            while True:
                r = subprocess.run(
                    ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                     SUBMIT_HOST, 'condor_wait -wait 60 %s' % log],
                    capture_output=True, text=True, timeout=120)
                if r.returncode == 0 and 'All jobs' in r.stdout:
                    break
                q = _ssh('condor_q %s -af JobStatus HoldReason' % cid)
                for line in q.splitlines():
                    parts = line.split(None, 1)
                    if parts and parts[0] == '5':
                        reason = parts[1] if len(parts) > 1 else 'unknown'
                        raise RuntimeError('shard %s on %s (cluster %s) held: %s' % (tag, d, cid, reason))
                if r.returncode not in (0, 1):
                    raise RuntimeError('condor_wait failed for %s: %s' % (tag, r.stderr.strip()))

            missing = [
                m for m in shard_models
                if not os.path.exists(os.path.join(reconstructions_folder_path, '%s__%s.npy' % (m, d)))
            ]
            if missing:
                err = _ssh('strings %s/condor/assess/%s.err | tail -12' % (ROOT, tag))
                raise RuntimeError('shard %s on %s finished without caches for %s:\n%s'
                                   % (tag, d, missing, err))
            print('  %s (%d model(s)): cached (cluster %s)' % (d, len(shard_models), cid), flush=True)
    finally:
        for path in staged_files:
            if os.path.exists(path):
                os.remove(path)
    print('assessment complete - assess_models now loads warm caches')
