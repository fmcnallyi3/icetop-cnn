"""condor_assess.py: assess cold (model, dataset) pairs on condor from the notebook.

Cell usage (before the assess_models call):
    from condor_assess import assess_on_condor
    assess_on_condor(model_parameters, DATASETS, eps,
                     MODELS_FOLDER_PATH, RECONSTRUCTIONS_FOLDER_PATH)
    reconstructions, cuts = assess_models(model_parameters, DATASETS, eps,
                                          MODELS_FOLDER_PATH, RECONSTRUCTIONS_FOLDER_PATH)

The first call submits one condor job per cold pair (they run in parallel), then
BLOCKS on condor_wait until every cluster terminates - the cell returning IS the
notification. The second call loads everything from cache (fast).

One condor cluster per pair; condor_wait polls the shared event log per pair.
"""
import getpass
import json
import os
import re
import subprocess
import time

import numpy as np

SUBMIT_HOST = 'npx-submitter'
ROOT = os.getenv('ICETOP_CNN_DIR') or os.path.dirname(os.path.abspath(__file__))
_SCRATCH = os.getenv('ICETOP_CNN_SCRATCH_DIR') or f'/scratch/{getpass.getuser()}/icetop-cnn'
LOG_DIR = os.path.join(_SCRATCH, 'assess')

_SUB = '''Universe   = container
Executable = {root}/condor_shard.py
Arguments  = {model} {dataset} {models} {recon}
Log        = {logdir}/{model}__{dataset}.condor.log
Output     = {model}__{dataset}.out
Error      = {model}__{dataset}.err
container_image = /data/user/fmcnally/icetop-cnn/tf.sif
environment = "PATH=/usr/bin:/bin ICETOP_CNN_DATA_DIR=/data/user/jstowers/icetop-cnn"
getenv = True
initialdir = {root}
transfer_input_files = assessment.py,utils.py,condor_shard.py,condor/assess/assess_params.json,condor/assess/assess_{dataset}_ep.npz
request_memory = 8G
request_cpus = 2
requirements = HasSingularity
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
Queue 1
'''


def _ssh(cmd, timeout=120):
    r = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', SUBMIT_HOST, cmd],
        capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError('ssh %s failed: %s' % (SUBMIT_HOST, r.stderr.strip()))
    return r.stdout


def assess_on_condor(model_parameters, DATASETS, eps, models_folder_path,
                     reconstructions_folder_path):
    from assessment import model_weights_exist

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

    # Stage the shared inputs in initialdir (root on the submit host):
    params = {m: model_parameters[m] for m, _ in cold}
    with open(os.path.join(_rt, 'assess_params.json'), 'w') as f:
        json.dump(params, f)
    for d in sorted({d for _, d in cold}):
        np.savez(os.path.join(_rt, 'assess_%s_ep.npz' % d), **eps[d])

    clusters = {}
    for m, d in cold:
        sub = os.path.join(_rt, 'assess_%s__%s.sub' % (m, d))
        with open(sub, 'w') as f:
            f.write(_SUB.format(root=ROOT, model=m, dataset=d,
                                models=models_folder_path,
                                recon=reconstructions_folder_path,
                                logdir=LOG_DIR))
        out = _ssh('condor_submit %s' % sub)
        cid = re.findall(r'\d{6,}', out)[-1]
        clusters[(m, d)] = cid
        print('  submitted %s on %s: cluster %s' % (m, d, cid), flush=True)

    for (m, d), cid in clusters.items():
        log = '%s/%s__%s.condor.log' % (LOG_DIR, m, d)
        while True:
            try:
                out = _ssh('condor_wait %s' % log, timeout=3600)
            except subprocess.TimeoutExpired:
                continue  # pair still running past the ssh window; re-attach
            if 'All jobs' in out:
                break
            time.sleep(15)
        cache = os.path.join(reconstructions_folder_path, '%s__%s.npy' % (m, d))
        if os.path.exists(cache):
            print('  %s on %s: cached (cluster %s)' % (m, d, cid), flush=True)
        else:
            err = _ssh('strings %s/%s__%s.err | tail -8' % (ROOT, m, d))
            raise RuntimeError('shard %s on %s finished without a cache:\n%s'
                               % (m, d, err))
    print('assessment complete - assess_models now loads warm caches')
