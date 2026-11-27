#!/usr/bin/env python3
"""Condor shard: assess one (dataset, prep) model group through the normal assessment path.

Model params come from assess_params.json transferred alongside (written by condor_assess).
Event parameters are read directly from /data/user when available, or from staged npz.
Writes the recon caches, which the notebook's assess_models then loads.
"""
import json
import os
import sys


def main(dataset, models_folder, recon_folder, *model_names):
    sys.path.insert(0, os.getcwd())
    import numpy as np
    from assessment import _dataset_path, assess_models
    from utils import get_event_parameters

    models_list = [m for arg in model_names for m in arg.split(',') if m]
    assert models_list, 'shard requires at least one model name'

    inputs_dir = os.getcwd()
    with open(os.path.join(inputs_dir, 'assess_params.json')) as f:
        model_parameters = json.load(f)

    ds_tag = os.path.basename(dataset.rstrip('/')) or 'dataset'
    ep_npz = os.path.join(inputs_dir, 'assess_%s_ep.npz' % ds_tag)
    if os.path.exists(ep_npz):
        ep = dict(np.load(ep_npz, allow_pickle=True))
    else:
        ep = get_event_parameters(_dataset_path(dataset), composition='phof')

    shard_params = {m: model_parameters[m] for m in models_list}
    reconstructions, _ = assess_models(
        shard_params, [dataset], {dataset: ep}, models_folder, recon_folder)

    for m in models_list:
        n = len(reconstructions[(m, dataset)])
        print('shard done: %d recons for %s on %s' % (n, m, dataset))
        assert n > 0, 'shard produced an empty reconstruction for %s' % m


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'SELFTEST':
        sys.path.insert(0, os.getcwd())
        import assessment  # noqa: F401  - import check only
        print('shard imports OK')
    else:
        main(*sys.argv[1:])
