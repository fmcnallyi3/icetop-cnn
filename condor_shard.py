#!/usr/bin/env python3
"""Condor shard: assess ONE (model, dataset) pair through the normal assessment path.

Model params + ep come from JSON/npz transferred alongside (written by condor_assess).
Writes the recon cache, which the notebook's assess_models then loads.
"""
import json
import os
import sys


def main(model_name, dataset, models_folder, recon_folder):
    sys.path.insert(0, os.getcwd())
    import numpy as np
    from assessment import assess_models

    inputs_dir = os.getcwd()
    model_parameters = json.load(open(os.path.join(inputs_dir, 'assess_params.json')))
    ep = dict(np.load(os.path.join(inputs_dir, 'assess_%s_ep.npz' % dataset), allow_pickle=True))
    reconstructions, cuts = assess_models(
        {model_name: model_parameters[model_name]}, [dataset], {dataset: ep},
        models_folder, recon_folder)
    n = len(reconstructions[(model_name, dataset)])
    print('shard done: %d recons for %s on %s' % (n, model_name, dataset))
    assert n > 0, 'shard produced an empty reconstruction'


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'SELFTEST':
        sys.path.insert(0, os.getcwd())
        import assessment  # noqa: F401  - import check only
        print('shard imports OK')
    else:
        main(*sys.argv[1:5])
