# Data Extraction

Turn IceTop Level3 simulation files into the tensor arrays that `trainer.py` reads.

This folder is the `jstowers` branch work. `sim-extractor.py` is forked from
fmcnally's upstream; `sim-wrapper.py`, `sim-maker.py`, `sim-converter.py`,
`sim_utils.py`, and `pysubmit.py` were written or rewritten here.

## Pipeline

```
Level3 sim files (.i3.gz) + GCD
        │  sim-extractor.py      one condor job per batch, icetray env
        ▼
/data/user/$USER/icetop-cnn/sim/<name>/sim_<name>_<simset>_<batch>.npy
        │  sim-converter.py      one condor job, runs after extraction
        ▼
$ICETOP_CNN_DATA_DIR/simdata/<name>/
    icetop/icetop_<alias>_<simset>_<name>_<batch>.npy            (N,10,10,8) float32
    event_parameters/event_parameters_<alias>_<simset>_<name>_<batch>.npy
```

`sim-wrapper.py` drives the first two steps interactively. `sim-maker.py` and
`sim-converter.py` are the manual equivalents.

## Requirements

- A cobalt node with the icetop-cnn environment active (`icetop-cnn`).
- `pysubmit.py` next to the scripts (same folder, imported directly).
- Read access to `/data/ana/CosmicRay/IceTop_level3/sim/`.
- `$ICETOP_CNN_DATA_DIR` set for the converter. It falls back to
  `/data/user/$USER/icetop-cnn` if unset.

Run everything from `~/icetop-cnn/data_extraction`. The scripts hard-code that
path for the extractor and for condor logs.

## Quick start

```bash
cd ~/icetop-cnn/data_extraction
./sim-wrapper.py --dryrun     # print the plan, run nothing
./sim-wrapper.py -c           # extract, then queue the converter
```

`sim-wrapper.py` asks for an output folder name, then walks the tree:

1. **years** - only years that hold at least one of `p`, `He`, `O`, `Fe`.
2. **interaction models** - only models with extractable compositions.
3. **compositions** - `p`, `He`, `O`, `Fe`.
4. **simulation sets** and **versions** - only real `.i3` leaves.

### Prompt answers

Every prompt uses the same picker:

| Answer | Effect |
|---|---|
| `1 3` or `p He` | pick by number or name, space or comma separated |
| `a` | all options at this prompt |
| `aa` | all options now and at every later prompt |
| `b` | back out and redo the whole selection |

Nothing runs until the plan is printed and you confirm with `y`. The plan shows
the condor job count first.

### `sim-wrapper.py` flags

| Flag | Meaning |
|---|---|
| `-c`, `--convert` | queue `sim-converter.py` as one condor job after extraction |
| `-d`, `--dryrun` | print the `sim-maker.py` commands, run nothing |
| `-e`, `--he` | extract high-energy sets from `NewL3_MC` (log10(E/GeV) > 8.0) |
| `--test` | run the 2012 oldstructure set off-cluster as a smoke test |

## Manual use

### Extract

```bash
./sim-maker.py -i <comp root or simset dir> -o <out dir> [-c proton] [-g GCD] [-n 20000]
```

| Flag | Default | Meaning |
|---|---|---|
| `-i`, `--inputPath` | none | input files; may be a whole composition root |
| `-o`, `--out` | `/data/user/$USER/sim` | output folder, also the dataset name |
| `-c`, `--composition` | `proton` | `proton`, `helium`, `oxygen`, `iron`, `all` |
| `-n`, `--n` | `1000` | files per condor job |
| `-g`, `--gcdPath` | none | GCD file; omit to resolve per simset |
| `--mc` | off | read generated CORSIKA files instead of Level3 |
| `--test` | off | run 5 files off-cluster |

`sim-maker.py` groups files by simset, chunks each group into `-n` batches, and
submits one job per batch with `request_memory = 12 GB`. One batch is about
20000 files, roughly 300k events and 500 MB of output.

### Convert

```bash
./sim-converter.py -s <extraction folder> [-o both] [-i] [--overwrite]
```

| Flag | Default | Meaning |
|---|---|---|
| `-s`, `--source` | none | folder holding `sim_*.npy` |
| `-o`, `--output` | `both` | `array`, `param`, or `both` |
| `-i`, `--infill` | off | keep the infill array (DOMs 79+) |
| `--overwrite` | off | rewrite existing outputs |

The converter skips files that already exist unless you pass `--overwrite`.

## Output format

Detector arrays: `(N, 10, 10, 8)` float32. 78 stations on a 10x10 hex grid
(Y -4..5, X -5..4), 8 channels: `q1h, q1s, q2h, q2s, t1h, t1s, t2h, t2s`.

Event parameter dicts: `file_info`, `energy` (log10), `comp` (1/4/16/56),
`dir`, `plane_dir`, `laputop_dir`, `small_dir`, `passed_STA5`,
`uncontained_cut`, `quality_cut`.

Composition aliases in filenames: `12360` p, `12630` He, `12631` O, `12362` Fe.

The simset ID is in every filename on purpose. Batch numbers restart at 000 per
simset, so alias plus batch alone would collide across simsets and the
skip-if-exists check would silently drop data.

## Gotchas

- **Batch is per call.** `-n` batches inside one `sim-maker.py` call. One call
  per composition root, not per simset, or you get hundreds of tiny jobs.
- **GCD diffs.** Most Level3 GCD files carry `I3GeometryDiff`, not full
  geometry. The extractor probes both the GCD and the first data file before
  it decides to skip the uncompress step. A job that dies in about two minutes
  with no output is this.
- **Symlinks.** The extractor and the job estimator both ignore symlinks. A
  symlinked tree inflates file counts and pulls in stale files.
- **Per-simset GCD.** With no `-g`, the maker walks up from the batch's first
  file looking for `GCD/Level3_<simset>_GCD.i3.gz`, then for a lone year-wide
  GeoCalib file. Trees differ by year: 2012 nests energy bins, 2018 does not.
- **Ceph is slow.** `git status`, `git diff`, and `find` crawl on the repo.
  Verify edits by running the code, not by diffing.
- **High energy.** `--he` reads `dataset_info_v0.json` next to `NewL3_MC` and
  keeps only sets with `ebin_last > 8.0`. Sets named `bad_*` are quarantine.

## Files

| File | Role |
|---|---|
| `sim-wrapper.py` | interactive launcher, plan preview, job estimate |
| `sim-maker.py` | batches files and submits extractor jobs |
| `sim-extractor.py` | icetray module, writes per-event numpy dicts |
| `sim-converter.py` | numpy dicts to tensor arrays and parameter dicts |
| `sim_utils.py` | `load_data`, `dict_to_mat`, hex grid mapping |
| `pysubmit.py` | condor submit wrapper |
| `condor/` | exec, out, log, error files from every job |
