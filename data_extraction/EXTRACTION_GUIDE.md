# Data Extraction Pipeline

This pipeline converts IceTop Level3 simulation files into the tensor arrays expected by `trainer.py`

Codebase written by Ethan Door, Dr. McNally, and Johnathan Stowers 

## How the pipeline functions

```
Level3 sim files + GCD
	| sim-extractor.py (One Condor job per batch)
	▼
$ICETOP_CNN_DATA_DIR/sim/...
	| sim-converter.py (One Condor job per dataset)
	▼
$ICETOP_CNN_DATA_DIR/simdata/...
	(N, 10, 10, 8) float32 tensors 
```

## How to use the pipeline
`sim-wrapper.py` is intended to be the main entry point into the pipeline. It walks you through what data to collect, where to extract it from, and can leave you with a fully converted dataset at the end.

It is required that you are running these scripts in `~/icetop-cnn/data_extraction/` and are on a **npx-submitter** node with your virtual environment activated.

## Quick Start

```
./sim-wrapper.py --dryrun	#Preview the executiton plan without submitting anything 
./sim-wrapper.py -c		#Extract then automatically queue the conversion job
```

The wrapper starts by asking for an output folder name, then steps you through the directory tree:
1. **Years** — filters down to years containing at least one of `p`, `He`, `O`, or `Fe`.
2. **Interaction models** — shows only models with extractable compositions.
3. **Compositions** — `p`, `He`, `O`, `Fe`.
4. **Simulation sets & versions** — filters out empty folders so you only see actual `.i3` leaves.

### Navigation 
Every selection prompt accepts the same shortcuts:

| Input | What it does |
|---|---|
| `1 3` or `p He` | Select specific items by index or name (space- or comma-separated) |
| `a` | Select everything at the current prompt |
| `aa` | Select everything now *and* auto-accept all remaining prompts |
| `b` | Back out and restart your selections from scratch |

If everything looks good, confirm the Condor job count and confirm with `y`.

### Wrapper flags

| Flag | What it does |
|---|---|
| `-c`, `--convert` | Automatically queue `sim-converter.py` as a single Condor job once extraction finishes |
| `-d`, `--dryrun` | Print out the underlying `sim-maker.py` commands without actually running them |
| `-e`, `--he` | Pull high-energy sets from `NewL3_MC` (log10(E/GeV) > 8.0) |
| `--test` | Run a quick off-cluster smoke test against the 2012 old-structure set |

## Running manually

### 1. Extract

```bash
./sim-maker.py -i <comp root or simset dir> -o <out dir> [-c proton] [-g GCD] [-n 20000]
```

| Flag | Default | Description |
|---|---|---|
| `-i`, `--inputPath` | *required* | Path to input files (can point to an entire composition root) |
| `-o`, `--out` | `/data/user/$USER/sim` | Output directory (also sets the dataset name) |
| `-c`, `--composition` | `proton` | Target composition: `proton`, `helium`, `oxygen`, `iron`, or `all` |
| `-n`, `--n` | `1000` | Number of files bundled per Condor job |
| `-g`, `--gcdPath` | *auto* | Explicit GCD file path (leave blank to auto-detect per simset) |
| `--mc` | `off` | Read raw generated CORSIKA files instead of Level3 |
| `--test` | `off` | Process 5 files locally off-cluster |

Under the hood, `sim-maker.py` groups files by simulation set, splits them into chunks of `-n` files, and submits one Condor job per batch requesting `12 GB` of memory.

### 2. Convert

```bash
./sim-converter.py -s <extraction folder> [-o both] [-i] [--overwrite]
```

| Flag | Default | Description |
|---|---|---|
| `-s`, `--source` | *required* | Directory containing your extracted `sim_*.npy` files |
| `-o`, `--output` | `both` | What to write: `array`, `param`, or `both` |
| `-i`, `--infill` | `off` | Include the infill array (DOMs 79+) |
| `--overwrite` | `off` | Regenerate files even if they already exist |

By default, the converter skips existing outputs so you can safely re-run interrupted jobs unless you pass `--overwrite`.

## Traing a model on extracted data 

To train a model on the extracted dataset, you must take advatatage of the `-d` or `--dataset` flag in `submit.py`.

`-d` accepts the name of the dataset, and resolves to your default `ICETOP_CNN_DATA_DIR` path in your venv.

`--dataset` accepts the absolute path to any converted dataset path.

## File reference

| File | What it does |
|---|---|
| `sim-wrapper.py` | Interactive CLI launcher, tree navigator, and job estimator |
| `sim-maker.py` | Groups files into batches and submits extractor jobs to Condor |
| `sim-extractor.py` | IceTray module that parses `.i3.gz` frames into per-event NumPy dicts |
| `sim-converter.py` | Reshapes NumPy dicts into final training tensors and parameter arrays |
| `sim_utils.py` | Shared helpers (`load_data`, `dict_to_mat`, and hex-grid coordinate mapping) |
| `pysubmit.py` | Lightweight wrapper around `condor_submit` |
| `condor/` | Holds the generated `.sh`, `.out`, `.log`, and `.err` files for every submitted job |
