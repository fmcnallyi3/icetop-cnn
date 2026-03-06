#!/bin/bash

# Create virtual environment
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh`
unset PYTHONPATH
python -m virtualenv /data/user/$USER/.venv
source /data/user/$USER/.venv/bin/activate
pip install -U -r requirements.txt

# Fix tensorflow (blindly pasted from tensorflow install instructions)
pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
ln -sf ../nvidia/*/lib/*.so* .
popd
ln -sf $(find $(dirname $(dirname $(python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__file__)"))/*/bin/) -name ptxas -print -quit) $VIRTUAL_ENV/bin/ptxas

# Create JupyterHub kernel
JUPYTER_ICETOP_CNN_KERNEL_DIR="$HOME/.local/share/jupyter/kernels/icetop-cnn"
mkdir -p $JUPYTER_ICETOP_CNN_KERNEL_DIR

START_KERNEL="$JUPYTER_ICETOP_CNN_KERNEL_DIR/start-kernel.sh"
KERNEL="$JUPYTER_ICETOP_CNN_KERNEL_DIR/kernel.json"

# Populate the Jupyter Kernel File 
cat <<EOF > $START_KERNEL
#!/bin/sh
source /data/user/$USER/.venv/bin/activate
export ICETOP_CNN_DIR=$PWD
export ICETOP_CNN_DATA_DIR=/data/user/$USER/icetop-cnn
export ICETOP_CNN_SCRATCH_DIR=/scratch/$USER/icetop-cnn
exec /data/user/$USER/.venv/bin/python -m ipykernel_launcher -f \$1 2>/dev/null
EOF

chmod +x $START_KERNEL

cat <<EOF > $KERNEL
{
    "argv": [
        "$START_KERNEL",
        "{connection_file}"
    ],
    "display_name": "IceTop-CNN", 
    "language": "python",
    "metadata": {
        "debugger": "true"
    }
}
EOF

# Create settings for VSCode

mkdir -p $PWD/.vscode
cat <<EOF > $PWD/.vscode/settings.json
{
    "files.exclude": {
        "__pycache__": true,
        "**/.ipynb_checkpoints": true,
        ".venv": true,
        ".vscode": true,
    },
    "files.watcherExclude": {
        "**/.venv/**": true,
    },
    "python.analysis.extraPaths": [
        "/data/user/$USER/.venv/lib/python3.11/site-packages"
    ],
}
EOF

echo "TensorFlow environment and JupyterHub kernel initialized!"