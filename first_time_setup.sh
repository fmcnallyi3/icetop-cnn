#!/bin/sh

# Create virtual environment
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh`
python -m virtualenv $PWD/.venv
source $PWD/.venv/bin/activate
pip install -r requirements.txt

# Create JupyterHub kernel
JUPYTER_ICETOP_CNN_KERNEL_DIR="$HOME/.local/share/jupyter/kernels/icetop-cnn"
mkdir -p $JUPYTER_ICETOP_CNN_KERNEL_DIR

START_KERNEL="$JUPYTER_ICETOP_CNN_KERNEL_DIR/start-kernel.sh"
KERNEL="$JUPYTER_ICETOP_CNN_KERNEL_DIR/kernel.json"

cat <<EOF > $START_KERNEL
#!/bin/sh
source $PWD/.venv/bin/activate
exec $PWD/.venv/bin/python -m ipykernel_launcher -f $1 2>/dev/null
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

echo "TensorFlow environment initialized!"




