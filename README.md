# IceTop ML Cosmic Ray Reconstruction

## Table of Contents

- [Introduction](#introduction)
- [Installation](#how-to-install)
- [Usage](#user-guide)
- [Known Issues](#known-issues-wip)

## Introduction
Welcome to icetop-cnn!

## How to Install
### Log in to Cobalt
If you have configured your .ssh, then you may log in with the following command:
```bash
ssh cobalt
```
Otherwise, you may log in with the following commands:
```bash
ssh jdoe@pub.icecube.wisc.edu
ssh cobalt
```
*Be sure to replace the example username with your actual IceCube username.*\
There is a section on [configuring your SSH](#configuring-ssh) further on in this document.

### Clone the GitHub Repository
<p align="center">
  :warning:<b>WARNING</b>:warning:<br>
  It is discouraged for new users to specify a different name for the repository.<br>
  This is because some file paths depend on this naming convention.<br>
  For more experienced users, edit at your own risk.
</p>

From your home directory, the next step is to clone the GitHub repository.
```bash
git clone https://github.com/fmcnallyi3/icetop-cnn.git
```

### Navigate into the Repository
Assuming the default naming convention, make the cloned repository your new working directory.
```bash
cd icetop-cnn
```

### Running the Setup Script
You should now be ready to run the setup script.\
This will create your [TensorFlow](https://www.tensorflow.org/versions/r2.14/api_docs) environment
as well as the necessary files to use that environment in [JupyterHub](https://jupyterhub.icecube.wisc.edu/hub/).\
The process may take a few minutes and will notify you once completed.
```bash
./first_time_setup.sh
```

### Editing .bashrc
<p align="center">
  <b>NOTE</b><br>
  For the more experienced users who edited the name of the repository when cloning,<br>
  this next step is where you will be able to adjust your folder paths.
</p>

This is the final and one of the most cruicial steps. Without this, your scripts will not run.\
First navigate back to your home directory.
```bash
cd
```
Next, we need to edit the hidden ".bashrc" file. This file is run each time a new bash shell is created.\
For more information on .bashrc, visit [this](https://www.digitalocean.com/community/tutorials/bashrc-file-in-linux) website.\
In the text editor of your choice, copy the below function into your .bashrc:
```bash
# TensorFlow environment activation function
icetop-cnn() {
    # Path to the tensorflow virtual environment activation script
    VENV_PATH="$HOME/icetop-cnn/.venv"

    if [ "$VIRTUAL_ENV" == "$VENV_PATH" ]; then
        deactivate
        unset ICETOP_CNN_DIR
        unset ICETOP_CNN_DATA_DIR
        unset ICETOP_CNN_SCRATCH_DIR
    else
        source "$VENV_PATH"/bin/activate
        export ICETOP_CNN_DIR=$HOME/icetop-cnn
        export ICETOP_CNN_DATA_DIR=/data/user/$USER/icetop-cnn
        export ICETOP_CNN_SCRATCH_DIR=/scratch/$USER/icetop-cnn
    fi
}
```

### Finished
Congratulations! You are now ready to begin working on the icetop-cnn project.

## User Guide
### Configuring SSH
<p align="center">
  :warning:<b>WARNING</b>:warning:<br>
  Cobalt has 8 nodes to choose from, 1-8. It is VERY important that you choose one of the latter four nodes, 5-8.<br>
  This is because the former nodes use an older CPU architecture that is incompatable<br>
  with our machine learning library,
  <a target="_blank" rel="noopener noreferrer" href="https://www.tensorflow.org/versions/r2.14/api_docs">TensorFlow</a>.
</p>

New users are recommended to take the time to first configure their .ssh.\
It is also highly recommended to set up an authentication key. This avoids the need to enter your password each time you log in.\
If you are unfamiliar with how to do that, [here](https://www.ssh.com/academy/ssh/keygen) is a link to help get you started.

Below is an example configuration file, complete with entries for an authentication key.

```
# ~/.ssh/config:

# ICECUBE PUBLIC
Host pub
  HostName pub.icecube.wisc.edu
  User jdoe
  IdentityFile ~/.ssh/id_rsa

# COBALT
Host cobalt
  HostName cobalt06.icecube.wisc.edu
  User jdoe
  IdentityFile ~/.ssh/id_rsa
  ProxyJump jdoe@pub.icecube.wisc.edu

# SUBMITTER
Host submit
  HostName submitter.icecube.wisc.edu
  User jdoe
  IdentityFile ~/.ssh/id_rsa
  ProxyJump jdoe@pub.icecube.wisc.edu
```

## Known Issues (WIP)