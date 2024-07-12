# IceTop ML Cosmic Ray Reconstruction

## Table of Contents

- [Introduction](#introduction)
- [Installation](#how-to-install)
- [Usage](#user-guide)
- [Known Issues](#known-issues-wip)

## Introduction
Welcome to icetop-cnn!

As a high-level overview, this project aims to train neural networks on low-statistics data collected from the IceTop surface detector.

## How to Install
This installation tutorial assumes you have some familiarity with navigating the Linux operating system.\
If you are new to working in the command line or in a Linux environment, check out [this section](#linux) on Linux in our user guide.

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

### Running the Setup Script
Assuming the default naming convention, make the cloned repository your new working directory.
```bash
cd icetop-cnn
```
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
# TensorFlow environment activation toggle function
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
Congratulations! You are now ready to begin working on the icetop-cnn project.\
For an introduction to Machine Learning, be sure to check out the folder labeled "tutorial".\
For help on getting started, see our [user guide](#how-to-use-the-project).

## User Guide
### How to Use the Project
Your entry point is going to be **submit.py**. This program is meant to be called from the command line.\
To learn more about this program, you may print its help page by running it with the **-h** flag.
```bash
./submit.py -h
```
<p align="center">
  <b>NOTE</b><br>
  While it is possible, it is discouraged to run <b>trainer.py</b> directly as an executable.<br>
  For debugging off the cluster, simply run <b>submit.py</b> with the <b>-t</b> flag.
</p>

Models are created within 

Both **submit.py** and **trainer.py** offer the full range of features available with this project without any necessary modifications.\
As new functionality is added to the project, the scripts will need to be updated.\
For first-time users, it is encouraged to spend some time experimenting with existing functionality.


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

### Linux

## Known Issues (WIP)
This section is a work and progress and may be expanded as issues arise.\
If you have found an issue that you would like to report, please do so under the "Issues" section of the repository.

### Memory Limit Exceeded
Sometimes when submitting a job to the cluster, your job will be held with a "Policy Violation: Memory Limit Exceeded" error.\
You will find that in many cases, simply re-submitting the job will be sufficient to get past this error.\
Due to the nature of working with a computing cluster, there is some unpredictability with architectures and concurrent jobs that is unavoidable.\
If you find that the issue persists, it may be necessary to adjust the memory allocated in a model's submission file.

### No Module Named Matplotlib
This is a recently reported error that is yet to be thoroughly investigated.\
It seems that, in rare cases, the node that receives a job may not have matplotlib installed.\
Luckily, the model will have trained and saved its reconstructions by this point.\
The error only occurs when graphing the model's loss curves, which can be done manually by itself off the cluster.