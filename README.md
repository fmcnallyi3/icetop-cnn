![IceCube Neutrino Observatory](https://res.cloudinary.com/icecube/images/v1598387622/Header_HomeA_2000/Header_HomeA_2000.png)
# IceTop ML Cosmic Ray Reconstruction
<table align="center">
  <tr>
    <td><a href="https://github.com/fmcnallyi3/icetop-cnn?tab=readme-ov-file#introduction">Introduction</a></td>
    <td><a href="https://github.com/fmcnallyi3/icetop-cnn?tab=readme-ov-file#how-to-install">Installation</a></td>
    <td><a href="https://github.com/fmcnallyi3/icetop-cnn/wiki/User-Guide">User Guide</a></td>
    <td><a href="https://github.com/fmcnallyi3/icetop-cnn?tab=readme-ov-file#known-issues-wip">Known Issues</a></td>
  </tr>
</table>

## Introduction
Welcome to icetop-cnn!

As a high-level overview, this project aims to train neural networks on low-statistics data collected from the IceTop surface detector.

## How to Install
This installation tutorial assumes you have some familiarity with navigating a terminal and the Linux operating system.\
If you are new to working in the command line or in a Linux environment, check out [this section](https://github.com/fmcnallyi3/icetop-cnn/wiki/User-Guide#linux) on Linux in our [user guide](https://github.com/fmcnallyi3/icetop-cnn/wiki/User-Guide).

<details open>
  <summary><b>Log in to Cobalt</b></summary>

  First, open the command prompt (Windows) or terminal (Mac/Linux).\
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
  There is a section on [configuring your SSH](https://github.com/fmcnallyi3/icetop-cnn/wiki/User-Guide#configuring-ssh) in the [user guide](https://github.com/fmcnallyi3/icetop-cnn/wiki/User-Guide).
</details>
<br>
<details open>
  <summary><b>Clone the GitHub Repository</b></summary>
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
</details>
<br>
<details open>
  <summary><b>Running the Setup Script</b></summary>

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
</details>
<br>
<details open>
  <summary><b>Editing .bashrc</b></summary>
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
  You can now save and exit the file.

  Congratulations! You are now ready to begin working on the icetop-cnn project.\
  For an introduction to Machine Learning, be sure to check out the folder labeled "tutorial".\
  For help on getting started, see our [user guide](https://github.com/fmcnallyi3/icetop-cnn/wiki/User-Guide).
</details>

## Known Issues (WIP)
This section is a work and progress and may be expanded as issues arise.\
If you have found an issue that you would like to report, please do so under the "Issues" section of the repository.

<details>
  <summary><b>Memory Limit Exceeded</b></summary>
  <div>
    Sometimes when submitting a job to the cluster, your job will be held with a "Policy Violation: Memory Limit Exceeded" error.<br>
    You will find that in many cases, simply re-submitting the job will be sufficient to get past this error.<br>
    Due to the nature of working with a computing cluster, there is some unpredictability with architectures and concurrent jobs that is unavoidable.<br>
    If you find that the issue persists, it may be necessary to adjust the memory allocated in a model's submission file.
  </div>
</details>

<details>
  <summary><b>No Module Named Matplotlib</b></summary>
  <div>
    This is a recently reported error that is yet to be thoroughly investigated.<br>
    It seems that, in rare cases, the node that receives a job may not have matplotlib installed.<br>
    Luckily, the model will have trained and saved its reconstructions by this point.<br>
    The error only occurs when graphing the model's loss curves, which can be done manually by itself off the cluster.
  </div>
</details>
