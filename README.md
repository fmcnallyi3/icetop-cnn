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

This project aims to train neural networks on low-level cosmic-ray air shower data collected from the IceTop surface detector. It is built for energy but can be extended to core position, direction, and maybe even composition.

## How to Install
This installation tutorial assumes you have some familiarity with navigating a terminal and the Linux operating system. If you are new to working in the command line or in a Linux environment, check out [this section](https://github.com/fmcnallyi3/icetop-cnn/wiki/User-Guide#linux) on Linux in our [user guide](https://github.com/fmcnallyi3/icetop-cnn/wiki/User-Guide).

<details open>
  <summary><b>Log in to Cobalt</b></summary>

  IceTop-CNN is designed to use IceCube's computing resources. You must log in to a computing node ("cobalt") to get started. First, open the command prompt (Windows) or terminal (Mac/Linux). If you have [configured your SSH](https://github.com/fmcnallyi3/icetop-cnn/wiki/User-Guide#configuring-ssh), then you may log in with the following command:
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

  You should now be in your home directory on a cobalt node. To get code from this GitHub repository to your IceCube filesystem, you will have to copy it. Luckily, Git offers a way to do this easily.

  From your home directory, clone the GitHub repository.
  ```bash
  git clone https://github.com/fmcnallyi3/icetop-cnn.git
  ```
  <table align="center">
    <tr><td>
      <p align="center">
        :warning:<b>WARNING</b>:warning:<br>
        It is discouraged for new users to deviate from this command and specify a different name for the repository.<br>
        This is because some file paths depend on this naming convention. For more experienced users, edit at your own risk.
      </p>
    </td></tr>
  </table>
</details>
<br>
<details open>
  <summary><b>Run the Setup Script</b></summary>

  Before you can launch in to training your own models, you must create and initialize your working environment. This includes your [TensorFlow](https://www.tensorflow.org/versions/r2.14/api_docs) environment as well as the necessary files to use that environment in [JupyterHub](https://jupyterhub.icecube.wisc.edu/hub/). Thankfully, all the work is done by a single script, and all you have to do is run it. 

  Assuming the default naming convention, make the cloned repository your new working directory.
  ```bash
  cd icetop-cnn
  ```
  You should now be ready to run the setup script. The process may take a few minutes and will notify you once completed.
  ```bash
  ./first_time_setup.sh
  ```
</details>
<br>
<details open>
  <summary><b>Edit .bashrc</b></summary>

  Now that your environment has been initialized, we are going to write a function to activate it, as well as create some needed environment variables. This is a cruicial step; without this, your scripts will not work.

  First navigate back to your home directory.
  ```bash
  cd
  ```
  Next, we need to edit the hidden ".bashrc" file. This file is run each time a new bash shell is created. For more information on .bashrc, visit [this](https://www.digitalocean.com/community/tutorials/bashrc-file-in-linux) website. In the text editor of your choice, copy the below function into your .bashrc:
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
  <table align="center">
    <tr><td>
      <p align="center">
        <b>NOTE</b><br>
        For the more experienced users who edited the name of the repository when cloning,
        this step is where you should adjust your folder paths. Additionally, check out <a href=https://wiki.icecube.wisc.edu/index.php/Jupyterhub>this</a> page on how to configure your <a href=https://jupyterhub.icecube.wisc.edu/hub>JupyterHub</a> kernel.
      </p>
    </td></tr>
  </table>

  You can now save and exit the file. At this point, your .bashrc has been edited, but you are still running off of the older version before the "icetop-cnn" function was added. To restart your .bashrc, enter the following command:
  ```bash
    source .bashrc
  ```

  Congratulations! You are now ready to begin working on the icetop-cnn project.\
  For an introduction to machine learning, be sure to check out the folder labeled "[tutorial](https://github.com/fmcnallyi3/icetop-cnn/tree/main/tutorial)". This will guide you through the "Hello World" of machine learning with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset.

  For help on getting started with the project, see our [user guide](https://github.com/fmcnallyi3/icetop-cnn/wiki/User-Guide).
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
