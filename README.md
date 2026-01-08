![IceCube Neutrino Observatory](https://res.cloudinary.com/icecube/images/q_auto/v1602553309/gal_Detector_IMG_4263_207090c7a/gal_Detector_IMG_4263_207090c7a.jpg)
# IceTop ML Cosmic Ray Reconstruction

## <a href="https://github.com/fmcnallyi3/icetop-cnn?tab=readme-ov-file#revised-intro">New  to Ice-Top CNN? Click here!</a>
<table align="center">
  <tr>
    <td><a href="https://github.com/fmcnallyi3/icetop-cnn?tab=readme-ov-file#introduction">Introduction</a></td>
    <td><a href="https://github.com/fmcnallyi3/icetop-cnn?tab=readme-ov-file#how-to-install">Installation</a></td>
    <td><a href="https://github.com/fmcnallyi3/icetop-cnn/wiki/User-Guide">User Guide</a></td>
    <td><a href="https://github.com/fmcnallyi3/icetop-cnn?tab=readme-ov-file#known-issues-wip">Known Issues</a></td>
  </tr>
</table>

## Introduction
Welcome to IceTop-CNN!

This project aims to train neural networks for use with low-level cosmic-ray air shower data collected from the IceTop surface detector. It is built for energy but can be extended to core position, direction, and maybe even composition.

## How to Install
This installation tutorial assumes you have some familiarity with navigating a terminal and the Linux operating system. If you are new to working in the command line or in a Linux environment, check out [this section](https://github.com/fmcnallyi3/icetop-cnn/wiki/Useful-Resources#linux) on Linux in our wiki page on [useful resources](https://github.com/fmcnallyi3/icetop-cnn/wiki/Useful-Resources).

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
        This is because some file paths depend on this naming convention.<br>
        For more experienced users, edit at your own risk.
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
  Next, we need to edit the hidden ".bashrc" file. This file is run each time a new bash shell is created. For more information on .bashrc, visit [this](https://www.digitalocean.com/community/tutorials/bashrc-file-in-linux) website.
  
  Open the hidden ".bashrc" file.
  ```bash
  nano .bashrc
  ```
  Finally, copy the function below into your .bashrc.
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
</details>

Congratulations! You are now ready to begin working on the IceTop-CNN project. You can now activate your environment at any time by running the following command:
```bash
icetop-cnn
```
You may test that you did the installation correctly by returning to the project directory (`cd icetop-cnn`) and running the following command:
```bash
./submit.py -c pf -e 10 -p energy -n install_test -m mini0 -t
```

If all went well, you should see a model train for 10 epochs (~5s each), then be run over the assessment events (~10s).

<table align="center">
    <tr><td>
      <p align="center">
        <b>WARNING</b><br>
        While fine for a test, running on cobalt is computationally <i>expensive</i> and should not be done regularly. See the <a href=https://github.com/fmcnallyi3/icetop-cnn/wiki/User-Guide>User Guide</a> for best practices on model training. 
      </p>
    </td></tr>
</table>

For an introduction to machine learning, be sure to check out the folder labeled "[tutorial](https://github.com/fmcnallyi3/icetop-cnn/tree/main/tutorial)". This will guide you through the "Hello World" of machine learning with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset.

For help on getting started with the project, see our [User Guide](https://github.com/fmcnallyi3/icetop-cnn/wiki/User-Guide).

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




# Revised Intro
This setup is temporary. Anything above this part is the legacy Readme that was used until Fall '25. There is a VERY good chance there is information in there that's crucial that I've forgotten to include. My apologies in advance.
Please take note of any problems you had and report them to me, Johnathan Stowers, in detail! I'm very receptive to criticism, and I'm  sure Dr. McNally would like to hear about the notes you take over your experience regardless. The worst thing you can do is to not communicate or ask for help. 

*The real revision begins here! Thank you again!*
Welcome to IceTop-CNN!

This project aims to train a convolutional neural network (CNN) for use with low-level cosmic-ray air shower data collected from the IceTop surface detector. It is built for energy but can be extended to core position, direction, and perhaps even composition.

This is a guide for beginners! This is absolutely not the end all be all of working with Icecube, nor will it have the most detail. This guide is intended for setting up your basic environment, training your baseline, and creating a new model.

## Accessing the Project
The project interface is within JupyterHub for all intensive purposes. This means the project is based within a GUI, but it is required to work with the terminal for some work.

The interactivity with a command line interface (CLI) will be limited for this project. Instead, it’ll only be used for setting up the initial project and submitting models to the computing cluster. 
The two most important terminal commands you need to be familiar with are:
```bash 
ssh submit
```
And
```
./submit.py ~~~
```

## Downloading Project Files
First things first, enter into JupyterHub and log in with your IceCube credentials. On the next screen select the first server model, since this guide won’t need any of the extra computing power that the other servers offer. Depending on a future task, you may need the extra power, but it’s bad manners to use more resources than you need. 

Once you’re within a server, there should be a screen that shows lots of different options. Go to the other section, and select a terminal. Once inside of a terminal, run the following command:
```bash
git clone https://github.com/fmcnallyi3/icetop-cnn.git
```
This command will copy over the important set up files that are hosted within this GitHub repository. Do not change anything about this command, a lot of files within it depend on the specific file structure it sets up.

## Running the Setup Script
Now that the files are installed to your account, we need to move the terminal’s directory to within that folder. To navigate the folder directory within the CLI, run the following command:
```cd icetop-cnn```
CD will move the current directory to whatever path is specified within the command. Now that your terminal is active within the downloaded files, run this command to run the startup script:
```bash ./first_time_setup.sh```
This could take a few minutes, so don’t be stressed if it takes a while. You’ll be notified once everything is set up and finished.

## Entering the Virtual Environment 
What even is the virtual environment we just created?
Within our virtual environment are all of the packages and dependencies our code requires to run. This is set up this way so it can be run with minimal setup or work in the backend for anyone who uses it. 

Now we have to go back to our home directory to edit files that will allow us to access our virtual environment. To do this, run the following command:
```bash
cd
```
This command will always take you back out of the directory you’re currently in. 

Once you’re back here, we’re going to have to edit the bash file by inserting in our own code. Run this command to enter into a text editor for the bash file:
```bash
nano .bashrc
```
Nano is the name of our CLI text editor. You should be in a totally blank screen with some commands at the bottom. 
<table align="center">
    <tr><td>
      <p align="center">
        <b>NOTICE:</b><br>
            The carrot before the letter represents the control key. This means that Cntrl-C will NOT work as copy, and Cntrl-S is not set up to save. However, Cntrl-V will still function to copy text within nano. 
      </p>
    </td></tr>
</table>

Once within nano, copy the following text into your bash file:
```
## TensorFlow environment activation toggle function
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

Once this is done, press the following keys:
Control-O (To save the file)
Enter (To confirm the file name)
Control-X (To exit the file)

Now that the file is written, we just need to update our terminal to let it know that we added a new function to our bash file. To do this, run:
```bash
source .bashrc
```
Now that it’s been updated, run:
```bash
icetop-cnn
``` 
to enter the virtual environment.

<table align="center">
    <tr><td>
      <p align="center">
        <b>NOTICE:</b><br>
        If entering the virtual environment ever gives you an error AFTER this point, that means that you need to re-source your .bashrc file. To do this, simply rerun ```source .bashrc``` and you should be fine. 
      </p>
    </td></tr>
</table>

## Submitting a Model
To submit a model for training, the first thing that needs to be done is remoting into the submitter node. This can be done by running:
```bash
ssh submit
```
Then enter your IceCube password. Upon entering the note, you’ll leave your initial virtual environment so be sure to run the command to enter it once more. Now change into your project folder by running 
```bash
cd icetop-cnn
```

Now to train a model, you have to get comfortable with submit.py. The first baseline model is already available for you to train, so run the following commands:
```bash
./submit.py -c phof -e 50 -m baseline -n energy_baseline -p energy
```
And
```bash
./submit.py -c phof -e 50 -m baseline -n comp_baseline -p comp
```
The two most important parts of this command, for now, is knowing that after the -m flag is the model name, and after the -p flag is the analysis type. Please refer to the user guide for more information.

Since the jobs have been submitted to the node, they’ve entered a queue with all the other jobs. To see the progress on your jobs, run
```bash 
condor_q
```

To exit the submit node, just run:
```bash 
exit
```

## Analyzing a Model
Model analysis is contained within two notebooks, composition.ipynb and energy.ipynb. Once a job is finished, you have to enter the notebook and ensure that MODEL_NAMES_AND_NUCLEI is updated to contain the model. It’s in the first block under Model and Assessment selection, and is done in the fashion of a dict. Baseline is already configured for analysis, so for the first model this step can be skipped. However if you want to analyze another model, Baseline_Two, for the composition notebook it would look like
```
MODEL_NAMES_AND_NUCLEI = {
	‘comp_baseline’ : ‘phof’,
	‘comp_baseline_two’ : ‘phof’
	#Where ‘phof’ stands for the types of particles being analyzed
	#This will analyze both models at the same time for comparison
}
```
Before any code can begin being executed, you have to enter the virtual kernel as well. This is done by looking at the top right corner of the code window, so directly under the tabs with the files you have open, and left to the empty circle, and clicking on that name. Scroll all the way to the top of the list, select the Icetop-CNN Kernel, ensure that it’s your preferred kernel, and click select. Every new notebook you open, you have to make sure that the proper kernel is selected.

Now that the notebooks and kernel have been selected, just run every code cell one by one in order from top to bottom. It’s easiest to do this by pressing ‘shift + enter’. At the end of the notebooks and when all the code is executed, the graphs and important numbers will be calculated for further analysis. 

## Training a New Model 
Before starting this, it’s heavily recommended to go through the tutorial. Enter into the tutorial folder, and go through the notebook within the folder. Be sure that you understand each portion within the notebook.

To train the first model after baseline, go into the ‘Icetop-Cnn’ folder, then go into the ‘Architecture’ folder. This can be done in the GUI on the left side portion of the screen. Once within the Architecture folder, right click and duplicate ‘baseline.py’. Be sure to rename the file into what you want the model to be called.

Once within the file, go through the TensorFlow Docs to understand what each function does. Upon understanding what to change, and what your changes could do, submit the model to submit. As a reminder, that requires SSHing into the submit note, entering the virtual environment, and running
```bash 
./submit.py -c phof -e 50 -m YOUR_MODEL_NAME -n energy_YOUR_MODEL_NAME -p energy
```
And
```bash 
./submit.py -c phof -e 50 -m YOUR_MODEL_NAME -n comp_YOUR_MODEL_NAME -p comp
```

Once the models are finished being computed, add the model name to the analysis notebooks and run each cell. The data and graphs will be calculated and compared to any other models also put into the notebooks.

## Training Failures
Also look at error logs and whatnot (scroll to bottom of file)
Train on less epochs (like, 1) and train on Cobalt for a live session 
