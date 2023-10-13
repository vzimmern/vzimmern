# This script highlights the basic commands to execute in terminal on a clean version of Ubuntu 22.04 LTS to get DeepLabCut and Anipose to work for myoclonus detection/quantification. 

# set path to DLC-GPU.yaml file
$DLC-GPU-PATH = ~/Documents/glove-test-case/DLC-GPU.yaml

# Some basic software that have to be installed for CUDA/NVIDIA GPU

sudo apt install gcc # this is for NVIDIA
sudo apt install libcanberra-gtk-module libcanberra-gtk3-module # this is for DeepLabCut

# installing Anaconda/Miniconda
sudo apt install curl
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# initiate a conda environment
conda create --name tf python=3.9
conda activate tf

# at this stage, make sure to download the NVIDIA GEForce RTX 3090 (not the Ti version) for Linux 64 bit directly from the NVIDIA website and leave that file in the Downloads folder

# first step is to inactivate the X-server in Ubuntu
# this is done by pressing "CTRL + ALT + F1"
# then type "sudo service lightdm stop" -- in my case, the response was that lightdm was not installed
# confirm this by pressing "CTRL + ALT + F7" should give you a black screen -- the way to get out of that is "CTRL + ALT + F2" and log back in

# NVIDIA wants to install the Nouveau driver automatically but we want to install a different driver. Nouveau is a display driver for NVIDIA GPUs, developed as an open-source project through reverse-engineering of the NVIDIA driver. It ships with many current Linux distributions as the default display driver for NVIDIA hardware. It is not developed or supported by NVIDIA, and is not related to the NVIDIA driver, other than the fact that both Nouveau and the NVIDIA driver are capable of driving NVIDIA GPUs. Only one driver can control a GPU at a time, so if a GPU is being driven by the Nouveau driver, Nouveau must be disabled before installing the NVIDIA driver.

# For more information, please see: https://download.nvidia.com/XFree86/Linux-x86_64/304.137/README/commonproblems.html#nouveau

# It is recommended to create a new file, for example, /etc/modprobe.d/disable-nouveau.conf, rather than editing one of the existing files, such as the popular /etc/modprobe.d/blacklist.conf. Note that some module loaders will only look for configuration directives in files whose names end with .conf, so if you are creating a new file, make sure its name ends with .conf.

# Whether you choose to create a new file or edit an existing one, the following two lines will need to be added:

# blacklist nouveau
# options nouveau modeset=0

# The first line will prevent Nouveau's kernel module from loading automatically at boot. It will not prevent manual loading of the module, and it will not prevent the X server from loading the kernel module. The second line will prevent Nouveau from doing a kernel modeset. Without the kernel modeset, it is possible to unload Nouveau's kernel module, in the event that it is accidentally or intentionally loaded.

# You will need to reboot your system after adding these configuration directives in order for them to take effect.

# If nvidia-installer detects Nouveau is in use by the system, it will offer to create such a modprobe configuration file to disable Nouveau.

sudo nano /etc/modprobe.d/blacklist.conf # and add the 2 lines above

# NOW REBOOT THE COMPUTER!

# Next step, please make sure that you have the package 'make' installed.  If make is installed on your system, then please check that 'make` is in your PATH.

sudo apt install make

cd ~/Downloads
sudo sh NVIDIA-Linux-x86_64-535.54.03.run
# Unable to determine the path to install the libglvnd EGL vendor library config files. Check that you have pkg-config and the libglvnd development libraries installed, or specify a path with --glvnd-egl-config-path. -- this error did not keep NVIDIA from installing

# these next lines install CUDA system-wide for this Linux x86_64 Ubuntu setup

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# these next lines install CUDA toolkit and cudnn

nvidia-smi #make surethat NVIDIA GPU driver is installed
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163

# The system paths will be automatically configured when you activate this conda environment.
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
--
# tensorflow with CUDA installation
pip install --upgrade pip # will probably return "requirement already satisfied". Move on.
pip install tensorflow==2.12.*

# IN CASE NONE OF THE ABOVE WORKS, it seems another viable (perhaps easier approach) is
ubuntu-drivers devices # pick the recommended driver
sudo apt install nvidia-driver-535-server-open


# verify tensorflow with CPU 
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

# verify tensorflow with GPU
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# in previous versions of this script, this command allowed the conda activate command later on to function. I'm not sure it's still needed so it's commented out. 

#source ~/anaconda3/etc/profile.d/conda.sh 

#installing KDenlive for video editing so that videos are synchronized
# make sure to download the AppImage from the kdenlive website
sudo apt-get install fuse #needed for use of the appimage
chmod a+x kdenlive-23.04.3-x86_64.AppImage 
./kdenlive-23.04.03-x86_64.AppImage

# importing DeepLabCut with GPU compatibility with Anaconda
conda env create -f DLC-GPU.yaml
conda activate DLC-GPU
pip install 'deeplabcut[gui,tf,modelzoo]'
ipython
import deeplabcut 
deeplabcut.launch_dlc() # should open up the GUI to work on projects directly rather than using code in Python

# once deeplabcut 2D training is complete, then deactivate deeplabcut environment to start an anipose environment ??

# import and install anipose
conda create -n anipose-test-case python=3.7 tensorflow=1.15.0
conda activate anipose-test-case
python -m pip install deeplabcut
python -m pip install anipose
conda install mayavi ffmpeg
pip install --upgrade apptools
anipose

# anipose portion of the script
anipose analyze
anipose filter
anipose label-2d-filter
pip install opencv-contrib-python==4.6.0.66
anipose calibrate
anipose triangulate
conda install -c conda-forge libstdcxx-ng
anipose label-3d
anipose label-combined
