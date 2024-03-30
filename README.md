# Pursuit-Evasion Platform for Robot Learning (pep4rl)
## Basic Requirements
- Ubuntu 20.04
- ROS Noetic with some ROS package dependencies ([octomap](https://wiki.ros.org/octomap), [mavros](https://wiki.ros.org/mavros), [vision_msgs](https://wiki.ros.org/vision_msgs))
- Python 3.9
- Pytorch 2.0.0 (with CUDA 11.8, tensorboard and wandb)

We have provided the `requirements.txt` for the python environment.
## Installation Guide
### 1. One-line Command installation of ROS
This repo requires to install ROS Noetic with Ubuntu 20.04. Assuming you have Ubuntu 20.04 already installed on your machine, you can install ROS Noetic with a super simple one-line command from [fishros](https://github.com/fishros/install). First, enter the following command in your Terminal:
```commandline
wget http://fishros.com/install -O fishros && . fishros
```
Then, type `1` to start the installation of ROS. Note that the messages for installation choices generated after this command are in Chinese. You may need a translator to follow the important messages.  
### 2. Install ROS Package Dependencies
Once the ROS Noetic is properly setup, install the ROS packages ([octomap](https://wiki.ros.org/octomap), [mavros](https://wiki.ros.org/mavros), and [vision_msgs](https://wiki.ros.org/vision_msgs)) that this repo depends on:
```commandline
sudo apt install ros-${ROS_DISTRO}-octomap* && sudo apt install ros-${ROS_DISTRO}-mavros* && sudo apt install ros-${ROS_DISTRO}-vision-msgs
```
### 3. Create Python Environment
pip install
empy==3.3.4 (must be 3.3.4)
catkin_pkg
pyyaml
rospkg
numpy
opencv-python
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install 
matplotlib
tensorboard
wandb
tyro
### 4. Build and Setup the ROS workspace
Clone the repo:
```commandline
git clone https://github.com/Toyoid/pep4rl.git
```
add pep4rl workspace to PYTHONPATH for package searching -- but not work for pycharm
```
export PYTHONPATH="Path/To/pep4rl/learning:$PYTHONPATH"
```
Then follow the standard catkin_make process:
```commandline
cd Path/To/pep4rl/ros/
catkin_make
```
Lastly, setup environment variable by adding the following to your `~/.bashrc` file:
```commandline
source Path/To/uav_simulator/gazeboSetup.bash
```
Optionally, we recommend you also add the `pep4rl` workspace to your `~/.bashrc` file for the convenience of future usage:
```commandline
source Path/To/pep4rl/ros/devel/setup.bash
```
ISSUE
LIBTIFF
If you get an error like
```commandline
...
  File "/opt/ros/noetic/lib/python3/dist-packages/cv_bridge/core.py", line 91, in encoding_to_cvtype2
    from cv_bridge.boost.cv_bridge_boost import getCvType
ImportError: /lib/libgdal.so.26: undefined symbol: TIFFReadRGBATileExt, version LIBTIFF_4.0
```
add this to your `~/.bashrc`
```commandline
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
```
## Usage