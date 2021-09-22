# MR-GMMapping
<!-- 该项目是多机 GMM Submap 建图的工作 -->
This is the open-source project **MR-GMMapping**, a communication efficient **M**ulti-**R**obot **GMM**-based **Mapping** system.

The related paper "MR-GMMapping: Communication Efficient Multi-Robot Mapping System via Gaussian Mixture Model" is submitted to the IEEE Robotics and Automation Letters (RA-L) with the 2022 International Conference on Robotics and Automation（ICRA 2022）.

The video demo is resleased at [bilibili (Chinese Youtube)](https://www.bilibili.com/video/BV1TU4y1N7Yn?from=search&seid=3677956896739468342&spm_id_from=333.337.0.0) and [Youtube](https://www.youtube.com/watch?v=gIomG8qBk8U)

## Platform
- Multi-robots with NVIDIA Jetson TX2, Intel RealSense T265, and depth camera D435i
- ubuntu 18.04 (bash)
- ROS melodic
- python2
 
## Dependency

<!-- 项目基于 ROS 和 Python2
需要安装的库主要有 -->
### Pytorch for Place Recognition
```
pip install torch torchvision
```

Pre-trained model is available [here](https://cloud.tsinghua.edu.cn/d/b37383f4c5e145c2b92a/)
Then change the regarding path of `model13.ckpt` in `MapBuilderNode.py`.

### Python-PCL

Useful reference [here](https://python-pcl-fork.readthedocs.io/en/rc_patches4/install.html#install-python-pcl).

### GTSAM

```
git clone https://bitbucket.org/gtborg/gtsam.git
mkdir build
cd build
cmake .. -DGTSAM_INSTALL_CYTHON_TOOLBOX=on
make check
make install
cd cython
sudo python setup.py install
```

### Other dependency 
```
sudo apt install ros-melodic-tf2-ros
pip install autolab_core
pip install sklearn
pip install future
pip install transforms3d
pip install pygraph
```

For ROS libraries, you can use ```apt install ros-melodic-(missing library)``` to install the missing libraries.

## Datasets

The ROS bags of the multi-robot simulators can be downloaded at...

You can also use the keyboard to control the robots in the Gazebo simulator by running
```
roslaunch turtlebot3sim small_env_two_robots.launch 
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

## Usage Example

### Installation
download the repo and then put it in your ros package
```
catkin_make
source <repo_path>/devel/setup.bash
```


### Single Robot Map Building
```
roslaunch gmm_map_python visualgmm_realsence.launch
```


### Multi-robot Relative Pose Estimation
```
roslaunch gmm_map_python visualgmm_realsence_2robot.launch
```
If you want to run this system with your simulator, there are three params need to adjust for better performance:
- `match_thr`: small `match_thr` reduce the probability of scene mismatch。
- `fitness_thr`: smaller `fitness_thr` leads to a better global map accuracy after the first map merging.
- `new_self_submap_count`: adjust according to the camera parameters. Larger `new_self_submap_count` is better for place recognition, but will bring longer mapping delay.


<!-- 先启动可视化和建图节点

```
roslaunch gmm_map_python visualgmm.launch
```

再播放相关的数据记录(得cd到.bag文件所在的文件夹中)

```
rosbag play --clock --keep-alive 30sec.bag
```

2021年更新:
我们现在新建了全新 rosbag,在rosbag 中增加了命名空间.运行方法.

```
roslaunch gmm_map_python visualgmm_realsence.launch
```

注意,要先在 visualgmm_realsence.launch 中28,29 行设置rosbag的路径.

rosbag 下载链接:

[https://cloud.tsinghua.edu.cn/f/ecc82823171e4a0aa20f/?dl=1](https://cloud.tsinghua.edu.cn/f/ecc82823171e4a0aa20f/?dl=1) -->
