# GMM_Map_Python

该项目是多机 GMM Submap 建图的工作

## 环境配置

项目基于 ROS 和 Python2
需要安装的库主要有

```
sudo apt install ros-melodic-tf2-ros
pip install autolab_core
pip install sklearn
```

如果遇到库缺失,基本上可以通过 ```apt install ros-melodic-{缺失库}``` 进行安装.

## 数据集

目前采用自己录制的数据集 30sec.bag

## 使用方法