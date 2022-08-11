# ROS integration for mmdetection and TensorRT

## Requirements
- Ubuntu 22.04
- ROS2 Humble
- TensorRT 8

## Install
1. Install ros and colcon. See https://docs.ros.org/en/humble/Installation.html.
2. Install cuda, cudnn, TensorRT. See https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html.
3. Clone src and Install dependencies
```bash
mkdir -p ros_ws/src && cd ros_ws/src
git clone git@github.com:wep21/mmdet_tensorrt_ros.git
cd mmdet_tensorrt_ros
vcs import . < dependencies.repos
cd ../../
rosdep install --from-paths src --ignore-src -y
```
4. Build
```bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --symlink-install --packages-up-to mmdet_tensorrt_yolox
```

## Supported model

* [YOLOX](mmdet_tensorrt_yolox/docs/README.md)

## Reference

*  [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)