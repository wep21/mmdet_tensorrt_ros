// Copyright 2022 Daisuke Nishimatsu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MMDET_TENSORRT_YOLOX__MMDET_TENSORRT_YOLOX_NODE_HPP_
#define MMDET_TENSORRT_YOLOX__MMDET_TENSORRT_YOLOX_NODE_HPP_

#include <image_transport/image_transport.hpp>
#include <mmdet_tensorrt_yolox/mmdet_tensorrt_yolox.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mmdet_tensorrt_yolox
{
using LabelMap = std::map<int, std::string>;

class MmdetTrtYoloxNode : public rclcpp::Node
{
public:
  explicit MmdetTrtYoloxNode(const rclcpp::NodeOptions & node_options);

private:
  void onConnect();
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  bool readLabelFile(const std::string & label_path);

  image_transport::Publisher image_pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub_;

  image_transport::Subscriber image_sub_;

  rclcpp::TimerBase::SharedPtr timer_;

  LabelMap label_map_;
  std::unique_ptr<mmdet_tensorrt_yolox::MmdetTrtYolox> trt_yolox_;
};

}  // namespace mmdet_tensorrt_yolox

#endif  // MMDET_TENSORRT_YOLOX__MMDET_TENSORRT_YOLOX_NODE_HPP_
