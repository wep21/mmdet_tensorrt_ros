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

#include <ament_index_cpp/get_package_prefix.hpp>
#include <mmdet_tensorrt_yolox/mmdet_tensorrt_yolox_node.hpp>

#include <vision_msgs/msg/object_hypothesis.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <vision_msgs/msg/point2_d.hpp>
#include <vision_msgs/msg/pose2_d.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mmdet_tensorrt_yolox
{
MmdetTrtYoloxNode::MmdetTrtYoloxNode(const rclcpp::NodeOptions & node_options)
: Node("mmdet_tensorrt_yolox", node_options)
{
  using std::placeholders::_1;
  using namespace std::chrono_literals;

  if (!readLabelFile(declare_parameter("label_path", ""))) {
    RCLCPP_ERROR(this->get_logger(), "Could not find label file");
    rclcpp::shutdown();
  }
  trt_yolox_ = std::make_unique<mmdet_tensorrt_yolox::MmdetTrtYolox>(
    declare_parameter("model_path", ""), declare_parameter("precision", "fp32"),
    declare_parameter("score_threshold", 0.5), "", tensorrt_common::BatchConfig{1, 1, 1}, 1 << 30,
    std::vector<std::string>{
      ament_index_cpp::get_package_prefix("mmdeploy_tensorrt_ops") +
      "/lib/mmdeploy_tensorrt_ops/libmmdeploy_tensorrt_ops.so"});

  timer_ =
    rclcpp::create_timer(this, get_clock(), 100ms, std::bind(&MmdetTrtYoloxNode::onConnect, this));

  detections_pub_ =
    this->create_publisher<vision_msgs::msg::Detection2DArray>("~/out/detection", 1);
  image_pub_ = image_transport::create_publisher(this, "~/out/image");
}

void MmdetTrtYoloxNode::onConnect()
{
  using std::placeholders::_1;
  if (
    detections_pub_->get_subscription_count() == 0 &&
    detections_pub_->get_intra_process_subscription_count() == 0 &&
    image_pub_.getNumSubscribers() == 0) {
    image_sub_.shutdown();
  } else if (!image_sub_) {
    image_sub_ = image_transport::create_subscription(
      this, "~/in/image", std::bind(&MmdetTrtYoloxNode::onImage, this, _1), "raw",
      rmw_qos_profile_sensor_data);
  }
}

void MmdetTrtYoloxNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  vision_msgs::msg::Detection2DArray out_detections;

  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  const auto width = in_image_ptr->image.cols;
  const auto height = in_image_ptr->image.rows;

  mmdet_tensorrt_yolox::ObjectArrays objects;
  if (!trt_yolox_->doInference({in_image_ptr->image}, objects)) {
    RCLCPP_WARN(this->get_logger(), "Fail to inference");
    return;
  }
  for (const auto & yolox_object : objects.at(0)) {
    vision_msgs::msg::Detection2D detection{};
    detection.bbox.center = vision_msgs::build<vision_msgs::msg::Pose2D>()
                              .position(vision_msgs::build<vision_msgs::msg::Point2D>()
                                          .x(yolox_object.x_offset + yolox_object.width / 2)
                                          .y(yolox_object.y_offset + yolox_object.height / 2))
                              .theta(0.0);
    detection.bbox.size_x = yolox_object.width;
    detection.bbox.size_y = yolox_object.height;
    detection.results.emplace_back(
      vision_msgs::build<vision_msgs::msg::ObjectHypothesisWithPose>()
        .hypothesis(vision_msgs::build<vision_msgs::msg::ObjectHypothesis>()
                      .class_id(label_map_[yolox_object.type])
                      .score(yolox_object.score))
        .pose(geometry_msgs::msg::PoseWithCovariance{}));
    detection.header = msg->header;
    out_detections.detections.emplace_back(detection);
    const auto left = std::max(0, static_cast<int>(yolox_object.x_offset));
    const auto top = std::max(0, static_cast<int>(yolox_object.y_offset));
    const auto right =
      std::min(static_cast<int>(yolox_object.x_offset + yolox_object.width), width);
    const auto bottom =
      std::min(static_cast<int>(yolox_object.y_offset + yolox_object.height), height);
    cv::rectangle(
      in_image_ptr->image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3,
      8, 0);
  }
  image_pub_.publish(in_image_ptr->toImageMsg());

  out_detections.header = msg->header;
  detections_pub_->publish(out_detections);
}

bool MmdetTrtYoloxNode::readLabelFile(const std::string & label_path)
{
  std::ifstream label_file(label_path);
  if (!label_file.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Could not open label file. [%s]", label_path.c_str());
    return false;
  }
  int label_index{};
  std::string label;
  while (getline(label_file, label)) {
    label_map_.insert({label_index, label});
    ++label_index;
  }
  return true;
}

}  // namespace mmdet_tensorrt_yolox

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(mmdet_tensorrt_yolox::MmdetTrtYoloxNode)
