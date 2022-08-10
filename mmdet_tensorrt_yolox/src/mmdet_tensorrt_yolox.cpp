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

#include <mmdet_tensorrt_yolox/mmdet_tensorrt_yolox.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mmdet_tensorrt_yolox
{
MmdetTrtYolox::MmdetTrtYolox(
  const std::string & model_path, const std::string & precision,
  const float score_threshold,
  [[maybe_unused]] const std::string & cache_dir,
  const tensorrt_common::BatchConfig & batch_config,
  const size_t max_workspace_size,
  const std::vector<std::string> & plugin_paths)
: score_threshold_(score_threshold)
{
  trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
    model_path, precision, nullptr, batch_config,
    max_workspace_size, plugin_paths);
  trt_common_->setup();

  if (!trt_common_->isInitialized()) {
    return;
  }

  // GPU memory allocation
  const auto input_dims = trt_common_->getBindingDimensions(0);
  const auto input_size = std::accumulate(
    input_dims.d + 1, input_dims.d + input_dims.nbDims, 1, std::multiplies<int>());
  const auto out_labels_dims = trt_common_->getBindingDimensions(2);
  max_detections_ = out_labels_dims.d[1];
  input_d_ = cuda_utils::make_unique<float[]>(batch_config[2] * input_size);
  out_dets_d_ = cuda_utils::make_unique<float[]>(batch_config[2] * max_detections_ * 5);
  out_labels_d_ = cuda_utils::make_unique<int32_t[]>(batch_config[2] * max_detections_);
}
void MmdetTrtYolox::preprocess(const std::vector<cv::Mat> & images)
{
  const auto batch_size = images.size();
  auto input_dims = trt_common_->getBindingDimensions(0);
  input_dims.d[0] = batch_size;
  trt_common_->setBindingDimensions(0, input_dims);
  const float input_height = static_cast<float>(input_dims.d[2]);
  const float input_width = static_cast<float>(input_dims.d[3]);
  std::vector<cv::Mat> dst_images;
  scales_.clear();
  for (const auto & image : images) {
    cv::Mat dst_image;
    const float scale = std::min(input_width / image.cols, input_height / image.rows);
    scales_.emplace_back(scale);
    const auto scale_size = cv::Size(image.cols * scale, image.rows * scale);
    cv::resize(image, dst_image, scale_size, 0, 0, cv::INTER_CUBIC);
    const auto bottom = input_height - dst_image.rows;
    const auto right = input_width - dst_image.cols;
    copyMakeBorder(
      dst_image, dst_image, 0, bottom, 0, right,
      cv::BORDER_CONSTANT, {114, 114, 114});
    dst_images.emplace_back(dst_image);
  }
  const auto chw_images = cv::dnn::blobFromImages(
    dst_images, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);

  const auto data_length = chw_images.total();
  input_h_.reserve(data_length);
  const auto flat = chw_images.reshape(1, data_length);
  input_h_ = chw_images.isContinuous() ? flat : flat.clone();
}

bool MmdetTrtYolox::doInference(const std::vector<cv::Mat> & images, ObjectArrays & objects)
{
  if (!trt_common_->isInitialized()) {
    return false;
  }

  preprocess(images);

  CHECK_CUDA_ERROR(
    cudaMemcpy(
      input_d_.get(), input_h_.data(), input_h_.size() * sizeof(float), cudaMemcpyHostToDevice)
  );
  std::vector<void *> buffers = {
    input_d_.get(), out_dets_d_.get(), out_labels_d_.get()};

  trt_common_->enqueueV2(buffers.data(), *stream_, nullptr);

  const auto batch_size = images.size();
  auto out_dets = std::make_unique<float[]>(5 * batch_size * max_detections_);
  auto out_labels = std::make_unique<int32_t[]>(batch_size * max_detections_);

  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      out_dets.get(), out_dets_d_.get(),
      sizeof(float) * 5 * batch_size * max_detections_, cudaMemcpyDeviceToHost,
      *stream_));
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      out_labels.get(), out_labels_d_.get(),
      sizeof(int32_t) * batch_size * max_detections_, cudaMemcpyDeviceToHost,
      *stream_));
  cudaStreamSynchronize(*stream_);
  objects.clear();
  for (size_t i = 0; i < batch_size; ++i) {
    ObjectArray object_array(max_detections_);
    for (size_t j = 0; j < max_detections_; ++j) {
      Object object{};
      object.score = out_dets[i * max_detections_ * 5 + j * 5 + 4];
      if (object.score < score_threshold_) {break;}
      const auto x1 = out_dets[i * max_detections_ * 5 + j * 5] / scales_[i];
      const auto y1 = out_dets[i * max_detections_ * 5 + j * 5 + 1] / scales_[i];
      const auto x2 = out_dets[i * max_detections_ * 5 + j * 5 + 2] / scales_[i];
      const auto y2 = out_dets[i * max_detections_ * 5 + j * 5 + 3] / scales_[i];
      object.x_offset = std::clamp(0, static_cast<int32_t>(x1), images[i].cols);
      object.y_offset = std::clamp(0, static_cast<int32_t>(y1), images[i].rows);
      object.width = static_cast<int32_t>(std::max(0.0F, x2 - x1));
      object.height = static_cast<int32_t>(std::max(0.0F, y2 - y1));
      object.type = out_labels[i * max_detections_ + j];
      object_array.emplace_back(object);
    }
    objects.emplace_back(object_array);
  }
  return true;
}

}  // namespace mmdet_tensorrt_yolox
