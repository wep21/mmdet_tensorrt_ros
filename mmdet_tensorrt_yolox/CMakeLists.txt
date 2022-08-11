cmake_minimum_required(VERSION 3.5)
project(mmdet_tensorrt_yolox)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(OpenCV REQUIRED)

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/mmdet_tensorrt_yolox.cpp
)

ament_target_dependencies(${PROJECT_NAME}
  OpenCV
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
  TENSORRT_VERSION_MAJOR=${TENSORRT_VERSION_MAJOR}
)

ament_auto_add_library(yolox_single_image_inferece_node SHARED
  src/yolox_single_image_inference_node.cpp
)

ament_target_dependencies(yolox_single_image_inferece_node
  OpenCV
)

target_link_libraries(yolox_single_image_inferece_node
  ${PROJECT_NAME}
  stdc++fs
)

target_compile_definitions(yolox_single_image_inferece_node PRIVATE
  TENSORRT_VERSION_MAJOR=${TENSORRT_VERSION_MAJOR}
)

rclcpp_components_register_node(yolox_single_image_inferece_node
  PLUGIN "mmdet_tensorrt_yolox::YoloXSingleImageInferenceNode"
  EXECUTABLE yolox_single_image_inferece
)

ament_auto_add_library(${PROJECT_NAME}_node SHARED
  src/mmdet_tensorrt_yolox_node.cpp
)

ament_target_dependencies(${PROJECT_NAME}_node
  OpenCV
)

target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}
)

target_compile_definitions(${PROJECT_NAME}_node PRIVATE
  TENSORRT_VERSION_MAJOR=${TENSORRT_VERSION_MAJOR}
)

rclcpp_components_register_node(${PROJECT_NAME}_node
  PLUGIN "mmdet_tensorrt_yolox::MmdetTrtYoloxNode"
  EXECUTABLE ${PROJECT_NAME}_node_exe
)

set(DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data")
if(NOT EXISTS "${DATA_PATH}")
  execute_process(COMMAND mkdir -p ${DATA_PATH})
endif()

ament_download(https://github.com/wep21/mmdeploy_model_zoo/raw/main/yolox/yolox-tiny.onnx
  DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/data
  MD5 e50b97142f9872dad101edb8f169b288
  FILENAME yolox-tiny.onnx
)

ament_download(https://github.com/wep21/mmdeploy_model_zoo/raw/main/yolox/yolox-s.onnx
  DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/data
  MD5 1cf017aaf3b24d77bdc942c1223ca0f0
  FILENAME yolox-s.onnx
)

if(BUILD_TESTING)
  set(ament_cmake_clang_format_CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_uncrustify
  )
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
    data
    launch
)
