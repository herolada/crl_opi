cmake_minimum_required(VERSION 3.8)
project(crl_opi)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(GeographicLib REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackedPose.msg"
  "msg/TrackedPoseArray.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

# ONNX Runtime – use the local source/build tree by default, but allow overrides
if(DEFINED ENV{ONNXRUNTIME_ROOT} AND NOT "$ENV{ONNXRUNTIME_ROOT}" STREQUAL "")
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
else()
  set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")
endif()
set(ONNXRUNTIME_BUILD_DIR "${ONNXRUNTIME_ROOT}/build/Linux/Release")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "Could not find ONNX Runtime header: ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
endif()
find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime libonnxruntime.so
  HINTS "${ONNXRUNTIME_BUILD_DIR}" "${ONNXRUNTIME_ROOT}/lib"
  REQUIRED)

# ── DETECTION NODE ────────────────────────────────────────────────────────────────────
add_executable(opi_detection_node src/opi_detection_node.cpp)

target_include_directories(opi_detection_node PRIVATE
  include
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(opi_detection_node
  rclcpp
  sensor_msgs
  vision_msgs
  cv_bridge
  image_transport
  OpenCV
)

target_link_libraries(opi_detection_node
  ${ONNXRUNTIME_LIBRARY}
  ${OpenCV_LIBS}
)

# ── LOCALIZATION NODE ────────────────────────────────────────────────────────────────────
add_executable(opi_localization_node src/opi_localization_node.cpp)
 
target_include_directories(opi_localization_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
 
ament_target_dependencies(opi_localization_node
  rclcpp
  sensor_msgs
  vision_msgs
  nav_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  OpenCV
)
 
target_link_libraries(opi_localization_node ${OpenCV_LIBS})

# ── TRACKER NODE ────────────────────────────────────────────────────────────────────
add_executable(opi_tracker_node src/opi_tracker_node.cpp)
 
target_include_directories(opi_tracker_node PRIVATE
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${GeographicLib_INCLUDE_DIRS}
)
 
ament_target_dependencies(opi_tracker_node
  rclcpp
  cv_bridge
  sensor_msgs
  vision_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
  std_msgs
)

target_link_libraries(opi_tracker_node
  ${GeographicLib_LIBRARIES}
)

rosidl_target_interfaces(opi_tracker_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")


install(TARGETS opi_detection_node 
                opi_localization_node
                opi_tracker_node
        DESTINATION lib/${PROJECT_NAME})
        
install(DIRECTORY launch models DESTINATION share/${PROJECT_NAME} OPTIONAL)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
