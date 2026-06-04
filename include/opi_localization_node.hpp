#pragma once

#include <array>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <opencv2/opencv.hpp>

namespace opi_localization
{

class OpiLocalizationNode : public rclcpp::Node
{
public:
  explicit OpiLocalizationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // ── Parameters ────────────────────────────────────────────────────────────
  double adr_width_m_;     // ADR hazard panel physical width  [m]
  double adr_height_m_;    // ADR hazard panel physical height [m]
  double drone_width_m_;   // Drone physical width  [m]
  double drone_height_m_;  // Drone physical height [m]
  double camo_width_m_;    // Camouflaged person shoulder width [m]
  double camo_height_m_;   // Camouflaged person height [m]

  std::string map_frame_;
  std::string camera_frame_;

  // ── State ─────────────────────────────────────────────────────────────────
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  bool    camera_info_received_{false};

  // ── TF ────────────────────────────────────────────────────────────────────
  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // ── ROS I/O ───────────────────────────────────────────────────────────────
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr           camera_info_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr     detections_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr        detections_pub_;

  // ── Callbacks ─────────────────────────────────────────────────────────────
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg);
  void detectionsCallback(const vision_msgs::msg::Detection2DArray::ConstSharedPtr & msg);

  // ── Helpers ───────────────────────────────────────────────────────────────
  // Returns the 4 model corners [TL, TR, BR, BL] for the given class.
  std::array<cv::Point3f, 4> getModelPoints(const std::string & class_id) const;

  // Attempt to solve PnP with IPPE for a single detection.
  bool solvePlacard(const vision_msgs::msg::Detection2D & det,
                    cv::Vec3d & rvec, cv::Vec3d & tvec) const;

  // Transform a pose from camera frame to map frame via TF.
  bool transformToMap(const cv::Vec3d & rvec, const cv::Vec3d & tvec,
                      const std_msgs::msg::Header & header,
                      geometry_msgs::msg::Pose & pose_in_map) const;
};

}  // namespace opi_localization
