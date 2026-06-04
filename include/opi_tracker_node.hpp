#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <crl_opi/msg/tracked_pose_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <Eigen/Dense>

namespace opi_tracker
{

// ── Internal hypothesis ───────────────────────────────────────────────────────
struct OpiHypothesis
{
  uint32_t        id;
  std::string     class_id;           // "adr", "drone", or "camo"
  Eigen::Vector3d centroid;           // running mean of all assigned measurements [m]
  uint32_t        count{0};           // number of measurements assigned so far
  rclcpp::Time    last_seen;          // stamp of the most recently assigned measurement
  std::string     frame;
  bool            visited{false};     // true once the robot has reached this OPI
};

// ── Node ─────────────────────────────────────────────────────────────────────
class OpiTrackerNode : public rclcpp::Node
{
public:
  explicit OpiTrackerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // ── Parameters ────────────────────────────────────────────────────────────
  double   cluster_radius_m_;
  uint32_t min_count_;
  double   opi_reached_distance_;

  std::string map_frame_;
  std::string tracked_topic_;
  std::string goals_topic_;
  std::string marker_topic_;
  std::string image_topic_;
  std::string img_save_path_;

  // ── State ─────────────────────────────────────────────────────────────────
  std::map<uint32_t, OpiHypothesis> hypotheses_;
  uint32_t next_id_{0};
  geometry_msgs::msg::PoseStamped latest_robot_pose_;
  bool have_robot_pose_{false};

  // ── TF ────────────────────────────────────────────────────────────────────
  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // ── ROS I/O ───────────────────────────────────────────────────────────────
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detections_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr            odom_sub_;
  rclcpp::Publisher<crl_opi::msg::TrackedPoseArray>::SharedPtr        hypotheses_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr         unvisited_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr  markers_pub_;

  rclcpp::TimerBase::SharedPtr publish_timer_;

  // ── Callbacks ─────────────────────────────────────────────────────────────
  void detectionsCallback(const vision_msgs::msg::Detection2DArray::ConstSharedPtr & msg);
  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr & msg);
  void publishTimerCallback();

  // ── Helpers ───────────────────────────────────────────────────────────────
  void takePhoto(int id, const std::string & specifier);

  // Find the nearest hypothesis of the same class within cluster_radius_m_.
  OpiHypothesis * findNearest(const Eigen::Vector3d & meas, const std::string & class_id);

  // Transform the latest robot pose into the requested frame.
  bool getRobotPoseInFrame(
    const std::string & target_frame,
    const rclcpp::Time & stamp,
    geometry_msgs::msg::PoseStamped & robot_pose) const;

  // Build and publish a MarkerArray for RViz visualisation.
  void publishMarkers(const rclcpp::Time & stamp);

  // Returns per-class sphere colour (r, g, b).
  static std::tuple<float, float, float> classColor(const std::string & class_id);
};

}  // namespace opi_tracker
