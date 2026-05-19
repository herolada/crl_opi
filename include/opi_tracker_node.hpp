#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <Eigen/Dense>

namespace opi_tracker
{

// ── Internal hypothesis ───────────────────────────────────────────────────────
struct OpiHypothesis
{
  uint32_t      id;
  Eigen::Vector3d centroid;       // running mean of all assigned measurements [m]
  uint32_t        count{0};       // number of measurements assigned so far
  rclcpp::Time    last_seen;      // stamp of the most recently assigned measurement
};

// ── Node ─────────────────────────────────────────────────────────────────────
class OpiTrackerNode : public rclcpp::Node
{
public:
  explicit OpiTrackerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // ── Parameters ────────────────────────────────────────────────────────────
  double   cluster_radius_m_;     // merge radius: new meas within this → same cluster
  uint32_t min_count_;            // minimum observations before hypothesis is reported
  double   prune_timeout_s_;      // remove hypothesis if not seen for this long

  std::string map_frame_;
  std::string output_topic_;
  std::string marker_topic_;

  // ── State ─────────────────────────────────────────────────────────────────
  std::map<uint32_t, OpiHypothesis> hypotheses_;
  uint32_t next_id_{0};

  // ── ROS I/O ───────────────────────────────────────────────────────────────
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr positions_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr    hypotheses_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;

  // Periodic publish timer
  rclcpp::TimerBase::SharedPtr publish_timer_;

  // ── Callbacks ─────────────────────────────────────────────────────────────
  void positionsCallback(const geometry_msgs::msg::PoseArray::ConstSharedPtr & msg);
  void publishTimerCallback();

  // ── Helpers ───────────────────────────────────────────────────────────────

  // Find the nearest hypothesis to a measurement.
  // Returns pointer (non-owning) or nullptr if none within cluster_radius_m_.
  OpiHypothesis * findNearest(const Eigen::Vector3d & meas);

  // Remove stale hypotheses.
  void pruneHypotheses(const rclcpp::Time & now);

  // Build and publish a MarkerArray for RViz visualisation.
  void publishMarkers(const rclcpp::Time & stamp);
};

}  // namespace opi_tracker
