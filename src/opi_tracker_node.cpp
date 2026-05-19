#include "opi_tracker_node.hpp"

#include <limits>
#include <sstream>

namespace opi_tracker
{

// ─────────────────────────────────────────────────────────────────────────────
OpiTrackerNode::OpiTrackerNode(const rclcpp::NodeOptions & options)
: Node("opi_tracker_node", options)
{
  // ── Parameters ──────────────────────────────────────────────────────────
  cluster_radius_m_ = declare_parameter<double>("cluster_radius_m", 0.75);
  min_count_        = static_cast<uint32_t>(declare_parameter<int>("min_count", 5));
  prune_timeout_s_  = declare_parameter<double>("prune_timeout_s", 60.0);
  map_frame_        = declare_parameter<std::string>("map_frame",   "map");
  double publish_hz = declare_parameter<double>("publish_hz", 2.0);

  std::string input_topic  = declare_parameter<std::string>("input_topic",  "opi/positions_raw");
  output_topic_            = declare_parameter<std::string>("output_topic", "opi/hypotheses");
  marker_topic_            = declare_parameter<std::string>("marker_topic", "opi/markers");

  // ── ROS I/O ─────────────────────────────────────────────────────────────
  positions_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
    input_topic, rclcpp::SystemDefaultsQoS(),
    std::bind(&OpiTrackerNode::positionsCallback, this, std::placeholders::_1));

  hypotheses_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
    output_topic_, rclcpp::SystemDefaultsQoS());

  markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    marker_topic_, rclcpp::SystemDefaultsQoS());

  auto period = std::chrono::duration<double>(1.0 / publish_hz);
  publish_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&OpiTrackerNode::publishTimerCallback, this));

  RCLCPP_INFO(get_logger(),
    "OPI tracker node ready. cluster_radius=%.2f m, min_count=%u, prune=%.1f s",
    cluster_radius_m_, min_count_, prune_timeout_s_);
}

// ─────────────────────────────────────────────────────────────────────────────
OpiHypothesis * OpiTrackerNode::findNearest(const Eigen::Vector3d & meas)
{
  double         best_dist = cluster_radius_m_;
  OpiHypothesis * best_hyp  = nullptr;

  for (auto & [id, hyp] : hypotheses_) {
    double d = (hyp.centroid - meas).norm();
    if (d < best_dist) {
      best_dist = d;
      best_hyp  = &hyp;
    }
  }
  return best_hyp;
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::pruneHypotheses(const rclcpp::Time & now)
{
  std::vector<uint32_t> to_remove;
  for (const auto & [id, hyp] : hypotheses_) {
    double age = (now - hyp.last_seen).seconds();
    if (age > prune_timeout_s_) {
      to_remove.push_back(id);
    }
  }
  for (uint32_t id : to_remove) {
    RCLCPP_INFO(get_logger(), "Pruned stale OPI hypothesis id=%u", id);
    hypotheses_.erase(id);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::positionsCallback(
  const geometry_msgs::msg::PoseArray::ConstSharedPtr & msg)
{
  rclcpp::Time stamp(msg->header.stamp);
  // pruneHypotheses(stamp);

  for (const auto & pose : msg->poses) {
    Eigen::Vector3d meas(pose.position.x, pose.position.y, pose.position.z);

    OpiHypothesis * hyp = findNearest(meas);

    if (hyp != nullptr) {
      // Update running mean (online / incremental mean)
      ++hyp->count;
      hyp->centroid += (meas - hyp->centroid) / static_cast<double>(hyp->count);
      hyp->last_seen = stamp;
    } else {
      // Create new hypothesis
      OpiHypothesis new_hyp;
      new_hyp.id        = next_id_++;
      new_hyp.centroid  = meas;
      new_hyp.count     = 1;
      new_hyp.last_seen = stamp;
      hypotheses_[new_hyp.id] = new_hyp;
      RCLCPP_INFO(get_logger(), "New OPI hypothesis id=%u at [%.2f, %.2f, %.2f]",
                  new_hyp.id, meas.x(), meas.y(), meas.z());
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::publishTimerCallback()
{
  rclcpp::Time now = get_clock()->now();

  geometry_msgs::msg::PoseArray out;
  out.header.stamp    = now;
  out.header.frame_id = map_frame_;

  for (const auto & [id, hyp] : hypotheses_) {
    if (hyp.count < min_count_) continue;

    geometry_msgs::msg::Pose p;
    p.position.x    = hyp.centroid.x();
    p.position.y    = hyp.centroid.y();
    p.position.z    = hyp.centroid.z();
    p.orientation.w = 1.0;
    out.poses.push_back(p);
  }

  hypotheses_pub_->publish(out);
  publishMarkers(now);
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::publishMarkers(const rclcpp::Time & stamp)
{
  visualization_msgs::msg::MarkerArray marker_array;

  // Delete all old markers first
  visualization_msgs::msg::Marker delete_all;
  delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
  delete_all.header.frame_id = map_frame_;
  delete_all.header.stamp    = stamp;
  marker_array.markers.push_back(delete_all);

  for (const auto & [id, hyp] : hypotheses_) {
    if (hyp.count < min_count_) continue;

    // Sphere at centroid
    visualization_msgs::msg::Marker sphere;
    sphere.header.frame_id = map_frame_;
    sphere.header.stamp    = stamp;
    sphere.ns              = "opi_hypotheses";
    sphere.id              = static_cast<int>(id);
    sphere.type            = visualization_msgs::msg::Marker::SPHERE;
    sphere.action          = visualization_msgs::msg::Marker::ADD;
    sphere.pose.position.x = hyp.centroid.x();
    sphere.pose.position.y = hyp.centroid.y();
    sphere.pose.position.z = hyp.centroid.z();
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25;
    sphere.color.r = 1.0f; sphere.color.g = 0.5f; sphere.color.b = 0.0f;
    sphere.color.a = 0.9f;
    sphere.lifetime = rclcpp::Duration(0, 0);  // persist until deleted

    // Text label above sphere
    visualization_msgs::msg::Marker text;
    text.header        = sphere.header;
    text.ns            = "opi_labels";
    text.id            = static_cast<int>(id);
    text.type          = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action        = visualization_msgs::msg::Marker::ADD;
    text.pose          = sphere.pose;
    text.pose.position.z += 0.3;
    text.scale.z       = 0.20;
    text.color.r = text.color.g = text.color.b = 1.0f;
    text.color.a = 1.0f;
    std::ostringstream ss;
    ss << "OPI #" << id << "\n(n=" << hyp.count << ")";
    text.text = ss.str();
    text.lifetime = rclcpp::Duration(0, 0);

    marker_array.markers.push_back(sphere);
    marker_array.markers.push_back(text);
  }

  markers_pub_->publish(marker_array);
}

}  // namespace opi_tracker

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<opi_tracker::OpiTrackerNode>());
  rclcpp::shutdown();
  return 0;
}
