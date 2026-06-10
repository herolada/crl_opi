#include "opi_tracker_node.hpp"

#include <cmath>
#include <sstream>
#include <tuple>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/wait_for_message.hpp>

#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/UTMUPS.hpp>

#include <sys/stat.h>
#include <sys/types.h>
#define MKDIR(path) mkdir(path, 0755)

namespace opi_tracker
{

// ─────────────────────────────────────────────────────────────────────────────
OpiTrackerNode::OpiTrackerNode(const rclcpp::NodeOptions & options)
: Node("opi_tracker_node", options)
{
  // ── Parameters ──────────────────────────────────────────────────────────
  cluster_radius_m_     = declare_parameter<double>("cluster_radius_m", 0.75);
  min_count_            = static_cast<uint32_t>(declare_parameter<int>("min_count", 5));
  opi_reached_distance_ = declare_parameter<double>("opi_reached_distance", 1.0);
  map_frame_            = declare_parameter<std::string>("map_frame", "map");
  double publish_hz     = declare_parameter<double>("publish_hz", 2.0);

  std::string input_topic = declare_parameter<std::string>("input_topic",  "opi/positions_raw");
  tracked_topic_          = declare_parameter<std::string>("tracked_topic", "opi/tracked");
  goals_topic_            = declare_parameter<std::string>("goals_topic",   "opi/goals");
  marker_topic_           = declare_parameter<std::string>("marker_topic",  "opi/markers");
  std::string odom_topic  = declare_parameter<std::string>("odom_topic",    "/odom");
  image_topic_            = declare_parameter("image_topic",    "/luxonis/oak/rgb/image_raw");
  img_save_path_          = declare_parameter("img_save_path",  "~/opi_images/");
  robot_frame_            = declare_parameter<std::string>("robot_frame", "os_sensor");
  MKDIR(img_save_path_.c_str());

  // CSV path defaults to <img_save_path>/opi_log.csv
  {
    std::string default_csv = img_save_path_;
    if (!default_csv.empty() &&
        default_csv.back() != '/' && default_csv.back() != '\\')
      default_csv += '/';
    default_csv += "opi_log.csv";
    csv_save_path_ = declare_parameter<std::string>("csv_save_path", default_csv);
  }

  // ── TF ──────────────────────────────────────────────────────────────────
  tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // ── ROS I/O ─────────────────────────────────────────────────────────────
  detections_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
    input_topic, rclcpp::SystemDefaultsQoS(),
    std::bind(&OpiTrackerNode::detectionsCallback, this, std::placeholders::_1));

  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    odom_topic, rclcpp::SystemDefaultsQoS(),
    std::bind(&OpiTrackerNode::odomCallback, this, std::placeholders::_1));

  hypotheses_pub_ = create_publisher<crl_opi::msg::TrackedPoseArray>(
    tracked_topic_, rclcpp::SystemDefaultsQoS());

  unvisited_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
    goals_topic_, rclcpp::SystemDefaultsQoS());

  markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    marker_topic_, rclcpp::SystemDefaultsQoS());

  auto period = std::chrono::duration<double>(1.0 / publish_hz);
  publish_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&OpiTrackerNode::publishTimerCallback, this));

  RCLCPP_INFO(get_logger(),
    "OPI tracker node ready. cluster_radius=%.2f m, min_count=%u, reached=%.2f m",
    cluster_radius_m_, min_count_, opi_reached_distance_);
}

// ─────────────────────────────────────────────────────────────────────────────
OpiHypothesis * OpiTrackerNode::findNearest(
  const Eigen::Vector3d & meas, const std::string & class_id)
{
  double          best_dist = cluster_radius_m_;
  OpiHypothesis * best_hyp  = nullptr;

  for (auto & [id, hyp] : hypotheses_) {
    if (hyp.class_id != class_id) continue;  // never merge across classes
    double d = (hyp.centroid - meas).norm();
    if (d < best_dist) {
      best_dist = d;
      best_hyp  = &hyp;
    }
  }
  return best_hyp;
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::detectionsCallback(
  const vision_msgs::msg::Detection2DArray::ConstSharedPtr & msg)
{
  rclcpp::Time stamp(msg->header.stamp);

  geometry_msgs::msg::PoseStamped robot_pose;
  const bool can_check_visited =
    have_robot_pose_ && getRobotPoseInFrame(msg->header.frame_id, stamp, robot_pose);

  for (const auto & det : msg->detections) {
    if (det.results.empty()) continue;

    const std::string & class_id = det.results[0].hypothesis.class_id;
    const auto & pos = det.results[0].pose.pose.position;
    Eigen::Vector3d meas(pos.x, pos.y, pos.z);

    OpiHypothesis * hyp = findNearest(meas, class_id);

    if (hyp != nullptr) {
      ++hyp->count;
      hyp->centroid += (meas - hyp->centroid) / static_cast<double>(hyp->count);
      hyp->last_seen = stamp;
    } else {
      OpiHypothesis new_hyp;
      new_hyp.id         = next_id_++;
      new_hyp.class_id   = class_id;
      new_hyp.centroid   = meas;
      new_hyp.count      = 1;
      new_hyp.first_seen = stamp;
      new_hyp.last_seen  = stamp;
      new_hyp.frame      = msg->header.frame_id;

      // Record pose in robot frame at first detection
      geometry_msgs::msg::PoseStamped ps_in, ps_out;
      ps_in.header = msg->header;
      ps_in.pose   = det.results[0].pose.pose;
      try {
        tf_buffer_->transform(ps_in, ps_out, robot_frame_, tf2::durationFromSec(0.1));
        new_hyp.robot_pos = Eigen::Vector3d(
          ps_out.pose.position.x, ps_out.pose.position.y, ps_out.pose.position.z);
        new_hyp.robot_ori = Eigen::Quaterniond(
          ps_out.pose.orientation.w, ps_out.pose.orientation.x,
          ps_out.pose.orientation.y, ps_out.pose.orientation.z);
        new_hyp.has_robot_frame_pose = true;
      } catch (const tf2::TransformException & ex) {
        RCLCPP_DEBUG(get_logger(),
          "Could not get robot-frame pose for new OPI id=%u: %s", new_hyp.id, ex.what());
      }

      hypotheses_[new_hyp.id] = new_hyp;
      hyp = &hypotheses_.at(new_hyp.id);

      RCLCPP_INFO(get_logger(), "New OPI hypothesis id=%u class=%s at [%.2f, %.2f, %.2f]",
                  hyp->id, class_id.c_str(), meas.x(), meas.y(), meas.z());
      hyp->image_filename = takePhoto(static_cast<int>(hyp->id), "new");
    }

    updateEcefAndUtm(*hyp);

    if (can_check_visited && !hyp->visited) {
      const double dx = hyp->centroid.x() - robot_pose.pose.position.x;
      const double dy = hyp->centroid.y() - robot_pose.pose.position.y;
      if (std::hypot(dx, dy) <= opi_reached_distance_) {
        hyp->visited = true;
        RCLCPP_INFO(get_logger(),
          "Marked OPI hypothesis id=%u class=%s visited", hyp->id, hyp->class_id.c_str());
      }
    }
  }

  writeCsv();
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr & msg)
{
  latest_robot_pose_.header = msg->header;
  latest_robot_pose_.pose   = msg->pose.pose;
  have_robot_pose_          = true;

  geometry_msgs::msg::PoseStamped robot_pose_transformed;

  for (auto & [id, hyp] : hypotheses_) {
    if (robot_pose_transformed == geometry_msgs::msg::PoseStamped() ||
        robot_pose_transformed.header.frame_id != hyp.frame)
    {
      rclcpp::Time stamp(msg->header.stamp);
      if (!getRobotPoseInFrame(hyp.frame, stamp, robot_pose_transformed)) continue;
    }

    if (!hyp.visited) {
      const double dx = hyp.centroid.x() - robot_pose_transformed.pose.position.x;
      const double dy = hyp.centroid.y() - robot_pose_transformed.pose.position.y;
      const double distance_xy = std::hypot(dx, dy);

      if (distance_xy <= opi_reached_distance_) {
        hyp.visited = true;
        RCLCPP_INFO(get_logger(),
          "Marked OPI hypothesis id=%u class=%s visited at XY distance %.2f m",
          hyp.id, hyp.class_id.c_str(), distance_xy);
        takePhoto(static_cast<int>(hyp.id), "closeup");
      } else {
        RCLCPP_INFO(get_logger(),
          "OPI too far id=%u class=%s at XY distance %.2f m",
          hyp.id, hyp.class_id.c_str(), distance_xy);
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::publishTimerCallback()
{
  rclcpp::Time now = get_clock()->now();

  crl_opi::msg::TrackedPoseArray out;
  geometry_msgs::msg::PoseArray  unvisited_out;
  out.header.stamp    = now;
  out.header.frame_id = map_frame_;
  unvisited_out.header = out.header;

  for (const auto & [id, hyp] : hypotheses_) {
    if (hyp.count < min_count_) continue;

    crl_opi::msg::TrackedPose tracked_pose;
    tracked_pose.id                  = id;
    tracked_pose.class_id            = hyp.class_id;
    tracked_pose.pose.position.x     = hyp.centroid.x();
    tracked_pose.pose.position.y     = hyp.centroid.y();
    tracked_pose.pose.position.z     = hyp.centroid.z();
    tracked_pose.pose.orientation.w  = 1.0;
    tracked_pose.visited             = hyp.visited;
    out.poses.push_back(tracked_pose);

    if (!hyp.visited) {
      unvisited_out.poses.push_back(tracked_pose.pose);
    }
  }

  hypotheses_pub_->publish(out);
  unvisited_pub_->publish(unvisited_out);
  publishMarkers(now);
}

// ─────────────────────────────────────────────────────────────────────────────
std::string OpiTrackerNode::takePhoto(int id, const std::string & specifier)
{
  RCLCPP_INFO(get_logger(), "Trying to take a photo of OPI %d.", id);
  sensor_msgs::msg::Image img_msg;
  auto timeout = std::chrono::seconds(1);
  bool received_msg = rclcpp::wait_for_message(img_msg, shared_from_this(), image_topic_, timeout);
  if (!received_msg) {
    RCLCPP_WARN(get_logger(),
      "Failed to take photo of the OPI! wait_for_msg did not receive a msg in %ld s",
      timeout.count());
    return "";
  }

  cv_bridge::CvImageConstPtr cv_img =
    cv_bridge::toCvShare(std::make_shared<sensor_msgs::msg::Image>(img_msg), img_msg.encoding);

  if (cv_img->image.empty()) {
    RCLCPP_ERROR(get_logger(), "The image is empty.");
    return "";
  }

  std::string separator =
    (!img_save_path_.empty() &&
     img_save_path_.back() != '/' &&
     img_save_path_.back() != '\\')
        ? "/"
        : "";

  std::string save_path =
    img_save_path_ + separator + "OPI_" + std::to_string(id) + "_" + specifier + ".png";
  cv::imwrite(save_path, cv_img->image);
  RCLCPP_INFO(get_logger(), "Saved photo of OPI %d to %s", id, save_path.c_str());
  return save_path;
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::updateEcefAndUtm(OpiHypothesis & hyp)
{
  try {
    geometry_msgs::msg::PointStamped pt_in, pt_out;
    pt_in.header.frame_id = hyp.frame;
    pt_in.header.stamp    = hyp.last_seen;
    pt_in.point.x         = hyp.centroid.x();
    pt_in.point.y         = hyp.centroid.y();
    pt_in.point.z         = hyp.centroid.z();
    tf_buffer_->transform(pt_in, pt_out, "earth", tf2::durationFromSec(0.1));
    hyp.ecef_pos = Eigen::Vector3d(pt_out.point.x, pt_out.point.y, pt_out.point.z);

    double lat, lon, alt;
    GeographicLib::Geocentric::WGS84().Reverse(
      hyp.ecef_pos.x(), hyp.ecef_pos.y(), hyp.ecef_pos.z(), lat, lon, alt);
    GeographicLib::UTMUPS::Forward(lat, lon,
      hyp.utm_zone, hyp.utm_northp, hyp.utm_easting, hyp.utm_northing);
    hyp.utm_alt = alt;
    hyp.has_ecef = true;
  } catch (const std::exception & ex) {
    RCLCPP_DEBUG(get_logger(),
      "ECEF/UTM update failed for id=%u: %s", hyp.id, ex.what());
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::writeCsv()
{
  std::ofstream f(csv_save_path_, std::ios::trunc);
  if (!f.is_open()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
      "Cannot write OPI CSV to '%s'", csv_save_path_.c_str());
    return;
  }

  f << "id,class_id,timestamp_first_s,"
       "robot_x,robot_y,robot_z,robot_qx,robot_qy,robot_qz,robot_qw,"
       "ecef_x,ecef_y,ecef_z,"
       "utm_zone,utm_northp,utm_easting,utm_northing,utm_alt,"
       "image_filename\n";
  f << std::fixed;

  for (const auto & [id, hyp] : hypotheses_) {
    f << id << ","
      << hyp.class_id << ","
      << std::setprecision(3) << hyp.first_seen.seconds() << ",";

    if (hyp.has_robot_frame_pose) {
      f << std::setprecision(4)
        << hyp.robot_pos.x() << "," << hyp.robot_pos.y() << "," << hyp.robot_pos.z() << ","
        << hyp.robot_ori.x() << "," << hyp.robot_ori.y() << "," << hyp.robot_ori.z() << ","
        << hyp.robot_ori.w();
    } else {
      f << ",,,,,,";   // 6 commas for 7 empty fields (rx,ry,rz,qx,qy,qz,qw)
    }
    f << ",";

    if (hyp.has_ecef) {
      f << std::setprecision(3)
        << hyp.ecef_pos.x() << "," << hyp.ecef_pos.y() << "," << hyp.ecef_pos.z() << ","
        << hyp.utm_zone << "," << (hyp.utm_northp ? "N" : "S") << ","
        << std::setprecision(2)
        << hyp.utm_easting << "," << hyp.utm_northing << ","
        << std::setprecision(3) << hyp.utm_alt;
    } else {
      f << ",,,,,,,";   // 7 commas for 8 empty fields (ex,ey,ez,zone,northp,ue,un,ua)
    }
    f << "," << hyp.image_filename << "\n";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
bool OpiTrackerNode::getRobotPoseInFrame(
  const std::string & target_frame,
  const rclcpp::Time & /*stamp*/,
  geometry_msgs::msg::PoseStamped & robot_pose) const
{
  if (!have_robot_pose_) return false;

  if (target_frame.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "Cannot compare OPI and robot pose: empty target frame.");
    return false;
  }

  if (latest_robot_pose_.header.frame_id == target_frame) {
    robot_pose = latest_robot_pose_;
    return true;
  }

  try {
    tf_buffer_->transform(latest_robot_pose_, robot_pose, target_frame,
                          tf2::durationFromSec(0.1));
    return true;
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "Failed to transform robot pose from '%s' to '%s': %s",
      latest_robot_pose_.header.frame_id.c_str(), target_frame.c_str(), ex.what());
    return false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
std::tuple<float, float, float> OpiTrackerNode::classColor(const std::string & class_id)
{
  if (class_id == "drone") return {0.20f, 0.50f, 0.90f};   // blue
  if (class_id == "camo")  return {0.20f, 0.75f, 0.20f};   // green
  return {0.74f, 0.25f, 0.74f};                             // purple (adr + fallback)
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::publishMarkers(const rclcpp::Time & stamp)
{
  visualization_msgs::msg::MarkerArray marker_array;

  visualization_msgs::msg::Marker delete_all;
  delete_all.action          = visualization_msgs::msg::Marker::DELETEALL;
  delete_all.header.frame_id = map_frame_;
  delete_all.header.stamp    = stamp;
  marker_array.markers.push_back(delete_all);

  for (const auto & [id, hyp] : hypotheses_) {
    if (hyp.count < min_count_) continue;

    auto [r, g, b] = classColor(hyp.class_id);

    // Sphere at centroid
    visualization_msgs::msg::Marker sphere;
    sphere.header.frame_id    = map_frame_;
    sphere.header.stamp       = stamp;
    sphere.ns                 = "opi_hypotheses";
    sphere.id                 = static_cast<int>(id);
    sphere.type               = visualization_msgs::msg::Marker::SPHERE;
    sphere.action             = visualization_msgs::msg::Marker::ADD;
    sphere.pose.position.x    = hyp.centroid.x();
    sphere.pose.position.y    = hyp.centroid.y();
    sphere.pose.position.z    = hyp.centroid.z();
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.5;
    sphere.color.r = r; sphere.color.g = g; sphere.color.b = b;
    sphere.color.a = 0.9f;
    sphere.lifetime = rclcpp::Duration(0, 0);

    // Text label above sphere
    visualization_msgs::msg::Marker text;
    text.header        = sphere.header;
    text.ns            = "opi_labels";
    text.id            = static_cast<int>(id);
    text.type          = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action        = visualization_msgs::msg::Marker::ADD;
    text.pose          = sphere.pose;
    text.pose.position.z += 0.6;
    text.scale.z       = 0.40;
    text.color.r = text.color.g = text.color.b = 0.16f;
    text.color.a = 1.0f;
    std::ostringstream ss;
    ss << hyp.class_id << " #" << id << "\n(n=" << hyp.count << ")";
    text.text    = ss.str();
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
