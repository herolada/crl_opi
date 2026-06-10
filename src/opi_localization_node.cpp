#include "opi_localization_node.hpp"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

namespace opi_localization
{

// ─────────────────────────────────────────────────────────────────────────────
OpiLocalizationNode::OpiLocalizationNode(const rclcpp::NodeOptions & options)
: Node("opi_localization_node", options)
{
  // ── Parameters ──────────────────────────────────────────────────────────
  adr_width_m_   = declare_parameter<double>("adr_width_m",   0.40);
  adr_height_m_  = declare_parameter<double>("adr_height_m",  0.30);
  drone_width_m_  = declare_parameter<double>("drone_width_m",  0.50);
  drone_height_m_ = declare_parameter<double>("drone_height_m", 0.50);
  camo_width_m_   = declare_parameter<double>("camo_width_m",   0.50);
  camo_height_m_  = declare_parameter<double>("camo_height_m",  1.80);

  map_frame_    = declare_parameter<std::string>("map_frame",    "map");
  camera_frame_ = declare_parameter<std::string>("camera_frame", "camera_optical_frame");

  std::string camera_info_topic =
    declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
  std::string input_topic  = declare_parameter<std::string>("input_topic",  "opi/detections");
  std::string output_topic = declare_parameter<std::string>("output_topic", "opi/positions_raw");

  // ── TF ──────────────────────────────────────────────────────────────────
  tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // ── ROS I/O ─────────────────────────────────────────────────────────────
  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic, rclcpp::SensorDataQoS(),
    std::bind(&OpiLocalizationNode::cameraInfoCallback, this, std::placeholders::_1));

  detections_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
    input_topic, rclcpp::SystemDefaultsQoS(),
    std::bind(&OpiLocalizationNode::detectionsCallback, this, std::placeholders::_1));

  detections_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(
    output_topic, rclcpp::SystemDefaultsQoS());

  RCLCPP_INFO(get_logger(),
    "OPI localization node ready. adr=%.2f×%.2f m, drone=%.2f×%.2f m, camo=%.2f×%.2f m",
    adr_width_m_, adr_height_m_,
    drone_width_m_, drone_height_m_,
    camo_width_m_, camo_height_m_);
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiLocalizationNode::cameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
{
  if (camera_info_received_) return;

  camera_frame_ = msg->header.frame_id;

  camera_matrix_ = (cv::Mat_<double>(3, 3) <<
    msg->k[0], msg->k[1], msg->k[2],
    msg->k[3], msg->k[4], msg->k[5],
    msg->k[6], msg->k[7], msg->k[8]);

  dist_coeffs_ = cv::Mat(1, static_cast<int>(msg->d.size()), CV_64F);
  for (size_t i = 0; i < msg->d.size(); ++i)
    dist_coeffs_.at<double>(0, static_cast<int>(i)) = msg->d[i];

  camera_info_received_ = true;
  RCLCPP_INFO(get_logger(), "CameraInfo received and stored.");
}

// ─────────────────────────────────────────────────────────────────────────────
std::array<cv::Point3f, 4> OpiLocalizationNode::getModelPoints(
  const std::string & class_id) const
{
  float w, h;
  if (class_id == "drone") {
    w = static_cast<float>(drone_width_m_);
    h = static_cast<float>(drone_height_m_);
  } else if (class_id == "camo") {
    w = static_cast<float>(camo_width_m_);
    h = static_cast<float>(camo_height_m_);
  } else {
    // "adr" and any unknown class fall back to ADR panel dimensions
    w = static_cast<float>(adr_width_m_);
    h = static_cast<float>(adr_height_m_);
  }

  float hw = w / 2.0f;
  float hh = h / 2.0f;
  // Order: top-left, top-right, bottom-right, bottom-left (origin at centre, z=0)
  return {{
    cv::Point3f(-hw,  hh, 0.0f),
    cv::Point3f( hw,  hh, 0.0f),
    cv::Point3f( hw, -hh, 0.0f),
    cv::Point3f(-hw, -hh, 0.0f),
  }};
}

// ─────────────────────────────────────────────────────────────────────────────
bool OpiLocalizationNode::solvePlacard(
  const vision_msgs::msg::Detection2D & det,
  cv::Vec3d & rvec, cv::Vec3d & tvec) const
{
  if (det.results.empty()) return false;

  const std::string & class_id = det.results[0].hypothesis.class_id;
  auto model_pts = getModelPoints(class_id);

  float cx = static_cast<float>(det.bbox.center.position.x);
  float cy = static_cast<float>(det.bbox.center.position.y);
  float hw = static_cast<float>(det.bbox.size_x / 2.0);
  float hh = static_cast<float>(det.bbox.size_y / 2.0);

  // Image corners in same order as model_points: TL, TR, BR, BL
  std::vector<cv::Point2f> image_points = {
    {cx - hw, cy - hh},
    {cx + hw, cy - hh},
    {cx + hw, cy + hh},
    {cx - hw, cy + hh},
  };

  std::vector<cv::Point3f> obj_pts(model_pts.begin(), model_pts.end());

  // IPPE — best for planar targets
  std::vector<cv::Vec3d> rvecs, tvecs;
  std::vector<double>    repr_errors;

  bool ok = cv::solvePnPGeneric(
    obj_pts, image_points,
    camera_matrix_, dist_coeffs_,
    rvecs, tvecs,
    false, cv::SOLVEPNP_IPPE,
    cv::noArray(), cv::noArray(),
    repr_errors);

  if (!ok || rvecs.empty()) return false;

  // IPPE returns two solutions; pick the one with positive z and lower reprojection error
  int best = 0;
  if (rvecs.size() > 1) {
    bool z0_positive = (tvecs[0][2] > 0.0);
    bool z1_positive = (tvecs[1][2] > 0.0);
    if (!z0_positive && z1_positive) {
      best = 1;
    } else if (z0_positive && z1_positive) {
      best = (repr_errors[0] <= repr_errors[1]) ? 0 : 1;
    }
  }

  if (tvecs[best][2] <= 0.0) {
    RCLCPP_DEBUG(get_logger(), "solvePnP: no valid solution with positive z.");
    return false;
  }

  rvec = rvecs[best];
  tvec = tvecs[best];
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
bool OpiLocalizationNode::transformToMap(
  const cv::Vec3d & /*rvec*/, const cv::Vec3d & tvec,
  const std_msgs::msg::Header & header,
  geometry_msgs::msg::Pose & pose_in_map) const
{
  geometry_msgs::msg::PoseStamped pose_camera;
  pose_camera.header.frame_id = camera_frame_;
  pose_camera.header.stamp    = header.stamp;
  pose_camera.pose.position.x = tvec[0];
  pose_camera.pose.position.y = tvec[1];
  pose_camera.pose.position.z = tvec[2];
  pose_camera.pose.orientation.w = 1.0;  // orientation not consumed downstream

  geometry_msgs::msg::PoseStamped pose_map;
  try {
    tf_buffer_->transform(pose_camera, pose_map, map_frame_,
                          tf2::durationFromSec(1.0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(get_logger(), "TF transform failed: %s", ex.what());
    return false;
  }

  pose_in_map = pose_map.pose;
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiLocalizationNode::detectionsCallback(
  const vision_msgs::msg::Detection2DArray::ConstSharedPtr & msg)
{
  if (!camera_info_received_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "Waiting for CameraInfo …");
    return;
  }
  if (msg->detections.empty()) return;

  vision_msgs::msg::Detection2DArray out_array;
  out_array.header = msg->header;

  for (const auto & det : msg->detections) {
    if (det.results.empty()) continue;

    cv::Vec3d rvec, tvec;
    if (!solvePlacard(det, rvec, tvec)) continue;

    geometry_msgs::msg::Pose pose_map;
    if (!transformToMap(rvec, tvec, msg->header, pose_map)) continue;

    // Republish this detection with the 3D pose filled in
    vision_msgs::msg::Detection2D out_det = det;
    out_det.results[0].pose.pose = pose_map;
    out_array.detections.push_back(out_det);
  }

  if (!out_array.detections.empty()) {
    detections_pub_->publish(out_array);
  }
}

}  // namespace opi_localization

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<opi_localization::OpiLocalizationNode>());
  rclcpp::shutdown();
  return 0;
}
