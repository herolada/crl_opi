"""Launch the OPI detection, localization, and tracking pipeline."""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="simulation/bag or not",
        ),
        DeclareLaunchArgument(
            "rotate_image_180",
            default_value="false",
            description="Rotate the subscribed image by 180 degrees before detection",
        ),
        DeclareLaunchArgument(
            "detection_hz",
            default_value="1.0",
            description="Maximum object detection frequency per subscribed camera topic; 0.0 processes every frame",
        ),
        DeclareLaunchArgument(
            "model",
            default_value="best_dgx.onnx",
            description="ONNX model file for object detection",
        ),
    ]

    model_path = PathJoinSubstitution([
        FindPackageShare("crl_opi"),
        "models",
        LaunchConfiguration("model")
    ])

    return LaunchDescription(
        declared_args +
        [Node(
            package="crl_opi",
            executable="opi_detection_node",
            name="opi_detection_node",
            output="screen",
            parameters=[{
                "model_path": model_path,
                "camera_topics": ["camera/image_raw"],
                "class_names": ["adr", "drone", "camo"],
                "conf_threshold": 0.40,
                "nms_threshold": 0.45,
                "input_width": 416,
                "input_height": 416,
                "rotate_image_180": LaunchConfiguration("rotate_image_180"),
                "detection_hz": LaunchConfiguration("detection_hz"),
                "camera_info_topic": "camera/camera_info",
                "output_topic": "opi/detections",
                "use_sim_time": LaunchConfiguration("use_sim_time")}],
            remappings=[
                ("camera/image_raw", "/basler_front/image_raw"),
                ("camera/camera_info", "/basler_front/camera_info"),
            ],
        ),
        Node(
            package="crl_opi",
            executable="opi_localization_node",
            name="opi_localization_node",
            output="screen",
            parameters=[{
                "adr_width_m":   0.40,
                "adr_height_m":  0.30,
                "drone_width_m":  0.50,
                "drone_height_m": 0.50,
                "camo_width_m":   0.50,
                "camo_height_m":  1.80,
                "map_frame": "map",
                "camera_frame": "oak_rgb_camera_optical_frame",
                "camera_info_topic": "camera/camera_info",
                "input_topic": "opi/detections",
                "output_topic": "opi/positions_raw",
                "use_sim_time": LaunchConfiguration("use_sim_time")}],
            remappings=[
                ("camera/camera_info", "/basler_front/camera_info"),
                ("opi/detections", "/opi/detections"),
                ("opi/positions_raw", "/opi/positions_raw"),
            ],
        ),
        Node(
            package="crl_opi",
            executable="opi_tracker_node",
            name="opi_tracker_node",
            output="screen",
            parameters=[{
                "opi_reached_distance": 3.0,
                "cluster_radius_m": 5.0,
                "min_count": 1,
                "map_frame": "map",
                "publish_hz": 2.0,
                "input_topic": "opi/positions_raw",
                "tracked_topic": "opi/tracked",
                "goals_topic": "opi/goals",
                "marker_topic": "opi/markers",
                "odom_topic": "/liorf/mapping/baselink_odometry",
                "image_topic": "/basler_front/image_raw",
                "img_save_path": "/home/robot/opi_images/",
                "use_sim_time": LaunchConfiguration("use_sim_time")}],
            remappings=[
                ("opi/positions_raw", "/opi/positions_raw"),
                ("opi/goals", "/opi/goals"),
                ("opi/tracked", "/opi/tracked"),
                ("opi/markers", "/opi/markers"),
            ],
        ),
        # ros2 run tf2_ros static_transform_publisher --frame-id os_sensor --child-frame-id pylon_camera --x 0 --y 0 --z 0 --roll 3.14 --pitch -1.57 --yaw 0
        # TODO BUG: this makes the z coordinate be upside down for the localized OPIs
        # Node(
        #     package="tf2_ros",
        #     executable="static_transform_publisher",
        #     name="fake_pylon_camera_tf",
        #     output="screen",
        #     arguments=[
        #         "--frame-id", "os_sensor",
        #         "--child-frame-id", "pylon_camera",
        #         "--x", "0",
        #         "--y", "0",
        #         "--z", "0",
        #         "--roll", "3.14",
        #         "--pitch", "-1.57",
        #         "--yaw", "0",
        #     ],
        # )
        ]
    )
