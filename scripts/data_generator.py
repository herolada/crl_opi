#!/usr/bin/env python3
"""Generate synthetic ADR hazard panel detections from ROS 2 MCAP image topics.

This script samples frames from a ROS 2 bag, inserts a synthetic ADR plate with
randomized pose and appearance, and exports YOLO detection labels alongside the
composited images.

Example:
    python3 crl_opi/scripts/data_generator.py \
        --bag /path/to/recording \
        --topic /camera/image_raw \
        --output-dir crl_opi/data/synth_adr \
        --sample-every-seconds 10
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from rclpy.serialization import deserialize_message
import rosbag2_py
from cv_bridge import CvBridge
from rosidl_runtime_py.utilities import get_message

@dataclass(frozen=True)
class FrameSample:
    image: np.ndarray
    timestamp_seconds: float
    sequence_index: int


@dataclass(frozen=True)
class PlateAppearance:
    fill_rgba: tuple[int, int, int, int]
    upper_digits: str
    lower_digits: str
    border_scale: float
    digit_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample frames from a ROS 2 MCAP bag, insert a synthetic ADR hazard plate, "
            "and export YOLO labels."
        )
    )
    parser.add_argument("--bag", type=Path, required=True, help="Path to the ROS 2 bag directory or MCAP file.")
    parser.add_argument("--topic", required=True, help="Camera image topic to read from the bag.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument(
        "--sample-every-seconds",
        type=float,
        default=10.0,
        help="Minimum time between sampled images from the selected topic.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of generated images.",
    )
    parser.add_argument(
        "--storage-id",
        default="mcap",
        help='ROS bag storage id, e.g. "mcap" for ROS 2 Jazzy/Kilted MCAP bags.',
    )
    parser.add_argument(
        "--flip",
        choices=("none", "rotate180", "horizontal", "vertical"),
        default="none",
        help="Optional image flip/rotation applied to each decoded frame before compositing.",
    )
    parser.add_argument(
        "--filename-prefix",
        default="adr_synth",
        help="Prefix for generated image and label filenames.",
    )
    parser.add_argument(
        "--image-extension",
        choices=("jpg", "png"),
        default="png",
        help="Image format for saved samples.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality when using .jpg output.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")

    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help='YOLO class id to write. Default matches a single-class "hazmat_plate" dataset.',
    )

    parser.add_argument(
        "--plate-width-m",
        type=float,
        default=0.40,
        help="Physical ADR plate width in meters.",
    )
    parser.add_argument(
        "--plate-height-m",
        type=float,
        default=0.30,
        help="Physical ADR plate height in meters.",
    )
    parser.add_argument(
        "--border-thickness-m",
        type=float,
        default=0.02,
        help="Physical thickness of the outer border and middle divider in meters.",
    )
    parser.add_argument(
        "--digit-height-m",
        type=float,
        default=0.10,
        help="Approximate physical digit height in meters.",
    )
    parser.add_argument(
        "--plate-color-min",
        default="d86a53ff",
        help="Minimum plate fill color as 8-digit RGBA hex.",
    )
    parser.add_argument(
        "--plate-color-max",
        default="efad00ff",
        help="Maximum plate fill color as 8-digit RGBA hex.",
    )
    parser.add_argument(
        "--plate-render-width",
        type=int,
        default=800,
        help="Texture width used when rendering the synthetic plate before warping.",
    )

    parser.add_argument(
        "--min-plate-width-px",
        type=float,
        default=50.0,
        help="Minimum target plate width in pixels.",
    )
    parser.add_argument(
        "--plate-width-mean-frac",
        type=float,
        default=0.10,
        help="Mean target plate width as a fraction of image width.",
    )
    parser.add_argument(
        "--plate-width-std-frac",
        type=float,
        default=0.08,
        help="Standard deviation of target plate width as a fraction of image width.",
    )
    parser.add_argument(
        "--max-plate-width-frac",
        type=float,
        default=0.5,
        help="Maximum target plate width as a fraction of image width.",
    )
    parser.add_argument(
        "--center-y-mean-frac",
        type=float,
        default=0.5,
        help="Mean plate y center as a fraction of image height.",
    )
    parser.add_argument(
        "--center-y-std-frac",
        type=float,
        default=0.18,
        help="Standard deviation of plate y center as a fraction of image height.",
    )
    parser.add_argument(
        "--roll-deg-range",
        type=float,
        default=45.0,
        help="Roll angle sampled uniformly from [-range, range] degrees.",
    )
    parser.add_argument(
        "--pitch-deg-range",
        type=float,
        default=65.0,
        help="Pitch angle sampled uniformly from [-range, range] degrees.",
    )
    parser.add_argument(
        "--yaw-deg-range",
        type=float,
        default=65.0,
        help="Yaw angle sampled uniformly from [-range, range] degrees.",
    )
    parser.add_argument(
        "--focal-length-x-px",
        type=float,
        default=None,
        help="Approximate camera focal length in pixels. Defaults to 1.2 * image width.",
    )
    parser.add_argument(
        "--focal-length-y-px",
        type=float,
        default=None,
        help="Approximate vertical focal length in pixels. Defaults to focal_length_x_px.",
    )
    parser.add_argument(
        "--max-placement-attempts",
        type=int,
        default=50,
        help="How many random placements to try per frame before giving up.",
    )
    parser.add_argument(
        "--min-visible-area-frac",
        type=float,
        default=0.85,
        help="Minimum fraction of the projected plate area that must remain inside the image.",
    )
    parser.add_argument(
        "--min-box-size-px",
        type=float,
        default=12.0,
        help="Minimum bounding box width and height in pixels after projection.",
    )

    parser.add_argument(
        "--plate-blur-kernel",
        type=int,
        default=3,
        help="Optional Gaussian blur kernel size applied to the warped plate. Use 0 to disable.",
    )
    parser.add_argument(
        "--plate-noise-std",
        type=float,
        default=5.0,
        help="Gaussian noise standard deviation added to the warped plate pixels.",
    )
    parser.add_argument(
        "--plate-brightness-jitter",
        type=float,
        default=0.12,
        help="Brightness multiplier jitter. Sampled uniformly from [1-j, 1+j].",
    )
    parser.add_argument(
        "--border-scale-jitter",
        type=float,
        default=0.15,
        help="Uniform border thickness jitter. Sampled from [1-j, 1+j].",
    )
    parser.add_argument(
        "--digit-scale-jitter",
        type=float,
        default=0.12,
        help="Uniform text size jitter. Sampled from [1-j, 1+j].",
    )
    return parser.parse_args()


def parse_rgba_hex(value: str) -> np.ndarray:
    cleaned = value.strip().removeprefix("#")
    if len(cleaned) != 8:
        raise ValueError(f"Expected 8 hex digits for RGBA color, got {value!r}.")
    return np.array([int(cleaned[i : i + 2], 16) for i in range(0, 8, 2)], dtype=np.float32)


def random_plate_color(rng: np.random.Generator, min_rgba: np.ndarray, max_rgba: np.ndarray) -> tuple[int, int, int, int]:
    t = float(rng.uniform(0.0, 1.0))
    rgba = np.round(min_rgba + t * (max_rgba - min_rgba)).astype(np.uint8)
    return int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3])


def fit_text_scale(text: str, font_face: int, target_height: int, thickness: int) -> float:
    scale = 1.0
    for _ in range(20):
        (_, height), baseline = cv2.getTextSize(text, font_face, scale, thickness)
        measured = height + baseline
        if measured <= 0:
            break
        scale *= target_height / measured
        if abs(measured - target_height) < 2:
            break
    return max(scale, 0.1)


def render_plate_texture(
    appearance: PlateAppearance,
    render_width: int,
    plate_width_m: float,
    plate_height_m: float,
    border_thickness_m: float,
    digit_height_m: float,
) -> np.ndarray:
    render_height = int(round(render_width * plate_height_m / plate_width_m))
    border_px = max(
        2,
        int(round(render_width * border_thickness_m * appearance.border_scale / plate_width_m)),
    )
    digit_height_px = max(
        8,
        int(round(render_height * digit_height_m * appearance.digit_scale / plate_height_m)),
    )

    plate = np.zeros((render_height, render_width, 4), dtype=np.uint8)
    plate[:, :, 0] = appearance.fill_rgba[2]
    plate[:, :, 1] = appearance.fill_rgba[1]
    plate[:, :, 2] = appearance.fill_rgba[0]
    plate[:, :, 3] = appearance.fill_rgba[3]

    black = (0, 0, 0, 255)
    cv2.rectangle(plate, (0, 0), (render_width - 1, render_height - 1), black, border_px)
    divider_y = render_height // 2
    cv2.rectangle(
        plate,
        (0, max(0, divider_y - border_px // 2)),
        (render_width - 1, min(render_height - 1, divider_y + math.ceil(border_px / 2))),
        black,
        thickness=-1,
    )

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = max(2, border_px // 2)
    upper_scale = fit_text_scale(appearance.upper_digits, font_face, digit_height_px, font_thickness)
    lower_scale = fit_text_scale(appearance.lower_digits, font_face, digit_height_px, font_thickness)

    upper_size, upper_baseline = cv2.getTextSize(
        appearance.upper_digits, font_face, upper_scale, font_thickness
    )
    lower_size, lower_baseline = cv2.getTextSize(
        appearance.lower_digits, font_face, lower_scale, font_thickness
    )

    upper_box_left = render_width // 2
    upper_box_top = border_px
    upper_box_right = render_width - border_px
    upper_box_bottom = divider_y - border_px

    upper_x = upper_box_right - upper_size[0] - int(0.08 * render_width)
    upper_y = upper_box_top + (upper_box_bottom - upper_box_top + upper_size[1]) // 2
    upper_x = max(upper_box_left + border_px, upper_x)

    lower_box_top = divider_y + border_px
    lower_box_bottom = render_height - border_px
    lower_x = (render_width - lower_size[0]) // 2
    lower_y = lower_box_top + (lower_box_bottom - lower_box_top + lower_size[1]) // 2

    cv2.putText(
        plate,
        appearance.upper_digits,
        (upper_x, upper_y),
        font_face,
        upper_scale,
        black,
        font_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        plate,
        appearance.lower_digits,
        (lower_x, lower_y),
        font_face,
        lower_scale,
        black,
        font_thickness,
        cv2.LINE_AA,
    )

    alpha = plate[:, :, 3:4].astype(np.float32) / 255.0
    plate[:, :, :3] = np.round(plate[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
    return plate


def euler_xyz_degrees_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def project_plate_corners(
    image_width: int,
    image_height: int,
    plate_width_m: float,
    plate_height_m: float,
    focal_length_x_px: float,
    focal_length_y_px: float,
    target_center_x_px: float,
    target_center_y_px: float,
    target_width_px: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray | None:
    cx = image_width / 2.0
    cy = image_height / 2.0
    z_m = focal_length_x_px * plate_width_m / max(target_width_px, 1e-6)
    tx = (target_center_x_px - cx) * z_m / focal_length_x_px
    ty = (target_center_y_px - cy) * z_m / focal_length_y_px

    corners_local = np.array(
        [
            [-plate_width_m / 2.0, -plate_height_m / 2.0, 0.0],
            [plate_width_m / 2.0, -plate_height_m / 2.0, 0.0],
            [plate_width_m / 2.0, plate_height_m / 2.0, 0.0],
            [-plate_width_m / 2.0, plate_height_m / 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    rotation = euler_xyz_degrees_to_matrix(roll_deg, pitch_deg, yaw_deg)
    corners_camera = (rotation @ corners_local.T).T + np.array([tx, ty, z_m], dtype=np.float32)

    if np.any(corners_camera[:, 2] <= 1e-4):
        return None

    projected = np.empty((4, 2), dtype=np.float32)
    projected[:, 0] = focal_length_x_px * (corners_camera[:, 0] / corners_camera[:, 2]) + cx
    projected[:, 1] = focal_length_y_px * (corners_camera[:, 1] / corners_camera[:, 2]) + cy
    return projected


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def create_visibility_masks(image_shape: tuple[int, int], projected_corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    polygon = np.round(projected_corners).astype(np.int32)
    plate_mask = np.zeros((height, width), dtype=np.uint8)
    visible_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(plate_mask, polygon, 255)

    clipped = projected_corners.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    cv2.fillConvexPoly(visible_mask, np.round(clipped).astype(np.int32), 255)
    return plate_mask, visible_mask


def sample_projected_plate(
    rng: np.random.Generator,
    image_width: int,
    image_height: int,
    args: argparse.Namespace,
) -> np.ndarray:
    focal_length_x_px = args.focal_length_x_px or (1.2 * image_width)
    focal_length_y_px = args.focal_length_y_px or focal_length_x_px
    max_plate_width_px = max(args.min_plate_width_px, args.max_plate_width_frac * image_width)
    mean_plate_width_px = args.plate_width_mean_frac * image_width
    std_plate_width_px = args.plate_width_std_frac * image_width

    for _ in range(args.max_placement_attempts):
        center_x_px = float(rng.uniform(0.0, image_width - 1))
        center_y_px = float(
            np.clip(
                rng.normal(args.center_y_mean_frac * image_height, args.center_y_std_frac * image_height),
                0.0,
                image_height - 1,
            )
        )
        target_width_px = float(
            np.clip(
                rng.normal(mean_plate_width_px, std_plate_width_px),
                args.min_plate_width_px,
                max_plate_width_px,
            )
        )
        roll_deg = float(rng.uniform(-args.roll_deg_range, args.roll_deg_range))
        pitch_deg = float(rng.uniform(-args.pitch_deg_range, args.pitch_deg_range))
        yaw_deg = float(rng.uniform(-args.yaw_deg_range, args.yaw_deg_range))

        projected = project_plate_corners(
            image_width=image_width,
            image_height=image_height,
            plate_width_m=args.plate_width_m,
            plate_height_m=args.plate_height_m,
            focal_length_x_px=focal_length_x_px,
            focal_length_y_px=focal_length_y_px,
            target_center_x_px=center_x_px,
            target_center_y_px=center_y_px,
            target_width_px=target_width_px,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
        )
        if projected is None:
            continue

        x_min = float(np.min(projected[:, 0]))
        y_min = float(np.min(projected[:, 1]))
        x_max = float(np.max(projected[:, 0]))
        y_max = float(np.max(projected[:, 1]))
        box_width = x_max - x_min
        box_height = y_max - y_min
        if box_width < args.min_box_size_px or box_height < args.min_box_size_px:
            continue

        full_area = polygon_area(projected)
        if full_area <= 1.0:
            continue

        _, visible_mask = create_visibility_masks((image_height, image_width), projected)
        visible_area = float(np.count_nonzero(visible_mask))
        if visible_area / full_area < args.min_visible_area_frac:
            continue

        return projected

    raise RuntimeError(
        "Could not find a valid plate placement. Try relaxing angle/visibility/size parameters."
    )


def apply_plate_appearance_jitter(
    rng: np.random.Generator,
    warped_rgba: np.ndarray,
    blur_kernel: int,
    noise_std: float,
    brightness_jitter: float,
) -> np.ndarray:
    result = warped_rgba.copy()
    if blur_kernel and blur_kernel > 1:
        kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        result[:, :, :3] = cv2.GaussianBlur(result[:, :, :3], (kernel, kernel), 0)
        result[:, :, 3] = cv2.GaussianBlur(result[:, :, 3], (kernel, kernel), 0)

    brightness = float(rng.uniform(1.0 - brightness_jitter, 1.0 + brightness_jitter))
    result[:, :, :3] = np.clip(result[:, :, :3].astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    if noise_std > 0.0:
        noise = rng.normal(0.0, noise_std, size=result[:, :, :3].shape).astype(np.float32)
        result[:, :, :3] = np.clip(result[:, :, :3].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return result


def warp_plate_to_image(
    plate_rgba: np.ndarray,
    image_width: int,
    image_height: int,
    projected_corners: np.ndarray,
) -> np.ndarray:
    src_h, src_w = plate_rgba.shape[:2]
    src_corners = np.array(
        [[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src_corners, projected_corners.astype(np.float32))
    return cv2.warpPerspective(
        plate_rgba,
        homography,
        (image_width, image_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def composite_rgba_over_bgr(background_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    background = background_bgr.astype(np.float32)
    overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    composited = overlay_rgb + background * (1.0 - alpha)
    return np.clip(composited, 0, 255).astype(np.uint8)


def yolo_box_from_corners(projected_corners: np.ndarray, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x_min = float(np.clip(np.min(projected_corners[:, 0]), 0, image_width - 1))
    x_max = float(np.clip(np.max(projected_corners[:, 0]), 0, image_width - 1))
    y_min = float(np.clip(np.min(projected_corners[:, 1]), 0, image_height - 1))
    y_max = float(np.clip(np.max(projected_corners[:, 1]), 0, image_height - 1))

    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0

    return (
        x_center / image_width,
        y_center / image_height,
        width / image_width,
        height / image_height,
    )


def format_timestamp(nanoseconds: int) -> float:
    return nanoseconds / 1e9


def extract_image_header_time_seconds(message, fallback_seconds: float) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return fallback_seconds
    sec = float(getattr(stamp, "sec", 0))
    nanosec = float(getattr(stamp, "nanosec", 0))
    return sec + nanosec / 1e9


def iter_ros_images_from_bag(
    bag_path: Path,
    topic_name: str,
    storage_id: str,
    sample_every_seconds: float,
    max_samples: int | None,
    flip_mode: str,
) -> Iterator[FrameSample]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise ImportError(
            "ROS 2 Python dependencies are required. Source your ROS 2 environment and make "
            "sure rosbag2_py, rclpy, and rosidl_runtime_py are available."
        ) from exc

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    if topic_name not in topic_types:
        available = ", ".join(sorted(topic_types)) or "<none>"
        raise ValueError(f"Topic {topic_name!r} not found in bag. Available topics: {available}")

    topic_type = topic_types[topic_name]
    message_type = get_message(topic_type)
    previous_sample_time: float | None = None
    emitted = 0
    sequence_index = 0

    while reader.has_next():
        current_topic, raw_data, timestamp_ns = reader.read_next()
        if current_topic != topic_name:
            continue

        ros_message = deserialize_message(raw_data, message_type)
        message_time = extract_image_header_time_seconds(ros_message, format_timestamp(timestamp_ns))
        if previous_sample_time is not None and (message_time - previous_sample_time) < sample_every_seconds:
            continue

        try:
            image = ros_image_to_bgr(ros_message, topic_type)
        except Exception as exc:
            raise RuntimeError(f"Failed to convert image message from topic {topic_name!r}: {exc}") from exc
        image = apply_image_flip(image, flip_mode)

        yield FrameSample(image=image, timestamp_seconds=message_time, sequence_index=sequence_index)
        previous_sample_time = message_time
        emitted += 1
        sequence_index += 1

        if max_samples is not None and emitted >= max_samples:
            break


def ros_image_to_bgr(message, topic_type: str) -> np.ndarray:
    if topic_type == "sensor_msgs/msg/CompressedImage":
        encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV could not decode compressed image bytes.")
        return image

    if topic_type == "sensor_msgs/msg/Image":
        try:
            from cv_bridge import CvBridge
        except ImportError as exc:
            raise ImportError(
                "cv_bridge is required for raw sensor_msgs/msg/Image topics."
            ) from exc
        return CvBridge().imgmsg_to_cv2(message, desired_encoding="bgr8")

    raise ValueError(
        f"Unsupported topic type {topic_type!r}. Expected sensor_msgs/msg/Image "
        "or sensor_msgs/msg/CompressedImage."
    )


def apply_image_flip(image: np.ndarray, flip_mode: str) -> np.ndarray:
    if flip_mode == "none":
        return image
    if flip_mode == "rotate180":
        return cv2.rotate(image, cv2.ROTATE_180)
    if flip_mode == "horizontal":
        return cv2.flip(image, 1)
    if flip_mode == "vertical":
        return cv2.flip(image, 0)
    raise ValueError(f"Unsupported flip mode {flip_mode!r}")


def write_image(image_path: Path, image: np.ndarray, image_extension: str, jpeg_quality: int) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if image_extension == "jpg":
        success = cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    else:
        success = cv2.imwrite(str(image_path), image)
    if not success:
        raise RuntimeError(f"Failed to write image to {image_path}")


def write_yolo_label(label_path: Path, class_id: int, yolo_box: tuple[float, float, float, float]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    x_center, y_center, width, height = yolo_box
    label_path.write_text(
        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
        encoding="utf-8",
    )


def sanitize_name_part(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    sanitized = sanitized.strip("_")
    return sanitized or "unknown"


def bag_name_from_path(bag_path: Path) -> str:
    if bag_path.is_file():
        return sanitize_name_part(bag_path.stem)
    return sanitize_name_part(bag_path.name)


def topic_name_fragment(topic_name: str) -> str:
    parts = [part for part in topic_name.split("/") if part]
    if len(parts) >= 2:
        return sanitize_name_part(parts[1])
    if parts:
        return sanitize_name_part(parts[0])
    return "topic"


def build_plate_appearance(
    rng: np.random.Generator,
    min_rgba: np.ndarray,
    max_rgba: np.ndarray,
    border_scale_jitter: float,
    digit_scale_jitter: float,
) -> PlateAppearance:
    return PlateAppearance(
        fill_rgba=random_plate_color(rng, min_rgba, max_rgba),
        upper_digits=f"{int(rng.integers(0, 100)):02d}",
        lower_digits=f"{int(rng.integers(0, 10000)):04d}",
        border_scale=float(rng.uniform(1.0 - border_scale_jitter, 1.0 + border_scale_jitter)),
        digit_scale=float(rng.uniform(1.0 - digit_scale_jitter, 1.0 + digit_scale_jitter)),
    )


def ensure_output_layout(output_dir: Path) -> tuple[Path, Path]:
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    min_rgba = parse_rgba_hex(args.plate_color_min)
    max_rgba = parse_rgba_hex(args.plate_color_max)
    images_dir, labels_dir = ensure_output_layout(args.output_dir)
    bag_name = bag_name_from_path(args.bag)
    topic_fragment = topic_name_fragment(args.topic)
    sample_prefix = sanitize_name_part(f"{args.filename_prefix}_{bag_name}_{topic_fragment}")

    generated_count = 0
    for frame in iter_ros_images_from_bag(
        bag_path=args.bag,
        topic_name=args.topic,
        storage_id=args.storage_id,
        sample_every_seconds=args.sample_every_seconds,
        max_samples=args.max_samples,
        flip_mode=args.flip,
    ):
        base_image = frame.image.copy()
        image_height, image_width = base_image.shape[:2]
        appearance = build_plate_appearance(
            rng,
            min_rgba,
            max_rgba,
            border_scale_jitter=args.border_scale_jitter,
            digit_scale_jitter=args.digit_scale_jitter,
        )
        projected_corners = sample_projected_plate(rng, image_width, image_height, args)

        plate_rgba = render_plate_texture(
            appearance=appearance,
            render_width=args.plate_render_width,
            plate_width_m=args.plate_width_m,
            plate_height_m=args.plate_height_m,
            border_thickness_m=args.border_thickness_m,
            digit_height_m=args.digit_height_m,
        )
        warped_rgba = warp_plate_to_image(plate_rgba, image_width, image_height, projected_corners)
        warped_rgba = apply_plate_appearance_jitter(
            rng=rng,
            warped_rgba=warped_rgba,
            blur_kernel=args.plate_blur_kernel,
            noise_std=args.plate_noise_std,
            brightness_jitter=args.plate_brightness_jitter,
        )
        composited = composite_rgba_over_bgr(base_image, warped_rgba)
        yolo_box = yolo_box_from_corners(projected_corners, image_width, image_height)

        stem = f"{sample_prefix}_{generated_count:06d}"
        image_path = images_dir / f"{stem}.{args.image_extension}"
        label_path = labels_dir / f"{stem}.txt"
        write_image(image_path, composited, args.image_extension, args.jpeg_quality)
        write_yolo_label(label_path, args.class_id, yolo_box)
        generated_count += 1

        print(
            f"[{generated_count}] saved {image_path.name} "
            f"(t={frame.timestamp_seconds:.3f}s, digits={appearance.upper_digits}/{appearance.lower_digits})"
        )

    if generated_count == 0:
        raise RuntimeError("No images were generated. Check the bag path, topic name, and sampling interval.")

    print(f"Generated {generated_count} synthetic samples in {args.output_dir}")


if __name__ == "__main__":
    main()
