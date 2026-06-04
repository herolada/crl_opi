#!/usr/bin/env python3
"""Train and export an object detector with Ultralytics YOLO."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET_YAML = PACKAGE_DIR / "data" / "data.yaml"
DEFAULT_RUNS_DIR = PACKAGE_DIR / "runs" / "training"


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a pretrained Ultralytics YOLO model on a custom dataset "
            "and export the best checkpoint to ONNX."
        )
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Pretrained Ultralytics checkpoint or model alias, e.g. yolov8s.pt or yolo11s.pt.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET_YAML,
        help="Path to the Ultralytics dataset YAML.",
    )
    parser.add_argument("--imgsz", type=int, default=416, help="Training and export image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--device", default="0", help='Training device, e.g. "0", "0,1", or "cpu".')
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Parent directory for training outputs.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional run name. Defaults to the model stem, e.g. yolov8s or yolo11s.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=4,
        help="How many validation images to visualize.",
    )
    parser.add_argument(
        "--preview-interval",
        type=int,
        default=5,
        help="Save validation prediction-vs-ground-truth previews every N epochs.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for preview images.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for preview images.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version for export.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export the ONNX model with dynamic axes.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the exported ONNX graph if the environment supports it.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Also log custom metrics and preview images to TensorBoard.",
    )
    return parser.parse_args()


def read_dataset_config(dataset_yaml: Path) -> dict:
    with dataset_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset config at {dataset_yaml} is not a YAML mapping.")
    return data


def resolve_split_dir(dataset_yaml: Path, data_config: dict, split: str) -> Path:
    dataset_root = Path(data_config.get("path", dataset_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (dataset_yaml.parent / dataset_root).resolve()

    split_value = data_config.get(split)
    if split_value is None:
        raise KeyError(f'Missing "{split}" entry in dataset config {dataset_yaml}.')

    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = (dataset_root / split_path).resolve()
    return split_path


def collect_samples(image_dir: Path, label_dir: Path, limit: int) -> list[Sample]:
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    samples: list[Sample] = []
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            samples.append(Sample(image_path=image_path, label_path=label_path))
        if len(samples) >= limit:
            break
    if not samples:
        raise FileNotFoundError(
            f"No matching validation image/label pairs found in {image_dir} and {label_dir}."
        )
    return samples


def load_labels(label_path: Path, image_shape: tuple[int, int, int]) -> list[tuple[int, int, int, int, int]]:
    height, width = image_shape[:2]
    labels: list[tuple[int, int, int, int, int]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, cx, cy, bw, bh = parts
            cx_f = float(cx) * width
            cy_f = float(cy) * height
            bw_f = float(bw) * width
            bh_f = float(bh) * height
            x1 = int(round(cx_f - bw_f / 2.0))
            y1 = int(round(cy_f - bh_f / 2.0))
            x2 = int(round(cx_f + bw_f / 2.0))
            y2 = int(round(cy_f + bh_f / 2.0))
            labels.append((int(class_id), x1, y1, x2, y2))
    return labels


def draw_ground_truth(
    image: np.ndarray,
    boxes: Iterable[tuple[int, int, int, int, int]],
    class_names: list[str] | None = None,
) -> np.ndarray:
    annotated = image.copy()
    for class_id, x1, y1, x2, y2 in boxes:
        label = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"gt:{label}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def draw_predictions(image: np.ndarray, result, class_names: list[str] | None = None) -> np.ndarray:
    annotated = image.copy()
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
        class_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"pred:{label} {conf:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def stack_preview(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    gt_panel = cv2.copyMakeBorder(ground_truth, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    pred_panel = cv2.copyMakeBorder(prediction, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(gt_panel, "Ground Truth", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(pred_panel, "Prediction", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return np.hstack((gt_panel, pred_panel))


class TrainingMonitor:
    """Ultralytics callback bundle for TensorBoard logging and preview exports."""

    def __init__(
        self, args: argparse.Namespace, val_samples: list[Sample], class_names: list[str] | None = None
    ) -> None:
        self.args = args
        self.val_samples = val_samples
        self.class_names = class_names
        self.writer: SummaryWriter | None = None
        self.run_dir: Path | None = None
        self.preview_dir: Path | None = None

    def on_train_start(self, trainer) -> None:
        self.run_dir = Path(trainer.save_dir)
        self.preview_dir = self.run_dir / "val_previews"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        if self.args.tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            self.writer.add_text("train/model", str(self.args.model))
            self.writer.add_text("train/data", str(self.args.data.resolve()))
            self.writer.add_text("train/hparams", json.dumps(training_overrides(self.args), indent=2))

    def on_fit_epoch_end(self, trainer) -> None:
        epoch = int(trainer.epoch) + 1
        if self.writer:
            metrics = getattr(trainer, "metrics", {}) or {}
            for key, value in metrics.items():
                if isinstance(value, (float, int)):
                    self.writer.add_scalar(key, value, epoch)
        self._log_per_class_metrics(trainer, epoch)

    def _log_per_class_metrics(self, trainer, epoch: int) -> None:
        validator = getattr(trainer, "validator", None)
        if validator is None:
            return
        val_metrics = getattr(validator, "metrics", None)
        if val_metrics is None:
            return

        # Ultralytics stores per-class data under metrics.box (DetMetrics) or directly on metrics
        box = getattr(val_metrics, "box", val_metrics)
        ap50 = getattr(box, "ap50", None)
        maps = getattr(box, "maps", None)
        ap_class_index = getattr(val_metrics, "ap_class_index", getattr(box, "ap_class_index", None))

        if ap50 is None or ap_class_index is None or len(ap50) == 0:
            return

        parts = []
        for i, cls_idx in enumerate(ap_class_index):
            if i >= len(ap50):
                break
            name = self.class_names[cls_idx] if self.class_names and cls_idx < len(self.class_names) else str(cls_idx)
            ap50_val = float(ap50[i])
            maps_val = float(maps[i]) if maps is not None and i < len(maps) else None
            parts.append(
                f"{name}: mAP50={ap50_val:.3f}"
                + (f" mAP50-95={maps_val:.3f}" if maps_val is not None else "")
            )
            if self.writer:
                self.writer.add_scalar(f"per_class/mAP50_{name}", ap50_val, epoch)
                if maps_val is not None:
                    self.writer.add_scalar(f"per_class/mAP50-95_{name}", maps_val, epoch)

        if parts:
            print(f"[Epoch {epoch:03d}] Per-class — {' | '.join(parts)}")

    def on_model_save(self, trainer) -> None:
        epoch = int(trainer.epoch) + 1
        if epoch % self.args.preview_interval != 0 and epoch != int(self.args.epochs):
            return
        self._render_previews(trainer, epoch)

    def on_train_end(self, trainer) -> None:
        self._render_previews(trainer, int(trainer.epoch) + 1, force_best=True)
        if self.writer:
            self.writer.flush()
            self.writer.close()

    def _render_previews(self, trainer, epoch: int, force_best: bool = False) -> None:
        if self.preview_dir is None:
            return

        checkpoint_path = Path(trainer.best if force_best and Path(trainer.best).exists() else trainer.last)
        predictor = YOLO(str(checkpoint_path))
        epoch_dir = self.preview_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        for index, sample in enumerate(self.val_samples):
            image_bgr = cv2.imread(str(sample.image_path))
            if image_bgr is None:
                continue

            gt_boxes = load_labels(sample.label_path, image_bgr.shape)
            prediction = predictor.predict(
                source=str(sample.image_path),
                imgsz=self.args.imgsz,
                conf=self.args.conf,
                iou=self.args.iou,
                verbose=False,
                device=self.args.device,
            )[0]

            preview = stack_preview(
                draw_ground_truth(image_bgr, gt_boxes, self.class_names),
                draw_predictions(image_bgr, prediction, self.class_names),
            )
            preview_path = epoch_dir / f"{sample.image_path.stem}_preview.jpg"
            cv2.imwrite(str(preview_path), preview)

            if self.writer:
                rgb_preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                chw_preview = np.transpose(rgb_preview, (2, 0, 1))
                self.writer.add_image(
                    tag=f"val_previews/{sample.image_path.stem}",
                    img_tensor=chw_preview,
                    global_step=epoch,
                )


def training_overrides(args: argparse.Namespace) -> dict:
    run_name = args.name or Path(args.model).stem
    return {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "project": str(args.project.resolve()),
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "cos_lr": True,        # cosine LR schedule
        "weight_decay": 0.001,
        "hsv_h": 0.05,  # was 0.7 — camo class relies on hue; aggressive shifts destroy that signal
        "hsv_s": 0.5,
        "hsv_v": 0.4,
        "degrees": 45.0,
        "translate": 0.1,
        "scale": 1.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 0.5,
        "mixup": 0.1,
        "copy_paste": 0.3,
        "copy_paste_mode": "flip",
        "auto_augment": "randaugment",
        "erasing": 0.4,
        "crop_fraction": 1.0,
        "plots": True,
        "save": True,
        "save_period": args.preview_interval,
        "val": True,
        "verbose": True,
    }

def main() -> None:
    args = parse_args()
    args.data = args.data.resolve()
    args.project = args.project.resolve()

    data_config = read_dataset_config(args.data)
    class_names: list[str] | None = data_config.get("names")
    val_image_dir = resolve_split_dir(args.data, data_config, "val")
    val_label_dir = val_image_dir.parent / "labels" #/ val_image_dir.name
    val_samples = collect_samples(val_image_dir, val_label_dir, args.preview_count)

    model = YOLO(args.model)
    monitor = TrainingMonitor(args, val_samples, class_names)
    model.add_callback("on_train_start", monitor.on_train_start)
    model.add_callback("on_fit_epoch_end", monitor.on_fit_epoch_end)
    model.add_callback("on_model_save", monitor.on_model_save)
    model.add_callback("on_train_end", monitor.on_train_end)

    results = model.train(**training_overrides(args))

    best_checkpoint = Path(results.save_dir) / "weights" / "best.pt"
    export_model = YOLO(str(best_checkpoint))
    export_path = export_model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
    )

    print(f"Training complete. Best checkpoint: {best_checkpoint}")
    print(f"ONNX export complete: {export_path}")
    if args.tensorboard:
        print(f"TensorBoard logs: {Path(results.save_dir) / 'tensorboard'}")
    print(f"Validation previews: {Path(results.save_dir) / 'val_previews'}")


if __name__ == "__main__":
    main()
