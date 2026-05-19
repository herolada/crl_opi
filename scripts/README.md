# ADR Hazard Panel Detector Training

This folder contains a small Ultralytics training pipeline for fine-tuning an ADR hazard identification panel detector and exporting the best checkpoint to ONNX for later C++/ROS 2 inference.

## Install

Use a Python environment with PyTorch support for your machine, then install:

```bash
python3 -m pip install ultralytics tensorboard onnx onnxruntime
```

## Train

From the workspace root:

```bash
python3 crl_opi/scripts/train_adr_panel_detector.py --tensorboard
```

Key options:

```bash
# Swap model families without editing code
python3 crl_opi/scripts/train_adr_panel_detector.py --model yolo11s.pt --tensorboard

# CPU-only training
python3 crl_opi/scripts/train_adr_panel_detector.py --device cpu --tensorboard

# Dynamic-shape ONNX export
python3 crl_opi/scripts/train_adr_panel_detector.py --dynamic --simplify --tensorboard
```

## Monitor Training

Launch TensorBoard against the run directory:

```bash
tensorboard --logdir crl_opi/runs/adr_panel_training
```

The training run also saves validation comparison images under:

```text
crl_opi/runs/adr_panel_training/<run_name>/val_previews/
```

Each preview shows ground-truth boxes side by side with model predictions so you can track qualitative progress during fine-tuning.
