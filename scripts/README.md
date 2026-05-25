# ADR Hazard Panel Detector Training

This folder contains a small Ultralytics training pipeline for fine-tuning an ADR hazard identification panel detector and exporting the best checkpoint to ONNX for later C++/ROS 2 inference.

## Possible improvements to try
Anything that will make the model faster, more accurate, or overcome the difference between the distribution of the training data and what the real robot sees.

Some ideas from the top of my head (feel free to try anything else though):
- Data
    - find more public datasets
    - make a custom dataset from public image
    - make a synthetic dataset from robot bag files & synthetic ADR panel injection
    - better data augmentation
- Model
    - different model (different YOLO, or another model e.g. FAST R-CNN)
    - different size of model (currently using 's' models)
    - better hyper-parameters (lr, bs, loss weights, weight decay, schedule, etc.)


## Install

I used `uv` for environment managment.

1. install uv (https://docs.astral.sh/uv/getting-started/installation/)
2. (inside scripts/) `uv sync`

## Train

From the workspace root:

```bash
uv run python crl_opi/scripts/train_adr_panel_detector.py --tensorboard
```

Or change model (should get automatically downloaded, if provided by the latest `ultralytics`):

```bash
uv run python crl_opi/scripts/train_adr_panel_detector.py --model yolo11s.pt --tensorboard
```

## Monitor Training

Launch TensorBoard against the run directory:

```bash
uv run tensorboard --logdir crl_opi/runs/adr_panel_training
```

The training run also saves validation comparison images under:

```text
crl_opi/runs/adr_panel_training/<run_name>/val_previews/
```

Each preview shows ground-truth boxes side by side with model predictions so you can track qualitative progress during fine-tuning.

## Data Generator
Generate synthethic data.
```
uv run data_generator.py --bag path_to_bag.mcap --topic /camera/topic/can_be_even_compressed --output-dir ../data/synth --sample-every-seconds 20 --flip {none/horizontal}
```