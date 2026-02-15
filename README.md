# APSC103 892C
---
# Action Detection Prototype

An action detection prototype project based on **YOLO26n-pose** + **MMAction2 STGCN++**.

**Documentation:** [English Docs](DOC_EN.md) | [中文](DOC_CH.md)

## Architecture

```
Video/Camera -> YOLO26n-pose (Pose Detection) -> Intermediate Layer -> STGCN++ (Action Recognition) -> Result
```

- **YOLO26n-pose**: Detects 17 COCO keypoints for all persons in every frame.
- **Intermediate Layer**: Accumulates multi-frame skeleton data using a sliding window and converts it to MMAction2 format.
- **STGCN++**: Performs action recognition based on skeleton sequences (NTU60 - 60 action classes).

## Requirements

- Python 3.10
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Patch Known mmaction2 Issues

```bash
uv run python patch_mmaction.py
```

### 3. Download Model Weights

```bash
uv run python -m src.download_models
```

### 4. Run

Process a video file:

```bash
uv run python -m src.main --video path/to/video.mp4
```

Real-time detection using a camera:

```bash
uv run python -m src.main --camera 0
```

### 5. Verify Installation

```bash
uv run python verify_pipeline.py
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | - | Path to the video file |
| `--camera` | - | Camera device ID |
| `--device` | cpu | Inference device (cpu / cuda:0) |
| `--window-size` | 48 | Sliding window size (number of frames) |
| `--stride` | 16 | Window sliding stride |
| `--pose-conf` | 0.5 | Pose detection confidence threshold |
| `--max-persons` | 2 | Maximum number of persons to track |
| `--top-k` | 5 | Return Top-K action recognition results |
| `--no-show` | false | Do not display the visualization window |

## Dependency Versions

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.1.0 (CPU) |
| ultralytics | 8.4.12 |
| mmcv-lite | 2.1.0 |
| mmengine | 0.10.7 |
| mmaction2 | 1.2.0 |
| numpy | <2.0 |

## Project Structure

```
action-detection/
  pyproject.toml           # Project configuration and dependencies
  configs/
    stgcnpp_config.py      # STGCN++ inference config
  models/                  # Model weights directory
  src/
    pose_estimator.py      # YOLO26n-pose wrapper
    skeleton_converter.py  # Skeleton data converter (sliding window)
    action_recognizer.py   # STGCN++ action recognition wrapper
    pipeline.py            # Integrated pipeline
    main.py                # CLI entry point
    download_models.py     # Model download tool
  verify_pipeline.py       # End-to-end verification script
  patch_mmaction.py        # mmaction2 import patch
```

## GPU Support

To use a GPU, modify the PyTorch index source in `pyproject.toml` to the CUDA version:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cu118"  # CUDA 11.8
explicit = true
```

Then reinstall dependencies:

```bash
uv sync
```

Specify the device when running:

```bash
uv run python -m src.main --video input.mp4 --device cuda:0
```
