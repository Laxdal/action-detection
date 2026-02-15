# Action Detection Project Documentation

> A real-time action detection prototype system based on YOLO26n-pose + MMAction2 STGCN++.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Project Directory Structure](#project-directory-structure)
- [Script Execution Guide](#script-execution-guide)
  - [Step 1: Install Dependencies](#step-1-install-dependencies)
  - [Step 2: Patch mmaction2](#step-2-patch-mmaction2)
  - [Step 3: Download Model Weights](#step-3-download-model-weights)
  - [Step 4: Verify Installation](#step-4-verify-installation)
  - [Step 5: Run Action Detection](#step-5-run-action-detection)
  - [Running Tests](#running-tests)
- [Detailed File Descriptions](#detailed-file-descriptions)
  - [Root Directory Files](#root-directory-files)
  - [src/ Source Modules](#src-source-modules)
  - [configs/ Configuration Files](#configs-configuration-files)
  - [models/ Model Weights](#models-model-weights)
  - [tests/ Test Modules](#tests-test-modules)
- [Data Flow & Pipeline Principle](#data-flow--pipeline-principle)
- [CLI Arguments Reference](#cli-arguments-reference)
- [Supported Action Classes](#supported-action-classes)
- [GPU Support](#gpu-support)

---

## Project Overview

This project is an **end-to-end real-time action detection prototype** capable of detecting human poses and recognizing actions from video files or camera feeds. The system consists of two core models:

1.  **YOLO26n-pose** — A lightweight pose estimation model based on ultralytics, which detects 17 COCO keypoints (nose, eyes, shoulders, elbows, wrists, hips, knees, ankles, etc.) for all persons in the frame.
2.  **STGCN++** — A Spatio-Temporal Graph Convolutional Network based on MMAction2, which takes multi-frame skeleton sequences as input to recognize 60 types of human actions (from the NTU RGB+D 60 dataset).

These two are connected via a **Sliding Window Conversion Layer**, responsible for accumulating frame-by-frame YOLO detection results and converting them into the skeleton sequence format required by STGCN++.

---

## System Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│ Video/Camera  │ ──> │  YOLO26n-pose    │ ──> │  Sliding Window   │ ──> │  STGCN++     │
│ (Input Source)│     │ (Pose Detection) │     │ (Data Converter)  │     │ (Action Rec.)│
└──────────────┘     └──────────────────┘     └──────────────────┘     └──────────────┘
                      ↓                        ↓                        ↓
                  17 COCO Keypoints        Accumulate N frames      Output Top-K Action
                  per frame                Convert to MMAction2     Classes & Confidence
```

**Process Flow:**

1.  Read images frame-by-frame from the video or camera.
2.  YOLO26n-pose detects coordinates and confidence scores for 17 keypoints for all persons in each frame.
3.  The sliding window converter accumulates frame data; inference is triggered when the window is full and the sliding stride is reached.
4.  STGCN++ receives the skeleton sequence and outputs the probability distribution for 60 action classes.
5.  Returns the Top-K most likely action labels and their confidence scores.

---

## Project Directory Structure

```
action-detection/
│
├── pyproject.toml              # Project configuration: dependencies, build config, pytest config
├── uv.lock                     # Dependency lock file for uv package manager
├── .python-version             # Specifies Python version (3.10)
├── .gitignore                  # Git ignore rules
├── README.md                   # Project introduction and quick start guide
├── DOCS.md                     # This document: Detailed project explanation
│
├── patch_mmaction.py           # [Tool Script] Patches known import bugs in mmaction2
├── verify_pipeline.py          # [Tool Script] End-to-end pipeline verification (using synthetic data)
├── video.mp4                   # Test video file
├── yolo26n-pose.pt             # YOLO26n-pose pre-trained model weights
│
├── configs/
│   └── stgcnpp_config.py       # STGCN++ inference configuration (model structure + test pipeline)
│
├── models/
│   └── stgcnpp_ntu60_xsub_2d.pth  # STGCN++ pre-trained weights (NTU60 XSub 2D Joint)
│
├── src/                        # Core source code package
│   ├── __init__.py             # Package initialization file (empty)
│   ├── __main__.py             # Package entry: supports running via 'python -m src'
│   ├── main.py                 # CLI entry: argument parsing + pipeline startup
│   ├── pipeline.py             # Integrated pipeline: links Pose -> Converter -> Action Recognition
│   ├── pose_estimator.py       # YOLO26n-pose wrapper
│   ├── skeleton_converter.py   # Sliding window skeleton data converter
│   ├── action_recognizer.py    # STGCN++ action recognition wrapper
│   └── download_models.py      # Model download tool script
│
└── tests/                      # Test modules
    ├── __init__.py             # Package initialization file (empty)
    ├── conftest.py             # Shared pytest fixtures
    ├── test_pose_estimator.py  # Unit tests for PoseEstimator
    ├── test_skeleton_converter.py  # Unit tests for SkeletonConverter
    ├── test_action_recognizer.py   # Unit tests for ActionRecognizer
    └── test_pipeline_integration.py  # End-to-end integration tests
```

---

## Script Execution Guide

Follow these **complete steps** to set up the environment and run action detection from scratch.

### Step 1: Install Dependencies

**Prerequisites:** Python 3.10 and the [uv](https://docs.astral.sh/uv/) package manager must be installed.

```bash
uv sync
```

This command installs all project dependencies (PyTorch, ultralytics, mmaction2, etc.) based on `pyproject.toml` and `uv.lock`, and creates a virtual environment.

### Step 2: Patch mmaction2

```bash
uv run python patch_mmaction.py
```

The `mmaction2` 1.2.0 pip package has a known bug—missing the `drn` module directory causes import errors. This script automatically wraps the problematic `from .drn.drn import DRN` import in a `try-except` block so it doesn't block execution.

**Note:** Run this only once. The script is idempotent and will skip patching if already applied.

### Step 3: Download Model Weights

```bash
uv run python -m src.download_models
```

This script downloads two pre-trained models:
-   **STGCN++ Weights** (`models/stgcnpp_ntu60_xsub_2d.pth`) — Downloaded from the official OpenMMLab source.
-   **YOLO26n-pose Weights** (`yolo26n-pose.pt`) — Automatically downloaded via the ultralytics library.

**Note:** If the model files already exist, the download is skipped. If you already have these files in the project, you can skip this step.

### Step 4: Verify Installation

```bash
uv run python verify_pipeline.py
```

Tests the four modules sequentially using synthetic data (randomly generated images):
1.  YOLO26n-pose Pose Detection
2.  Skeleton Data Format Conversion
3.  STGCN++ Action Recognition
4.  Pipeline Integration

**No real video is needed** to verify that the environment configuration is correct.

### Step 5: Run Action Detection

**Process a video file:**

```bash
uv run python -m src.main --video video.mp4
```

**Real-time detection using a camera:**

```bash
uv run python -m src.main --camera 0
```

**Run with custom arguments:**

```bash
uv run python -m src.main --video input.mp4 --device cuda:0 --window-size 64 --stride 8 --top-k 3
```

A visualization window will open during runtime (disable with `--no-show`), displaying the live feed and Top-K action recognition results. Press `q` to exit.

### Running Tests

```bash
uv run pytest
```

Runs all unit and integration tests under the `tests/` directory. Requires model files (`yolo26n-pose.pt` and `models/stgcnpp_ntu60_xsub_2d.pth`) to be present; otherwise, related tests will be skipped.

---

## Detailed File Descriptions

### Root Directory Files

#### `pyproject.toml` — Project Configuration

Defines project metadata, dependencies, and tool configurations:
-   **Core Dependencies**: `torch 2.1.0`, `torchvision 0.16.0` (CPU default), `ultralytics >=8.4.12`, `mmaction2 1.2.0`, `mmcv-lite`, `mmengine`, `numpy <2.0`.
-   **Dev Dependencies**: `pytest >=9.0.2`.
-   **PyTorch Source**: Configured for CPU wheel index; can be changed to CUDA version.
-   **pytest Config**: Test path set to `tests/`, ignores mmcv UserWarnings.

#### `patch_mmaction.py` — mmaction2 Import Patch Tool

| Attribute | Value |
|---|---|
| Type | Standalone tool script (run directly) |
| Usage | `uv run python patch_mmaction.py` |
| Timing | Run once after installing dependencies |
| Purpose | Fixes import error due to missing `drn` module in mmaction2 1.2.0 pip package |

**Logic:** Locates `models/localizers/__init__.py` in the mmaction installation path and wraps `from .drn.drn import DRN` in a `try-except` block.

#### `verify_pipeline.py` — End-to-End Verification Script

| Attribute | Value |
|---|---|
| Type | Standalone tool script (run directly) |
| Usage | `uv run python verify_pipeline.py` |
| Timing | For verification after environment setup |
| Purpose | Gradually tests 4 components using synthetic data |

**Verification Steps:**
1.  Generate 480×640 random image → YOLO26n-pose detection.
2.  Simulate 8 frames of data → SkeletonConverter sliding window.
3.  Converted data → STGCN++ action recognition inference.
4.  Comprehensive integration status check.

#### `video.mp4` — Test Video

A video file for testing and demonstration, used as input for the `--video` argument.

#### `yolo26n-pose.pt` — YOLO Model Weights

YOLO26n-pose pre-trained model provided by ultralytics, detecting 17 COCO human keypoints.

---

### src/ Source Modules

#### `src/__main__.py` — Module Entry

Allows running the project via `python -m src`, internally calling the `main()` function in `main.py`.

#### `src/main.py` — CLI Entry Point

| Attribute | Value |
|---|---|
| Type | Main program entry (run directly) |
| Usage | `uv run python -m src.main --video <path>` or `--camera <ID>` |
| Purpose | Parses CLI arguments, initializes pipeline, starts video/camera processing |

**Responsibilities:**
-   Parses input source (`--video` or `--camera`, mutually exclusive, one is required).
-   Parses model paths, inference device, window parameters, etc.
-   Checks if model files exist.
-   Creates an `ActionDetectionPipeline` instance and starts processing.
-   Defines a `print_prediction()` callback to print results to the console.

#### `src/pipeline.py` — Integrated Pipeline

| Attribute | Value |
|---|---|
| Type | Core module (imported by main.py) |
| Core Class | `ActionDetectionPipeline` |
| Purpose | Connects YOLO → Converter → STGCN++ components, provides video/camera interfaces |

**`ActionDetectionPipeline` Class:**
-   `__init__()`: Initializes three sub-components: `PoseEstimator`, `ActionRecognizer`, `SkeletonConverter`.
-   `process_video(video_path, callback, show)`: Processes video files frame-by-frame (detect -> infer), optionally showing a window with annotations.
-   `process_camera(camera_id, callback)`: Processes real-time camera input, always displays visualization window.
-   `_process_frame(frame)`: Single frame processing: Pose Detection → Add to Window → Trigger Inference (if condition met).
-   `_draw_annotations(frame, predictions)`: Draws semi-transparent background + action results text + buffer progress on the frame.
-   `reset()`: Resets sliding window state.

#### `src/pose_estimator.py` — Pose Detection Module

| Attribute | Value |
|---|---|
| Type | Core module (imported by pipeline.py) |
| Core Class | `PoseEstimator`, `PoseResult` |
| Dependency | ultralytics YOLO |
| Purpose | Wraps YOLO26n-pose to detect skeleton keypoints for all persons per frame |

**`PoseResult` Dataclass:**
-   `keypoints`: Keypoint coordinates (N, 17, 2), N = number of persons.
-   `keypoint_scores`: Keypoint confidence scores (N, 17).
-   `boxes`: Bounding boxes (N, 4), xyxy format.
-   `box_scores`: Box confidence scores (N,).
-   `img_shape`: Image dimensions (H, W).

**`PoseEstimator` Class:**
-   `predict(frame)`: Single frame detection, returns `PoseResult`.
-   `predict_batch(frames)`: Batch detection (calls single frame internally).

#### `src/skeleton_converter.py` — Skeleton Data Converter Module

| Attribute | Value |
|---|---|
| Type | Core module (imported by pipeline.py) |
| Core Class | `SkeletonConverter` |
| Purpose | Maintains sliding window, converts YOLO output to MMAction2 `pose_results` format |

**`SkeletonConverter` Class:**
-   `add_frame(pose_result)`: Adds a frame to buffer, returns whether inference should be triggered.
-   `get_pose_results()`: Gets converted data for current window, returns `(pose_results, img_shape)`.
-   `reset()`: Clears buffer and frame counter.
-   `is_ready` Property: Whether the window is full.
-   `buffer_size` Property: Current number of frames in buffer.

**Key Logic:**
-   Uses `deque(maxlen=window_size)` to maintain a fixed-size sliding window.
-   Triggers inference when buffer is full AND (frames since last output) ≥ stride.
-   Sorts by bounding box confidence, keeping max `max_persons` per frame.
-   Fills with zeros if no person is detected to ensure consistent data format.

#### `src/action_recognizer.py` — Action Recognition Module

| Attribute | Value |
|---|---|
| Type | Core module (imported by pipeline.py) |
| Core Class | `ActionRecognizer` |
| Dependency | mmaction2 |
| Purpose | Wraps STGCN++ to classify actions based on skeleton sequences |

**`ActionRecognizer` Class:**
-   `recognize(pose_results, img_shape)`: Infers on a skeleton sequence, returns all 60 action classes and scores (descending order).
-   `recognize_top_k(pose_results, img_shape, top_k)`: Returns the top K results.

**Built-in Constant `NTU60_LABELS`**: List of 60 NTU RGB+D 60 action class labels (English).

#### `src/download_models.py` — Model Download Tool

| Attribute | Value |
|---|---|
| Type | Tool script (run directly) |
| Usage | `uv run python -m src.download_models` |
| Purpose | Downloads STGCN++ and YOLO26n-pose pre-trained weights |

**Functions:**
-   `download_stgcnpp()`: Downloads STGCN++ weights from OpenMMLab to `models/` directory, showing a progress bar.
-   `download_yolo_pose()`: Triggers automatic download by initializing the ultralytics YOLO model.
-   Automatically skips if files exist.

---

### configs/ Configuration Files

#### `configs/stgcnpp_config.py` — STGCN++ Inference Config

Defines model structure and inference pipeline:
-   **Model Type**: `RecognizerGCN`
-   **Backbone**: `STGCN` (Adaptive Graph Conv + Residuals + Multi-Scale Temporal Conv MSTCN)
-   **Graph**: COCO layout, spatial strategy.
-   **Head**: `GCNHead`, 60 classes, 256 input channels.
-   **Test Pipeline**:
    1.  `PreNormalize2D`: 2D coordinate pre-normalization.
    2.  `GenSkeFeat`: Generate skeleton features (joint features).
    3.  `UniformSampleFrames`: Uniformly sample 100 frames.
    4.  `PoseDecode`: Pose decoding.
    5.  `FormatGCNInput`: Format GCN input (2 persons).
    6.  `PackActionInputs`: Pack action inputs.

---

### models/ Model Weights

#### `models/stgcnpp_ntu60_xsub_2d.pth`

STGCN++ pre-trained weights trained on NTU RGB+D 60 dataset (XSub split, 2D Joint), supporting 60 action classes. Sourced from OpenMMLab official model zoo.

---

### tests/ Test Modules

#### `tests/conftest.py` — Shared Fixtures

Provides shared pytest fixtures for all tests:
-   `project_root`: Project root directory path.
-   `fake_frame`: Synthetic 480×640 image (with simple human contours to improve detection).
-   `real_video_path`: Path to test video (skipped if missing).
-   `real_video_first_frame`: First frame of the real video.
-   `stgcnpp_config_path`: Path to STGCN++ config.
-   `stgcnpp_checkpoint_path`: Path to STGCN++ weights (skipped if missing).
-   `yolo_model_path`: Path to YOLO model (skipped if missing).

#### `tests/test_pose_estimator.py`

Tests `PoseEstimator` and `PoseResult`:
-   Verifies `PoseResult` data structure.
-   Verifies YOLO model loading and pose detection on images.
-   Checks output keypoint shapes, confidence scores, etc.

#### `tests/test_skeleton_converter.py`

Tests `SkeletonConverter`:
-   Verifies sliding window buffer behavior.
-   Verifies inference trigger conditions (window full + stride).
-   Verifies data format conversion correctness.
-   Verifies person truncation logic.
-   Verifies reset functionality.

#### `tests/test_action_recognizer.py`

Tests `ActionRecognizer`:
-   Verifies integrity of NTU60 label list (60 classes).
-   Verifies STGCN++ model loading and inference.
-   Verifies Top-K result format and sorting.

#### `tests/test_pipeline_integration.py`

End-to-end integration tests:
-   Tests complete pipeline using real models.
-   Tests video processing functionality.
-   Tests frame processing and annotation drawing.
-   Tests reset functionality.

---

## Data Flow & Pipeline Principle

```
Input Frame (BGR, H×W×3)
    │
    ▼
┌─────────────────────────────────────────┐
│  PoseEstimator.predict(frame)           │
│  ├─ YOLO26n-pose Model Inference        │
│  └─ Returns PoseResult:                 │
│       keypoints:      (N, 17, 2)        │  N = detected persons
│       keypoint_scores: (N, 17)          │  17 = COCO keypoints
│       boxes:           (N, 4)           │
│       box_scores:      (N,)             │
│       img_shape:       (H, W)           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  SkeletonConverter.add_frame(result)    │
│  ├─ Add to deque buffer                 │
│  ├─ Check if window full (≥ window_size)│
│  └─ Check if stride reached (≥ stride)  │
│     └─ Returns True → Trigger Inference │
└─────────────────────────────────────────┘
    │ (When True is returned)
    ▼
┌─────────────────────────────────────────┐
│  SkeletonConverter.get_pose_results()   │
│  ├─ Iterate frames in window            │
│  ├─ Sort by box_scores, keep max_persons│
│  ├─ Fill zero values for empty frames   │
│  └─ Returns (pose_results, img_shape):  │
│       pose_results: List[dict]          │
│         Each dict contains:             │
│           keypoints:       (M, 17, 2)   │  M ≤ max_persons
│           keypoint_scores: (M, 17)      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  ActionRecognizer.recognize_top_k(...)  │
│  ├─ mmaction2 inference_skeleton        │
│  ├─ Get probability distribution (60)   │
│  └─ Returns Top-K (label, score) list   │
└─────────────────────────────────────────┘
    │
    ▼
Output: [("drink water", 0.85), ("eat meal", 0.05), ...]
```

---

## CLI Arguments Reference

The following arguments are supported when running via `uv run python -m src.main`:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--video` | str | — | Video file path (mutually exclusive with `--camera`, one required) |
| `--camera` | int | — | Camera device ID (mutually exclusive with `--video`, one required) |
| `--pose-model` | str | `yolo26n-pose.pt` | YOLO pose model weights path |
| `--action-config` | str | `configs/stgcnpp_config.py` | STGCN++ configuration file path |
| `--action-checkpoint` | str | `models/stgcnpp_ntu60_xsub_2d.pth` | STGCN++ model weights path |
| `--device` | str | `cpu` | Inference device (`cpu` or `cuda:0`) |
| `--window-size` | int | `48` | Sliding window size (frames); larger is more stable but higher latency |
| `--stride` | int | `16` | Window sliding stride; smaller means higher inference frequency |
| `--pose-conf` | float | `0.5` | Pose detection confidence threshold; lower values are filtered |
| `--max-persons` | int | `2` | Max persons to track per frame |
| `--top-k` | int | `5` | Number of Top-K action results to return |
| `--no-show` | flag | `false` | Do not show visualization window (video mode only) |

---

## Supported Action Classes

The system supports recognizing 60 human actions defined by the NTU RGB+D 60 dataset:

| No. | Action | No. | Action |
|---|---|---|---|
| 1 | drink water | 31 | pointing to something |
| 2 | eat meal/snack | 32 | taking a selfie |
| 3 | brushing teeth | 33 | check time (from watch) |
| 4 | brushing hair | 34 | rub two hands together |
| 5 | drop | 35 | nod head/bow |
| 6 | pickup | 36 | shake head |
| 7 | throw | 37 | wipe face |
| 8 | sitting down | 38 | salute |
| 9 | standing up (from sitting) | 39 | put the palms together |
| 10 | clapping | 40 | cross hands in front |
| 11 | reading | 41 | sneeze/cough |
| 12 | writing | 42 | staggering |
| 13 | tear up paper | 43 | falling |
| 14 | wear jacket | 44 | touch head (headache) |
| 15 | take off jacket | 45 | touch chest (heart pain) |
| 16 | wear a shoe | 46 | touch back (backache) |
| 17 | take off a shoe | 47 | touch neck (neckache) |
| 18 | wear on glasses | 48 | nausea or vomiting |
| 19 | take off glasses | 49 | use a fan / feeling warm |
| 20 | put on a hat/cap | 50 | punching/slapping other person |
| 21 | take off a hat/cap | 51 | kicking other person |
| 22 | cheer up | 52 | pushing other person |
| 23 | hand waving | 53 | pat on back of other person |
| 24 | kicking something | 54 | point finger at other person |
| 25 | reach into pocket | 55 | hugging other person |
| 26 | hopping (one foot jumping) | 56 | giving something to other person |
| 27 | jump up | 57 | touch other person's pocket |
| 28 | make a phone call | 58 | handshaking |
| 29 | playing with phone/tablet | 59 | walking towards each other |
| 30 | typing on a keyboard | 60 | walking apart from each other |

---

## GPU Support

The default configuration uses CPU inference. To enable GPU acceleration, follow these steps:

**1. Modify PyTorch index source in `pyproject.toml`:**

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cu118"  # CUDA 11.8
explicit = true
```

Common CUDA version URLs:
- CUDA 11.8: `https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `https://download.pytorch.org/whl/cu121`

**2. Reinstall dependencies:**

```bash
uv sync
```

**3. Specify GPU device when running:**

```bash
uv run python -m src.main --video input.mp4 --device cuda:0
```