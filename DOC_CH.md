# Action Detection 项目文档

> 基于 YOLO26n-pose + MMAction2 STGCN++ 的实时动作检测原型系统

---

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [项目目录结构](#项目目录结构)
- [脚本运行指南](#脚本运行指南)
  - [第一步：安装依赖](#第一步安装依赖)
  - [第二步：修补 mmaction2](#第二步修补-mmaction2)
  - [第三步：下载模型权重](#第三步下载模型权重)
  - [第四步：验证安装](#第四步验证安装)
  - [第五步：运行动作检测](#第五步运行动作检测)
  - [运行测试](#运行测试)
- [各文件详细说明](#各文件详细说明)
  - [根目录文件](#根目录文件)
  - [src/ 源码模块](#src-源码模块)
  - [configs/ 配置文件](#configs-配置文件)
  - [models/ 模型权重](#models-模型权重)
  - [tests/ 测试模块](#tests-测试模块)
- [数据流与流水线原理](#数据流与流水线原理)
- [命令行参数参考](#命令行参数参考)
- [支持的动作类别](#支持的动作类别)
- [GPU 支持](#gpu-支持)

---

## 项目概述

本项目是一个**端到端实时动作检测原型**，能够从视频文件或摄像头中检测人体姿态并识别正在执行的动作。系统由两个核心模型组成：

1. **YOLO26n-pose** — 基于 ultralytics 的轻量级姿态估计模型，逐帧检测画面中所有人的 17 个 COCO 关键点（鼻子、眼睛、肩膀、手肘、手腕、髋部、膝盖、脚踝等）
2. **STGCN++** — 基于 MMAction2 的时空图卷积网络，将多帧骨架序列作为输入，识别出 60 种人体动作（来自 NTU RGB+D 60 数据集）

两者之间通过一个**滑动窗口转换层**连接，负责将逐帧的 YOLO 检测结果积累、转换为 STGCN++ 所需的骨架序列格式。

---

## 系统架构

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│ 视频/摄像头   │ ──> │  YOLO26n-pose    │ ──> │  滑动窗口转换层   │ ──> │  STGCN++     │
│ (输入源)      │     │  (姿态检测)       │     │  (数据格式转换)   │     │  (动作识别)   │
└──────────────┘     └──────────────────┘     └──────────────────┘     └──────────────┘
                      ↓                        ↓                        ↓
                  每帧 17 个               积累 N 帧骨架数据         输出 Top-K 动作
                  COCO 关键点              转为 MMAction2 格式       类别及置信度
```

**处理流程：**

1. 从视频/摄像头逐帧读取图像
2. YOLO26n-pose 对每帧检测所有人的 17 个关键点坐标和置信度
3. 滑动窗口转换器将帧数据累积，当窗口填满且到达滑动步长时触发推理
4. STGCN++ 接收骨架序列，输出 60 类动作的概率分布
5. 返回 Top-K 个最可能的动作标签和置信度

---

## 项目目录结构

```
action-detection/
│
├── pyproject.toml              # 项目配置：依赖声明、构建配置、pytest 配置
├── uv.lock                     # uv 包管理器的依赖锁定文件
├── .python-version             # 指定 Python 版本 (3.10)
├── .gitignore                  # Git 忽略规则
├── README.md                   # 项目简介和快速入门指南
├── DOCS.md                     # 本文档：详细项目说明
│
├── patch_mmaction.py           # [工具脚本] 修补 mmaction2 已知导入 bug
├── verify_pipeline.py          # [工具脚本] 端到端流水线验证（使用合成数据）
├── video.mp4                   # 测试视频文件
├── yolo26n-pose.pt             # YOLO26n-pose 预训练模型权重
│
├── configs/
│   └── stgcnpp_config.py       # STGCN++ 推理配置文件（模型结构 + 测试流水线）
│
├── models/
│   └── stgcnpp_ntu60_xsub_2d.pth  # STGCN++ 预训练模型权重（NTU60 XSub 2D Joint）
│
├── src/                        # 核心源码包
│   ├── __init__.py             # 包初始化文件（空）
│   ├── __main__.py             # 包入口：支持 python -m src 运行
│   ├── main.py                 # CLI 入口：命令行参数解析 + 启动流水线
│   ├── pipeline.py             # 整合流水线：串联姿态检测 → 骨架转换 → 动作识别
│   ├── pose_estimator.py       # YOLO26n-pose 姿态检测封装
│   ├── skeleton_converter.py   # 滑动窗口骨架数据转换器
│   ├── action_recognizer.py    # STGCN++ 动作识别封装
│   └── download_models.py      # 模型下载工具脚本
│
└── tests/                      # 测试模块
    ├── __init__.py             # 包初始化文件（空）
    ├── conftest.py             # pytest 共用 fixtures
    ├── test_pose_estimator.py  # PoseEstimator 单元测试
    ├── test_skeleton_converter.py  # SkeletonConverter 单元测试
    ├── test_action_recognizer.py   # ActionRecognizer 单元测试
    └── test_pipeline_integration.py  # 端到端集成测试
```

---

## 脚本运行指南

以下是从零开始搭建运行环境到实际运行动作检测的**完整步骤**，请按顺序执行。

### 第一步：安装依赖

**前置条件：** 已安装 Python 3.10 和 [uv](https://docs.astral.sh/uv/) 包管理器。

```bash
uv sync
```

此命令会根据 `pyproject.toml` 和 `uv.lock` 安装所有项目依赖（PyTorch、ultralytics、mmaction2 等），并创建虚拟环境。

### 第二步：修补 mmaction2

```bash
uv run python patch_mmaction.py
```

mmaction2 1.2.0 的 pip 包存在已知 bug——缺少 `drn` 模块目录导致导入报错。此脚本自动将有问题的 `from .drn.drn import DRN` 导入包裹在 `try-except` 中，使其不会阻塞程序运行。

**说明：** 此步骤只需运行一次。脚本具有幂等性，重复运行会自动跳过已修补的情况。

### 第三步：下载模型权重

```bash
uv run python -m src.download_models
```

此脚本会下载两个预训练模型：
- **STGCN++ 权重** (`models/stgcnpp_ntu60_xsub_2d.pth`) — 从 OpenMMLab 官方源下载
- **YOLO26n-pose 权重** (`yolo26n-pose.pt`) — 通过 ultralytics 库自动下载

**说明：** 如果模型文件已存在会自动跳过。如果项目中已经包含了这两个文件，此步骤可省略。

### 第四步：验证安装

```bash
uv run python verify_pipeline.py
```

使用合成数据（随机生成图像）依次测试四个模块是否正常工作：
1. YOLO26n-pose 姿态检测
2. 骨架数据格式转换
3. STGCN++ 动作识别
4. 流水线集成

**不需要真实视频**即可验证，适合确认环境配置正确。

### 第五步：运行动作检测

**处理视频文件：**

```bash
uv run python -m src.main --video video.mp4
```

**使用摄像头实时检测：**

```bash
uv run python -m src.main --camera 0
```

**带自定义参数运行：**

```bash
uv run python -m src.main --video input.mp4 --device cuda:0 --window-size 64 --stride 8 --top-k 3
```

运行时会打开一个可视化窗口（可通过 `--no-show` 禁用），窗口中显示实时画面和 Top-K 动作识别结果。按 `q` 键退出。

### 运行测试

```bash
uv run pytest
```

运行 `tests/` 目录下的所有单元测试和集成测试。需要模型文件存在（`yolo26n-pose.pt` 和 `models/stgcnpp_ntu60_xsub_2d.pth`），否则相关测试会自动跳过。

---

## 各文件详细说明

### 根目录文件

#### `pyproject.toml` — 项目配置

定义项目元数据、依赖声明和工具配置：
- **核心依赖**：`torch 2.1.0`、`torchvision 0.16.0`（默认 CPU 版本）、`ultralytics >=8.4.12`、`mmaction2 1.2.0`、`mmcv-lite`、`mmengine`、`numpy <2.0`
- **开发依赖**：`pytest >=9.0.2`
- **PyTorch 源**：配置了 CPU 版 wheel 索引，可改为 CUDA 版本
- **pytest 配置**：测试路径为 `tests/`，忽略 mmcv 的 UserWarning

#### `patch_mmaction.py` — mmaction2 导入修补工具

| 属性 | 值 |
|---|---|
| 类型 | 独立工具脚本（直接运行） |
| 运行方式 | `uv run python patch_mmaction.py` |
| 运行时机 | 安装依赖后运行一次 |
| 作用 | 修复 mmaction2 1.2.0 pip 包中 `drn` 模块缺失导致的导入错误 |

**原理：** 找到 mmaction 安装路径中的 `models/localizers/__init__.py`，将 `from .drn.drn import DRN` 包裹在 `try-except` 块中。

#### `verify_pipeline.py` — 端到端验证脚本

| 属性 | 值 |
|---|---|
| 类型 | 独立工具脚本（直接运行） |
| 运行方式 | `uv run python verify_pipeline.py` |
| 运行时机 | 环境配置完成后，用于验证 |
| 作用 | 使用合成数据逐步测试 4 个组件 |

**验证步骤：**
1. 生成 480×640 随机图像 → YOLO26n-pose 姿态检测
2. 模拟 8 帧数据 → SkeletonConverter 滑动窗口转换
3. 转换后的数据 → STGCN++ 动作识别推理
4. 综合集成状态确认

#### `video.mp4` — 测试视频

用于测试和演示的视频文件，可作为 `--video` 参数的输入。

#### `yolo26n-pose.pt` — YOLO 模型权重

YOLO26n-pose 预训练模型，由 ultralytics 提供，检测 17 个 COCO 人体关键点。

---

### src/ 源码模块

#### `src/__main__.py` — 模块入口

允许通过 `python -m src` 方式运行项目，内部调用 `main.py` 中的 `main()` 函数。

#### `src/main.py` — CLI 命令行入口

| 属性 | 值 |
|---|---|
| 类型 | 主程序入口（直接运行） |
| 运行方式 | `uv run python -m src.main --video <路径>` 或 `--camera <ID>` |
| 作用 | 解析命令行参数、初始化流水线、启动视频/摄像头处理 |

**职责：**
- 解析输入源（`--video` 或 `--camera`，二选一，必填）
- 解析模型路径、推理设备、窗口参数等
- 检查模型文件是否存在
- 创建 `ActionDetectionPipeline` 实例并启动处理
- 定义 `print_prediction()` 回调函数，打印每次识别结果到控制台

#### `src/pipeline.py` — 整合流水线

| 属性 | 值 |
|---|---|
| 类型 | 核心模块（被 main.py 导入） |
| 核心类 | `ActionDetectionPipeline` |
| 作用 | 串联 YOLO → 转换器 → STGCN++ 三个组件，提供视频/摄像头处理接口 |

**`ActionDetectionPipeline` 类：**
- `__init__()` — 初始化三个子组件：`PoseEstimator`、`ActionRecognizer`、`SkeletonConverter`
- `process_video(video_path, callback, show)` — 处理视频文件，逐帧读取、检测、推理，可选显示带标注的窗口
- `process_camera(camera_id, callback)` — 处理摄像头实时输入，始终显示可视化窗口
- `_process_frame(frame)` — 处理单帧：姿态检测 → 加入滑动窗口 → 触发推理（若条件满足）
- `_draw_annotations(frame, predictions)` — 在帧上绘制半透明背景 + 动作识别结果文字 + 缓冲区进度
- `reset()` — 重置滑动窗口状态

#### `src/pose_estimator.py` — 姿态检测模块

| 属性 | 值 |
|---|---|
| 类型 | 核心模块（被 pipeline.py 导入） |
| 核心类 | `PoseEstimator`、`PoseResult` |
| 依赖 | ultralytics YOLO |
| 作用 | 封装 YOLO26n-pose，对每帧图像检测所有人的骨架关键点 |

**`PoseResult` 数据类（dataclass）：**
- `keypoints` — 关键点坐标 (N, 17, 2)，N 为检测到的人数
- `keypoint_scores` — 关键点置信度 (N, 17)
- `boxes` — 检测框 (N, 4)，xyxy 格式
- `box_scores` — 检测框置信度 (N,)
- `img_shape` — 图像尺寸 (H, W)

**`PoseEstimator` 类：**
- `predict(frame)` — 单帧姿态检测，返回 `PoseResult`
- `predict_batch(frames)` — 批量检测（内部逐帧调用）

#### `src/skeleton_converter.py` — 骨架数据转换模块

| 属性 | 值 |
|---|---|
| 类型 | 核心模块（被 pipeline.py 导入） |
| 核心类 | `SkeletonConverter` |
| 作用 | 维护滑动窗口，将 YOLO 输出转换为 MMAction2 所需的 pose_results 格式 |

**`SkeletonConverter` 类：**
- `add_frame(pose_result)` — 向缓冲区添加一帧数据，返回是否应触发推理
- `get_pose_results()` — 获取当前窗口的转换数据，返回 `(pose_results, img_shape)`
- `reset()` — 清空缓冲区和帧计数
- `is_ready` 属性 — 窗口是否已满
- `buffer_size` 属性 — 当前缓冲区帧数

**关键逻辑：**
- 使用 `deque(maxlen=window_size)` 维护固定大小的滑动窗口
- 当缓冲区填满且距上次输出间隔 ≥ stride 时触发推理
- 按检测框置信度排序，每帧最多保留 `max_persons` 个人
- 没有检测到人时填充零值，确保数据格式一致

#### `src/action_recognizer.py` — 动作识别模块

| 属性 | 值 |
|---|---|
| 类型 | 核心模块（被 pipeline.py 导入） |
| 核心类 | `ActionRecognizer` |
| 依赖 | mmaction2 |
| 作用 | 封装 STGCN++，基于骨架序列进行动作分类 |

**`ActionRecognizer` 类：**
- `recognize(pose_results, img_shape)` — 对一段骨架序列推理，返回全部 60 类动作及其置信度（按降序排列）
- `recognize_top_k(pose_results, img_shape, top_k)` — 返回前 K 个结果

**内置常量 `NTU60_LABELS`**：60 种 NTU RGB+D 60 动作类别标签列表（英文）。

#### `src/download_models.py` — 模型下载工具

| 属性 | 值 |
|---|---|
| 类型 | 工具脚本（直接运行） |
| 运行方式 | `uv run python -m src.download_models` |
| 作用 | 下载 STGCN++ 和 YOLO26n-pose 预训练权重 |

**功能：**
- `download_stgcnpp()` — 从 OpenMMLab 下载 STGCN++ 权重到 `models/` 目录，显示下载进度条
- `download_yolo_pose()` — 通过初始化 ultralytics YOLO 模型触发自动下载
- 已有文件时自动跳过

---

### configs/ 配置文件

#### `configs/stgcnpp_config.py` — STGCN++ 推理配置

定义模型结构和推理流水线：
- **模型类型**：`RecognizerGCN`
- **骨干网络**：`STGCN`（自适应图卷积 + 残差连接 + 多尺度时间卷积 MSTCN）
- **图结构**：COCO 布局，空间模式
- **分类头**：`GCNHead`，60 类，256 维输入通道
- **测试流水线**：
  1. `PreNormalize2D` — 2D 坐标预归一化
  2. `GenSkeFeat` — 生成骨架特征（关节特征）
  3. `UniformSampleFrames` — 均匀采样 100 帧
  4. `PoseDecode` — 姿态解码
  5. `FormatGCNInput` — 格式化 GCN 输入（2 人）
  6. `PackActionInputs` — 打包动作输入

---

### models/ 模型权重

#### `models/stgcnpp_ntu60_xsub_2d.pth`

STGCN++ 预训练权重，在 NTU RGB+D 60 数据集（XSub 划分、2D 关节）上训练，支持 60 种动作识别。来源于 OpenMMLab 官方模型库。

---

### tests/ 测试模块

#### `tests/conftest.py` — 共用 Fixtures

为所有测试提供共享的 pytest fixtures：
- `project_root` — 项目根目录路径
- `fake_frame` — 合成 480×640 图像（带简易人形轮廓，提高检测率）
- `real_video_path` — 测试视频路径（不存在时跳过）
- `real_video_first_frame` — 真实视频第一帧
- `stgcnpp_config_path` — STGCN++ 配置路径
- `stgcnpp_checkpoint_path` — STGCN++ 权重路径（不存在时跳过）
- `yolo_model_path` — YOLO 模型路径（不存在时跳过）

#### `tests/test_pose_estimator.py`

测试 `PoseEstimator` 和 `PoseResult`：
- 验证 `PoseResult` 数据结构正确性
- 验证 YOLO 模型可加载并对图像进行姿态检测
- 检验输出的关键点形状、置信度等

#### `tests/test_skeleton_converter.py`

测试 `SkeletonConverter`：
- 验证滑动窗口缓冲区行为
- 验证推理触发条件（窗口满 + 步长）
- 验证数据格式转换正确性
- 验证人数截断逻辑
- 验证重置功能

#### `tests/test_action_recognizer.py`

测试 `ActionRecognizer`：
- 验证 NTU60 标签列表完整性（60 类）
- 验证 STGCN++ 模型加载和推理
- 验证 Top-K 结果格式和排序

#### `tests/test_pipeline_integration.py`

端到端集成测试：
- 使用真实模型测试完整流水线
- 测试视频处理功能
- 测试帧处理和标注绘制
- 测试重置功能

---

## 数据流与流水线原理

```
输入帧 (BGR, H×W×3)
    │
    ▼
┌─────────────────────────────────────────┐
│  PoseEstimator.predict(frame)           │
│  ├─ YOLO26n-pose 模型推理               │
│  └─ 返回 PoseResult:                    │
│       keypoints:      (N, 17, 2)        │  N = 检测到的人数
│       keypoint_scores: (N, 17)          │  17 = COCO 关键点数
│       boxes:           (N, 4)           │
│       box_scores:      (N,)             │
│       img_shape:       (H, W)           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  SkeletonConverter.add_frame(result)    │
│  ├─ 加入 deque 缓冲区                   │
│  ├─ 检查窗口是否满 (≥ window_size)       │
│  └─ 检查是否到达滑动步长 (≥ stride)      │
│     └─ 返回 True → 触发推理             │
└─────────────────────────────────────────┘
    │ (当返回 True 时)
    ▼
┌─────────────────────────────────────────┐
│  SkeletonConverter.get_pose_results()   │
│  ├─ 遍历窗口中所有帧                     │
│  ├─ 每帧按 box_scores 排序取前 max_persons│
│  ├─ 无人帧填充零值                       │
│  └─ 返回 (pose_results, img_shape):     │
│       pose_results: List[dict]          │
│         每个 dict 含:                    │
│           keypoints:       (M, 17, 2)   │  M ≤ max_persons
│           keypoint_scores: (M, 17)      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  ActionRecognizer.recognize_top_k(...)  │
│  ├─ mmaction2 inference_skeleton 推理    │
│  ├─ 获取 60 类动作概率分布               │
│  └─ 返回 Top-K (label, score) 列表      │
└─────────────────────────────────────────┘
    │
    ▼
输出: [("drink water", 0.85), ("eat meal", 0.05), ...]
```

---

## 命令行参数参考

通过 `uv run python -m src.main` 运行时，支持以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--video` | str | — | 视频文件路径（与 `--camera` 互斥，必填其一） |
| `--camera` | int | — | 摄像头设备 ID（与 `--video` 互斥，必填其一） |
| `--pose-model` | str | `yolo26n-pose.pt` | YOLO 姿态模型权重路径 |
| `--action-config` | str | `configs/stgcnpp_config.py` | STGCN++ 配置文件路径 |
| `--action-checkpoint` | str | `models/stgcnpp_ntu60_xsub_2d.pth` | STGCN++ 模型权重路径 |
| `--device` | str | `cpu` | 推理设备（`cpu` 或 `cuda:0`） |
| `--window-size` | int | `48` | 滑动窗口大小（帧数），越大识别越稳定但延迟越高 |
| `--stride` | int | `16` | 窗口滑动步长，越小推理频率越高 |
| `--pose-conf` | float | `0.5` | 姿态检测置信度阈值，低于此值的检测结果会被过滤 |
| `--max-persons` | int | `2` | 每帧最大跟踪人数 |
| `--top-k` | int | `5` | 返回的 Top-K 动作识别结果数量 |
| `--no-show` | flag | `false` | 不显示可视化窗口（仅视频模式有效） |

---

## 支持的动作类别

系统支持识别 NTU RGB+D 60 数据集定义的 60 种人体动作：

| 序号 | 动作 | 序号 | 动作 |
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

## GPU 支持

默认配置使用 CPU 推理。如需 GPU 加速，按以下步骤操作：

**1. 修改 `pyproject.toml` 中的 PyTorch 索引源：**

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cu118"  # CUDA 11.8
explicit = true
```

常用 CUDA 版本对应 URL：
- CUDA 11.8: `https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `https://download.pytorch.org/whl/cu121`

**2. 重新安装依赖：**

```bash
uv sync
```

**3. 运行时指定 GPU 设备：**

```bash
uv run python -m src.main --video input.mp4 --device cuda:0
```
