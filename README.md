# Action Detection Prototype

基于 YOLO26n-pose + MMAction2 STGCN++ 的动作检测原型项目。 具体技术文档可见: [DOCS.md](DOCS.md)

## 架构

```
视频/摄像头 -> YOLO26n-pose (姿态检测) -> 中间转换层 -> STGCN++ (动作识别) -> 结果
```

- **YOLO26n-pose**: 检测每帧中所有人的 17 个 COCO 关键点
- **中间转换层**: 滑动窗口积累多帧骨架数据，转换为 MMAction2 格式
- **STGCN++**: 基于骨架序列进行动作识别（NTU60 60 类动作）

## 环境要求

- Python 3.10
- [uv](https://docs.astral.sh/uv/) 包管理器

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 修补 mmaction2 已知问题

```bash
uv run python patch_mmaction.py
```

### 3. 下载模型权重

```bash
uv run python -m src.download_models
```

### 4. 运行

处理视频文件：

```bash
uv run python -m src.main --video path/to/video.mp4
```

使用摄像头实时检测：

```bash
uv run python -m src.main --camera 0
```

### 5. 验证安装

```bash
uv run python verify_pipeline.py
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--video` | - | 视频文件路径 |
| `--camera` | - | 摄像头设备 ID |
| `--device` | cpu | 推理设备 (cpu / cuda:0) |
| `--window-size` | 48 | 滑动窗口大小（帧数） |
| `--stride` | 16 | 窗口滑动步长 |
| `--pose-conf` | 0.5 | 姿态检测置信度阈值 |
| `--max-persons` | 2 | 最大跟踪人数 |
| `--top-k` | 5 | 返回 Top-K 动作识别结果 |
| `--no-show` | false | 不显示可视化窗口 |

## 依赖版本

| 组件 | 版本 |
|---|---|
| Python | 3.10 |
| PyTorch | 2.1.0 (CPU) |
| ultralytics | 8.4.12 |
| mmcv-lite | 2.1.0 |
| mmengine | 0.10.7 |
| mmaction2 | 1.2.0 |
| numpy | <2.0 |

## 项目结构

```
action-detection/
  pyproject.toml           # 项目配置和依赖
  configs/
    stgcnpp_config.py      # STGCN++ 推理配置
  models/                  # 模型权重目录
  src/
    pose_estimator.py      # YOLO26n-pose 封装
    skeleton_converter.py  # 骨架数据转换（滑动窗口）
    action_recognizer.py   # STGCN++ 动作识别封装
    pipeline.py            # 整合流水线
    main.py                # CLI 入口
    download_models.py     # 模型下载工具
  verify_pipeline.py       # 端到端验证脚本
  patch_mmaction.py        # mmaction2 导入修补
```

## GPU 支持

如需使用 GPU，修改 `pyproject.toml` 中的 PyTorch 索引源为 CUDA 版本：

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cu118"  # CUDA 11.8
explicit = true
```

然后重新安装：

```bash
uv sync
```

运行时指定设备：

```bash
uv run python -m src.main --video input.mp4 --device cuda:0
```
