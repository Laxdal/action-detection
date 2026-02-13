"""下载预训练模型权重。

用法:
    uv run python -m src.download_models
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

# STGCN++ NTU60 XSub 2D Joint 模型
STGCNPP_URL = (
    "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/"
    "stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/"
    "stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth"
)
STGCNPP_FILENAME = "stgcnpp_ntu60_xsub_2d.pth"

# YOLO26n-pose 模型（ultralytics 会自动下载，这里提供手动下载选项）
YOLO_MODEL_NAME = "yolo26n-pose.pt"


def download_file(url: str, dest: Path) -> None:
    """下载文件并显示进度。"""
    print(f"  下载: {url}")
    print(f"  保存到: {dest}")

    def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded / total_size * 100, 100)
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = "=" * filled + "-" * (bar_len - filled)
            sys.stdout.write(f"\r  [{bar}] {percent:.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress_hook)
    print()  # 换行


def download_stgcnpp(models_dir: Path) -> None:
    """下载 STGCN++ 权重。"""
    dest = models_dir / STGCNPP_FILENAME
    if dest.exists():
        print(f"[跳过] STGCN++ 权重已存在: {dest}")
        return

    print("[下载] STGCN++ 预训练权重 (NTU60 XSub 2D Joint)...")
    download_file(STGCNPP_URL, dest)
    print(f"[完成] STGCN++ 权重已保存到: {dest}")


def download_yolo_pose(project_root: Path) -> None:
    """触发 YOLO26n-pose 权重下载。

    ultralytics 在首次使用时会自动下载模型权重。
    这里通过初始化模型来触发下载。
    """
    dest = project_root / YOLO_MODEL_NAME
    if dest.exists():
        print(f"[跳过] YOLO26n-pose 模型已存在: {dest}")
        return

    print("[下载] YOLO26n-pose 模型权重...")
    try:
        from ultralytics import YOLO

        model = YOLO(YOLO_MODEL_NAME)
        print(f"[完成] YOLO26n-pose 模型已就绪。")
    except Exception as e:
        print(f"[错误] 下载 YOLO 模型失败: {e}")
        print("  请稍后重试，或手动下载模型。")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Action Detection - 模型下载工具")
    print("=" * 60)
    print()

    # 下载 STGCN++ 权重
    download_stgcnpp(models_dir)
    print()

    # 下载 YOLO26n-pose 权重
    download_yolo_pose(project_root)
    print()

    print("=" * 60)
    print("所有模型下载完成！")
    print()
    print("运行示例:")
    print("  uv run python -m src.main --video your_video.mp4")
    print("  uv run python -m src.main --camera 0")
    print("=" * 60)


if __name__ == "__main__":
    main()
