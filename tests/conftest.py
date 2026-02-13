"""测试共用 fixtures。"""

from pathlib import Path

import cv2
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def fake_frame():
    """生成一张 480x640 的合成图像（带简单人形轮廓以提高检测率）。"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (180, 180, 180)  # 浅灰色背景
    # 画一个简易人形：头、身体、四肢
    # 头
    cv2.circle(frame, (320, 100), 30, (100, 80, 60), -1)
    # 身体
    cv2.rectangle(frame, (290, 130), (350, 280), (60, 60, 120), -1)
    # 左臂
    cv2.line(frame, (290, 150), (230, 250), (60, 60, 120), 12)
    # 右臂
    cv2.line(frame, (350, 150), (410, 250), (60, 60, 120), 12)
    # 左腿
    cv2.line(frame, (305, 280), (280, 420), (40, 40, 100), 14)
    # 右腿
    cv2.line(frame, (335, 280), (360, 420), (40, 40, 100), 14)
    return frame


@pytest.fixture
def real_video_path():
    """返回测试视频路径（如果存在）。"""
    p = PROJECT_ROOT / "video.mp4"
    if p.exists():
        return str(p)
    pytest.skip("video.mp4 不存在，跳过真实视频测试")


@pytest.fixture
def real_video_first_frame(real_video_path):
    """从真实视频中读取第一帧。"""
    cap = cv2.VideoCapture(real_video_path)
    ret, frame = cap.read()
    cap.release()
    assert ret, "无法读取视频第一帧"
    return frame


@pytest.fixture
def stgcnpp_config_path(project_root):
    return str(project_root / "configs" / "stgcnpp_config.py")


@pytest.fixture
def stgcnpp_checkpoint_path(project_root):
    p = project_root / "models" / "stgcnpp_ntu60_xsub_2d.pth"
    if not p.exists():
        pytest.skip("STGCN++ 权重文件不存在")
    return str(p)


@pytest.fixture
def yolo_model_path(project_root):
    p = project_root / "yolo26n-pose.pt"
    if not p.exists():
        pytest.skip("YOLO 模型不存在")
    return str(p)
