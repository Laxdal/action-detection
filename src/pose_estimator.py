"""YOLO26n-pose 骨架检测封装模块。

使用 ultralytics YOLO26n-pose 模型检测每帧中所有人的 17 个 COCO 关键点。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class PoseResult:
    """单帧的姿态检测结果。

    Attributes:
        keypoints: 关键点坐标, shape (N, 17, 2), N 为人数。
        keypoint_scores: 关键点置信度, shape (N, 17)。
        boxes: 检测框, shape (N, 4), 格式为 xyxy。
        box_scores: 检测框置信度, shape (N,)。
        img_shape: 原始图像尺寸 (H, W)。
        track_ids: 追踪 ID, shape (N,), 可选。仅在使用 predict_track 时填充。
    """

    keypoints: np.ndarray
    keypoint_scores: np.ndarray
    boxes: np.ndarray
    box_scores: np.ndarray
    img_shape: tuple[int, int]
    track_ids: Optional[np.ndarray] = None


class PoseEstimator:
    """YOLO26n-pose 骨架检测器。

    Args:
        model_path: 模型权重路径，默认使用预训练的 yolo26n-pose.pt。
        device: 推理设备，如 'cpu' 或 'cuda:0'。
        conf_threshold: 检测置信度阈值。
    """

    def __init__(
        self,
        model_path: str = "yolo26n-pose.pt",
        device: str = "cpu",
        conf_threshold: float = 0.5,
    ) -> None:
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)

    def predict(self, frame: np.ndarray) -> PoseResult:
        """对单帧图像进行姿态检测。

        Args:
            frame: BGR 格式的图像, shape (H, W, 3)。

        Returns:
            PoseResult: 包含关键点、置信度和检测框信息。
        """
        results = self.model(
            frame,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False,
        )
        result = results[0]
        h, w = result.orig_shape

        if result.keypoints is not None and len(result.keypoints) > 0:
            # keypoints.xy: (N, 17, 2), keypoints.conf: (N, 17)
            kpts = result.keypoints.xy.cpu().numpy()
            confs = result.keypoints.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            box_scores = result.boxes.conf.cpu().numpy()
        else:
            kpts = np.zeros((0, 17, 2), dtype=np.float32)
            confs = np.zeros((0, 17), dtype=np.float32)
            boxes = np.zeros((0, 4), dtype=np.float32)
            box_scores = np.zeros((0,), dtype=np.float32)

        return PoseResult(
            keypoints=kpts,
            keypoint_scores=confs,
            boxes=boxes,
            box_scores=box_scores,
            img_shape=(h, w),
        )

    def predict_track(self, frame: np.ndarray) -> PoseResult:
        """对单帧图像进行姿态检测 + 追踪（持久化追踪 ID）。

        使用 YOLO 内置的 BoT-SORT/ByteTrack 追踪器为每个人分配持久 ID。
        需要在连续帧上调用以维持追踪状态。

        Args:
            frame: BGR 格式的图像, shape (H, W, 3)。

        Returns:
            PoseResult: 包含关键点、置信度、检测框和追踪 ID。
        """
        results = self.model.track(
            frame,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False,
            persist=True,
        )
        result = results[0]
        h, w = result.orig_shape

        if result.keypoints is not None and len(result.keypoints) > 0:
            kpts = result.keypoints.xy.cpu().numpy()
            confs = result.keypoints.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            box_scores = result.boxes.conf.cpu().numpy()
            # 追踪 ID：boxes.id 可能为 None（追踪器尚未分配）
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = np.arange(kpts.shape[0])
        else:
            kpts = np.zeros((0, 17, 2), dtype=np.float32)
            confs = np.zeros((0, 17), dtype=np.float32)
            boxes = np.zeros((0, 4), dtype=np.float32)
            box_scores = np.zeros((0,), dtype=np.float32)
            track_ids = np.array([], dtype=int)

        return PoseResult(
            keypoints=kpts,
            keypoint_scores=confs,
            boxes=boxes,
            box_scores=box_scores,
            img_shape=(h, w),
            track_ids=track_ids,
        )

    def predict_batch(self, frames: list[np.ndarray]) -> list[PoseResult]:
        """对多帧图像进行批量姿态检测。

        Args:
            frames: BGR 格式的图像列表。

        Returns:
            list[PoseResult]: 每帧的姿态检测结果。
        """
        return [self.predict(frame) for frame in frames]
