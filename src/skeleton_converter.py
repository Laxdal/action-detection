"""中间转换层：将 YOLO 关键点转换为 MMAction2 骨架推理格式。

负责维护滑动窗口、人物关联，以及将数据格式从 YOLO 输出转换为
MMAction2 inference_skeleton 所需的 pose_results 格式。
"""

from collections import deque
from typing import Optional

import numpy as np

from .pose_estimator import PoseResult


class SkeletonConverter:
    """滑动窗口骨架数据转换器。

    将多帧的 YOLO 姿态检测结果积累成 MMAction2 可用的格式。

    Args:
        window_size: 滑动窗口大小（帧数），默认 48。
        stride: 窗口滑动步长，默认 16。
        max_persons: 最大跟踪人数，默认 2。
    """

    def __init__(
        self,
        window_size: int = 48,
        stride: int = 16,
        max_persons: int = 2,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.max_persons = max_persons
        self._buffer: deque[PoseResult] = deque(maxlen=window_size)
        self._frame_count: int = 0
        self._last_output_frame: int = -stride  # 确保第一次窗口满时立即输出

    def add_frame(self, pose_result: PoseResult) -> bool:
        """添加一帧姿态检测结果到缓冲区。

        Args:
            pose_result: 单帧的姿态检测结果。

        Returns:
            bool: 是否应该触发一次动作识别推理。
        """
        self._buffer.append(pose_result)
        self._frame_count += 1

        # 检查是否应该输出：窗口已满且到达滑动步长
        if len(self._buffer) >= self.window_size:
            if (self._frame_count - self._last_output_frame) >= self.stride:
                return True
        return False

    def get_pose_results(self) -> Optional[tuple[list[dict], tuple[int, int]]]:
        """获取当前窗口的 pose_results 数据。

        将缓冲区中的数据转换为 mmaction2.apis.inference_skeleton 所需的
        pose_results 格式。

        Returns:
            如果窗口数据不足返回 None，否则返回 (pose_results, img_shape)：
            - pose_results: List[dict]，每个 dict 包含 'keypoints' 和
              'keypoint_scores'。
            - img_shape: (H, W) 原始帧尺寸。
        """
        if len(self._buffer) == 0:
            return None

        self._last_output_frame = self._frame_count

        frames = list(self._buffer)
        img_shape = frames[0].img_shape

        pose_results = []
        for frame_data in frames:
            n_persons = frame_data.keypoints.shape[0]

            if n_persons == 0:
                # 没有检测到人，填充零值
                kpts = np.zeros((1, 17, 2), dtype=np.float32)
                scores = np.zeros((1, 17), dtype=np.float32)
            else:
                # 按检测框置信度排序，取前 max_persons 个
                if n_persons > self.max_persons:
                    top_indices = np.argsort(frame_data.box_scores)[
                        -self.max_persons :
                    ][::-1]
                    kpts = frame_data.keypoints[top_indices]
                    scores = frame_data.keypoint_scores[top_indices]
                else:
                    kpts = frame_data.keypoints
                    scores = frame_data.keypoint_scores

            pose_results.append(
                {
                    "keypoints": kpts.astype(np.float32),
                    "keypoint_scores": scores.astype(np.float32),
                }
            )

        return pose_results, img_shape

    def reset(self) -> None:
        """重置缓冲区和帧计数。"""
        self._buffer.clear()
        self._frame_count = 0
        self._last_output_frame = -self.stride

    @property
    def is_ready(self) -> bool:
        """窗口是否有足够数据进行推理。"""
        return len(self._buffer) >= self.window_size

    @property
    def buffer_size(self) -> int:
        """当前缓冲区中的帧数。"""
        return len(self._buffer)
