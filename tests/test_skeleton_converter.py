"""SkeletonConverter 单元测试。"""

import numpy as np
import pytest

from src.pose_estimator import PoseResult
from src.skeleton_converter import SkeletonConverter


def _make_pose_result(n_persons=1, img_shape=(480, 640)):
    """构造一个模拟的 PoseResult。"""
    return PoseResult(
        keypoints=np.random.rand(n_persons, 17, 2).astype(np.float32) * 640,
        keypoint_scores=np.random.rand(n_persons, 17).astype(np.float32),
        boxes=np.random.rand(n_persons, 4).astype(np.float32) * 400,
        box_scores=np.random.rand(n_persons).astype(np.float32),
        img_shape=img_shape,
    )


class TestSkeletonConverter:
    """SkeletonConverter 功能测试。"""

    def test_init_defaults(self):
        conv = SkeletonConverter()
        assert conv.window_size == 48
        assert conv.stride == 16
        assert conv.max_persons == 2
        assert conv.buffer_size == 0
        assert not conv.is_ready

    def test_add_frame_not_ready(self):
        conv = SkeletonConverter(window_size=5, stride=3)
        for _ in range(4):
            ready = conv.add_frame(_make_pose_result())
            assert not ready
        assert conv.buffer_size == 4
        assert not conv.is_ready

    def test_add_frame_becomes_ready(self):
        conv = SkeletonConverter(window_size=5, stride=3)
        for i in range(5):
            ready = conv.add_frame(_make_pose_result())
        assert ready  # 第5帧应该触发
        assert conv.is_ready

    def test_stride_controls_output(self):
        conv = SkeletonConverter(window_size=4, stride=2)
        results_at = []
        for i in range(12):
            ready = conv.add_frame(_make_pose_result())
            if ready:
                results_at.append(i)
                conv.get_pose_results()  # 消费数据
        # 首次在 frame 3 (window_size=4)，之后每隔 stride=2
        assert len(results_at) >= 3

    def test_get_pose_results_format(self):
        conv = SkeletonConverter(window_size=8, stride=4, max_persons=2)
        for _ in range(8):
            conv.add_frame(_make_pose_result(n_persons=1))

        data = conv.get_pose_results()
        assert data is not None

        pose_results, img_shape = data
        assert len(pose_results) == 8
        assert img_shape == (480, 640)

        for frame_data in pose_results:
            assert "keypoints" in frame_data
            assert "keypoint_scores" in frame_data
            kpts = frame_data["keypoints"]
            scores = frame_data["keypoint_scores"]
            assert kpts.shape[1] == 17
            assert kpts.shape[2] == 2
            assert scores.shape[1] == 17
            assert kpts.dtype == np.float32
            assert scores.dtype == np.float32

    def test_max_persons_truncation(self):
        """超过 max_persons 时应截断到前 N 个。"""
        conv = SkeletonConverter(window_size=4, stride=2, max_persons=2)
        for _ in range(4):
            conv.add_frame(_make_pose_result(n_persons=5))

        data = conv.get_pose_results()
        pose_results, _ = data
        for frame_data in pose_results:
            assert frame_data["keypoints"].shape[0] <= 2

    def test_empty_frame_handling(self):
        """没有检测到人的帧不应导致崩溃。"""
        conv = SkeletonConverter(window_size=4, stride=2)
        for _ in range(4):
            empty_result = PoseResult(
                keypoints=np.zeros((0, 17, 2), dtype=np.float32),
                keypoint_scores=np.zeros((0, 17), dtype=np.float32),
                boxes=np.zeros((0, 4), dtype=np.float32),
                box_scores=np.zeros((0,), dtype=np.float32),
                img_shape=(480, 640),
            )
            conv.add_frame(empty_result)

        data = conv.get_pose_results()
        assert data is not None
        pose_results, _ = data
        # 应该用零值填充
        for frame_data in pose_results:
            assert frame_data["keypoints"].shape == (1, 17, 2)
            assert np.all(frame_data["keypoints"] == 0)

    def test_reset(self):
        conv = SkeletonConverter(window_size=4, stride=2)
        for _ in range(4):
            conv.add_frame(_make_pose_result())
        assert conv.buffer_size == 4

        conv.reset()
        assert conv.buffer_size == 0
        assert not conv.is_ready

    def test_sliding_window_overwrites(self):
        """超过窗口大小时旧数据应被丢弃。"""
        conv = SkeletonConverter(window_size=4, stride=2)
        for i in range(10):
            conv.add_frame(_make_pose_result())
        assert conv.buffer_size == 4  # deque maxlen 限制

    def test_mixed_person_counts(self):
        """帧间人数不同不应崩溃。"""
        conv = SkeletonConverter(window_size=4, stride=2, max_persons=3)
        for n in [0, 1, 3, 5]:
            conv.add_frame(_make_pose_result(n_persons=max(n, 0)))
        data = conv.get_pose_results()
        assert data is not None
        pose_results, _ = data
        assert len(pose_results) == 4
