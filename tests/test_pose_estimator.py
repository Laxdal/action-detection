"""PoseEstimator 单元测试。"""

import numpy as np
import pytest

from src.pose_estimator import PoseEstimator, PoseResult


class TestPoseResult:
    """PoseResult 数据类测试。"""

    def test_create_empty(self):
        pr = PoseResult(
            keypoints=np.zeros((0, 17, 2)),
            keypoint_scores=np.zeros((0, 17)),
            boxes=np.zeros((0, 4)),
            box_scores=np.zeros((0,)),
            img_shape=(480, 640),
        )
        assert pr.keypoints.shape == (0, 17, 2)
        assert pr.img_shape == (480, 640)

    def test_create_with_data(self):
        pr = PoseResult(
            keypoints=np.random.rand(2, 17, 2).astype(np.float32),
            keypoint_scores=np.random.rand(2, 17).astype(np.float32),
            boxes=np.random.rand(2, 4).astype(np.float32),
            box_scores=np.array([0.9, 0.8], dtype=np.float32),
            img_shape=(720, 1280),
        )
        assert pr.keypoints.shape == (2, 17, 2)
        assert pr.box_scores.shape == (2,)


class TestPoseEstimator:
    """PoseEstimator 模型测试。"""

    @pytest.fixture(autouse=True)
    def setup(self, yolo_model_path):
        self.estimator = PoseEstimator(
            model_path=yolo_model_path, device="cpu", conf_threshold=0.3
        )

    def test_predict_returns_pose_result(self, fake_frame):
        result = self.estimator.predict(fake_frame)
        assert isinstance(result, PoseResult)
        assert result.img_shape == (480, 640)

    def test_predict_keypoint_shape(self, fake_frame):
        result = self.estimator.predict(fake_frame)
        n = result.keypoints.shape[0]
        assert result.keypoints.shape == (n, 17, 2)
        assert result.keypoint_scores.shape == (n, 17)
        assert result.boxes.shape == (n, 4)
        assert result.box_scores.shape == (n,)

    def test_predict_real_video_frame(self, real_video_first_frame):
        """在真实视频帧上测试，应检测到人。"""
        result = self.estimator.predict(real_video_first_frame)
        assert isinstance(result, PoseResult)
        # 真实视频应至少检测到 1 个人
        assert result.keypoints.shape[0] >= 1, "真实视频帧应检测到至少1个人"
        assert result.keypoints.shape[1] == 17
        assert result.keypoints.shape[2] == 2

    def test_predict_batch(self, fake_frame):
        frames = [fake_frame, fake_frame, fake_frame]
        results = self.estimator.predict_batch(frames)
        assert len(results) == 3
        assert all(isinstance(r, PoseResult) for r in results)

    def test_predict_black_image_no_crash(self):
        """纯黑图像不应崩溃。"""
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.estimator.predict(black)
        assert isinstance(result, PoseResult)
        # 纯黑图上可能检测不到人
        assert result.keypoints.shape[1] == 17 or result.keypoints.shape[0] == 0

    def test_keypoint_values_in_range(self, real_video_first_frame):
        """关键点坐标应在图像范围内。"""
        result = self.estimator.predict(real_video_first_frame)
        if result.keypoints.shape[0] > 0:
            h, w = result.img_shape
            # 允许小幅超出（YOLO 可能预测出界）
            assert result.keypoints[:, :, 0].max() < w + 50
            assert result.keypoints[:, :, 1].max() < h + 50
            assert result.keypoint_scores.min() >= 0.0
            assert result.keypoint_scores.max() <= 1.0
