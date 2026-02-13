"""ActionRecognizer 单元测试。"""

import numpy as np
import pytest

from src.action_recognizer import ActionRecognizer, NTU60_LABELS


class TestNTU60Labels:
    """标签列表验证。"""

    def test_label_count(self):
        assert len(NTU60_LABELS) == 60

    def test_labels_are_strings(self):
        assert all(isinstance(l, str) for l in NTU60_LABELS)

    def test_no_duplicates(self):
        assert len(set(NTU60_LABELS)) == len(NTU60_LABELS)


class TestActionRecognizer:
    """ActionRecognizer 推理测试。"""

    @pytest.fixture(autouse=True)
    def setup(self, stgcnpp_config_path, stgcnpp_checkpoint_path):
        self.recognizer = ActionRecognizer(
            config_path=stgcnpp_config_path,
            checkpoint_path=stgcnpp_checkpoint_path,
            device="cpu",
        )

    def _make_pose_results(self, n_frames=30, n_persons=1):
        """生成模拟的 pose_results 列表。"""
        pose_results = []
        for _ in range(n_frames):
            pose_results.append(
                {
                    "keypoints": np.random.rand(n_persons, 17, 2).astype(
                        np.float32
                    )
                    * 640,
                    "keypoint_scores": np.random.rand(n_persons, 17).astype(
                        np.float32
                    ),
                }
            )
        return pose_results

    def test_recognize_returns_list(self):
        pose_results = self._make_pose_results()
        predictions = self.recognizer.recognize(pose_results, (480, 640))
        assert isinstance(predictions, list)
        assert len(predictions) == 60  # 60 个类

    def test_recognize_prediction_format(self):
        pose_results = self._make_pose_results()
        predictions = self.recognizer.recognize(pose_results, (480, 640))
        for label, score in predictions:
            assert isinstance(label, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_recognize_sorted_descending(self):
        pose_results = self._make_pose_results()
        predictions = self.recognizer.recognize(pose_results, (480, 640))
        scores = [s for _, s in predictions]
        assert scores == sorted(scores, reverse=True)

    def test_recognize_top_k(self):
        pose_results = self._make_pose_results()
        top5 = self.recognizer.recognize_top_k(
            pose_results, (480, 640), top_k=5
        )
        assert len(top5) == 5

    def test_recognize_top_k_1(self):
        pose_results = self._make_pose_results()
        top1 = self.recognizer.recognize_top_k(
            pose_results, (480, 640), top_k=1
        )
        assert len(top1) == 1
        label, score = top1[0]
        assert label in NTU60_LABELS

    def test_scores_sum_approximately_one(self):
        """softmax 输出的总和应接近 1。"""
        pose_results = self._make_pose_results()
        predictions = self.recognizer.recognize(pose_results, (480, 640))
        total = sum(s for _, s in predictions)
        assert abs(total - 1.0) < 0.01

    def test_different_frame_lengths(self):
        """不同帧长度都应可推理。"""
        for n_frames in [10, 30, 60, 100]:
            pose_results = self._make_pose_results(n_frames=n_frames)
            predictions = self.recognizer.recognize_top_k(
                pose_results, (480, 640), top_k=3
            )
            assert len(predictions) == 3

    def test_multi_person(self):
        """多人场景应正常工作。"""
        pose_results = self._make_pose_results(n_persons=3)
        predictions = self.recognizer.recognize_top_k(
            pose_results, (480, 640), top_k=5
        )
        assert len(predictions) == 5

    def test_zero_person_frames(self):
        """全空帧不应崩溃。"""
        pose_results = []
        for _ in range(30):
            pose_results.append(
                {
                    "keypoints": np.zeros((1, 17, 2), dtype=np.float32),
                    "keypoint_scores": np.zeros((1, 17), dtype=np.float32),
                }
            )
        predictions = self.recognizer.recognize_top_k(
            pose_results, (480, 640), top_k=5
        )
        assert len(predictions) == 5
