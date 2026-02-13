"""端到端集成测试：完整流水线。"""

import numpy as np
import pytest
import cv2

from src.pipeline import ActionDetectionPipeline


class TestPipelineIntegration:
    """使用真实模型的集成测试。"""

    @pytest.fixture(autouse=True)
    def setup(self, yolo_model_path, stgcnpp_config_path, stgcnpp_checkpoint_path):
        self.pipeline = ActionDetectionPipeline(
            pose_model_path=yolo_model_path,
            action_config_path=stgcnpp_config_path,
            action_checkpoint_path=stgcnpp_checkpoint_path,
            device="cpu",
            window_size=15,
            stride=8,
            pose_conf_threshold=0.3,
            max_persons=2,
            top_k=5,
        )

    def test_process_single_frame_no_output(self, fake_frame):
        """单帧不应触发推理（窗口未满）。"""
        result = self.pipeline._process_frame(fake_frame)
        assert result is None

    def test_process_enough_frames_produces_output(self, real_video_first_frame):
        """喂够窗口帧数后应产生输出。"""
        self.pipeline.reset()
        result = None
        for _ in range(20):  # > window_size=15
            result = self.pipeline._process_frame(real_video_first_frame)
            if result is not None:
                break
        assert result is not None, "窗口满后应产生识别结果"
        assert isinstance(result, list)
        assert len(result) == 5  # top_k=5
        for label, score in result:
            assert isinstance(label, str)
            assert isinstance(score, float)

    def test_process_video_file(self, real_video_path):
        """处理真实视频文件。"""
        collected = []

        def callback(frame_idx, predictions):
            collected.append(
                {"frame_idx": frame_idx, "predictions": predictions}
            )

        results = self.pipeline.process_video(
            real_video_path, callback=callback, show=False
        )

        # 应产生多次识别
        assert len(results) > 0, "应至少产生一次识别结果"
        assert len(collected) == len(results), "回调次数应与结果数一致"

        # 检验结果格式
        for r in results:
            assert "frame_idx" in r
            assert "predictions" in r
            assert isinstance(r["frame_idx"], int)
            assert isinstance(r["predictions"], list)
            for label, score in r["predictions"]:
                assert isinstance(label, str)
                assert 0.0 <= score <= 1.0

    def test_process_video_frame_indices_increasing(self, real_video_path):
        """结果中的帧索引应单调递增。"""
        results = self.pipeline.process_video(
            real_video_path, show=False
        )
        frame_indices = [r["frame_idx"] for r in results]
        assert frame_indices == sorted(frame_indices)
        # 检查没有重复
        assert len(frame_indices) == len(set(frame_indices))

    def test_reset_clears_state(self, real_video_first_frame):
        """重置后应重新开始积累。"""
        for _ in range(20):
            self.pipeline._process_frame(real_video_first_frame)
        self.pipeline.reset()
        assert self.pipeline.converter.buffer_size == 0

    def test_draw_annotations(self, fake_frame):
        """标注绘制不应崩溃。"""
        predictions = [("drinking", 0.85), ("reading", 0.1)]
        annotated = self.pipeline._draw_annotations(fake_frame, predictions)
        assert annotated.shape == fake_frame.shape
        assert annotated.dtype == np.uint8
        # 不应修改原图
        assert not np.array_equal(annotated, fake_frame)

    def test_draw_annotations_empty(self, fake_frame):
        """空预测列表不应崩溃。"""
        annotated = self.pipeline._draw_annotations(fake_frame, [])
        assert annotated.shape == fake_frame.shape

    def test_invalid_video_path_raises(self):
        """无效视频路径应抛出异常。"""
        with pytest.raises(ValueError, match="无法打开视频文件"):
            self.pipeline.process_video("nonexistent_video.mp4", show=False)
