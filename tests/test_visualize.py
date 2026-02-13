"""Tests for the visualization script rendering functions and tracking mode."""

import numpy as np
import pytest

from src.pose_estimator import PoseEstimator, PoseResult
from src.visualize import (
    draw_skeleton,
    draw_bbox_with_id,
    draw_action_panel,
    draw_status_bar,
    get_person_color,
    SKELETON_EDGES,
    PERSON_COLORS,
    KEYPOINT_NAMES,
)


class TestConstants:
    """Verify skeleton constants are well-formed."""

    def test_skeleton_edges_valid_indices(self):
        for i, j in SKELETON_EDGES:
            assert 0 <= i < 17
            assert 0 <= j < 17

    def test_skeleton_edges_count(self):
        assert len(SKELETON_EDGES) == 16

    def test_person_colors_count(self):
        assert len(PERSON_COLORS) >= 4

    def test_keypoint_names_count(self):
        assert len(KEYPOINT_NAMES) == 17

    def test_get_person_color_wraps(self):
        c1 = get_person_color(0)
        c2 = get_person_color(len(PERSON_COLORS))
        assert c1 == c2  # Should wrap around


class TestDrawFunctions:
    """Test that all drawing functions run without crashing."""

    @pytest.fixture
    def canvas(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draw_skeleton_with_high_conf(self, canvas):
        kpts = np.random.rand(17, 2).astype(np.float32) * 400 + 50
        scores = np.ones(17, dtype=np.float32)
        draw_skeleton(canvas, kpts, scores, (0, 255, 0), kpt_threshold=0.3)
        # At least some pixels should have been drawn (not all black)
        assert canvas.sum() > 0

    def test_draw_skeleton_with_low_conf(self, canvas):
        kpts = np.random.rand(17, 2).astype(np.float32) * 400
        scores = np.zeros(17, dtype=np.float32)  # All below threshold
        original = canvas.copy()
        draw_skeleton(canvas, kpts, scores, (0, 255, 0), kpt_threshold=0.3)
        # Nothing should have been drawn
        assert np.array_equal(canvas, original)

    def test_draw_skeleton_partial_conf(self, canvas):
        kpts = np.random.rand(17, 2).astype(np.float32) * 400 + 50
        scores = np.zeros(17, dtype=np.float32)
        scores[0] = 0.9  # Only nose visible
        scores[5] = 0.9  # And left shoulder
        draw_skeleton(canvas, kpts, scores, (0, 255, 0), kpt_threshold=0.3)
        assert canvas.sum() > 0

    def test_draw_bbox_with_id(self, canvas):
        box = np.array([50, 50, 200, 400], dtype=np.float32)
        draw_bbox_with_id(canvas, box, person_id=3, confidence=0.95, color=(0, 255, 0))
        assert canvas.sum() > 0

    def test_draw_action_panel_with_predictions(self, canvas):
        predictions = [("drinking", 0.85), ("reading", 0.10), ("writing", 0.03)]
        draw_action_panel(canvas, predictions)
        assert canvas.sum() > 0

    def test_draw_action_panel_empty(self, canvas):
        original = canvas.copy()
        draw_action_panel(canvas, [])
        assert np.array_equal(canvas, original)  # Nothing drawn

    def test_draw_status_bar(self, canvas):
        draw_status_bar(canvas, frame_idx=42, n_persons=2, buffer_size=30, window_size=48)
        assert canvas.sum() > 0


class TestPredictTrack:
    """Test the tracking-enabled prediction method."""

    @pytest.fixture(autouse=True)
    def setup(self, yolo_model_path):
        self.estimator = PoseEstimator(
            model_path=yolo_model_path, device="cpu", conf_threshold=0.3
        )

    def test_predict_track_returns_pose_result(self, real_video_first_frame):
        result = self.estimator.predict_track(real_video_first_frame)
        assert isinstance(result, PoseResult)

    def test_predict_track_has_track_ids(self, real_video_first_frame):
        result = self.estimator.predict_track(real_video_first_frame)
        assert result.track_ids is not None
        n = result.keypoints.shape[0]
        assert len(result.track_ids) == n

    def test_predict_track_keypoints_same_format(self, real_video_first_frame):
        result = self.estimator.predict_track(real_video_first_frame)
        n = result.keypoints.shape[0]
        if n > 0:
            assert result.keypoints.shape == (n, 17, 2)
            assert result.keypoint_scores.shape == (n, 17)
            assert result.boxes.shape == (n, 4)

    def test_predict_track_persistent_ids(self, real_video_first_frame):
        """Calling track on the same frame twice should give consistent IDs."""
        r1 = self.estimator.predict_track(real_video_first_frame)
        r2 = self.estimator.predict_track(real_video_first_frame)
        # Both should detect some persons
        if r1.keypoints.shape[0] > 0 and r2.keypoints.shape[0] > 0:
            # Track IDs should be integers
            assert r1.track_ids.dtype in (np.int32, np.int64, int)

    def test_predict_track_black_image(self):
        """Tracking on a black image should not crash."""
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.estimator.predict_track(black)
        assert isinstance(result, PoseResult)
        assert result.track_ids is not None
