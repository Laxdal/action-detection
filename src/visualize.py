
"""Enhanced visualization script with skeleton rendering, keypoints, and tracking IDs.

Reuses the same data processing pipeline (PoseEstimator -> SkeletonConverter ->
ActionRecognizer) but replaces the rendering layer to display:
  - Skeleton bones and keypoint dots
  - Bounding boxes with person tracking ID and confidence
  - Action recognition results overlay

Usage:
    uv run python -m src.visualize --video video.mp4
    uv run python -m src.visualize --camera 0
    uv run python -m src.visualize --video video.mp4 --device cuda:0
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .action_recognizer import ActionRecognizer
from .pose_estimator import PoseEstimator, PoseResult
from .skeleton_converter import SkeletonConverter

# ---------------------------------------------------------------------------
# COCO 17-keypoint skeleton definition
# ---------------------------------------------------------------------------
# Index -> name mapping for reference:
#  0: nose,  1: left_eye,  2: right_eye,  3: left_ear,  4: right_ear,
#  5: left_shoulder,  6: right_shoulder,  7: left_elbow,  8: right_elbow,
#  9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

SKELETON_EDGES = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16),
]

# Color palette for different persons (BGR)
PERSON_COLORS = [
    (0, 255, 0),    # green
    (255, 128, 0),  # blue-ish orange
    (0, 128, 255),  # orange
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (255, 255, 0),  # cyan
    (128, 0, 255),  # purple
    (0, 255, 128),  # spring green
]

KEYPOINT_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
]


def get_person_color(person_id: int) -> tuple:
    """Get a consistent color for a person based on their tracking ID."""
    return PERSON_COLORS[person_id % len(PERSON_COLORS)]


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    color: tuple,
    kpt_threshold: float = 0.3,
) -> None:
    """Draw skeleton bones and keypoint dots for one person.

    Args:
        frame: Image to draw on (modified in-place).
        keypoints: (17, 2) array of x, y coordinates.
        scores: (17,) array of confidence scores.
        color: BGR color tuple.
        kpt_threshold: Minimum confidence to draw a keypoint / bone.
    """
    # Draw bones
    for i, j in SKELETON_EDGES:
        if scores[i] > kpt_threshold and scores[j] > kpt_threshold:
            pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw keypoint dots
    for k in range(17):
        if scores[k] > kpt_threshold:
            cx, cy = int(keypoints[k, 0]), int(keypoints[k, 1])
            cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_bbox_with_id(
    frame: np.ndarray,
    box: np.ndarray,
    person_id: int,
    confidence: float,
    color: tuple,
) -> None:
    """Draw bounding box with person ID label.

    Args:
        frame: Image to draw on (modified in-place).
        box: (4,) array in xyxy format.
        person_id: Tracking ID for this person.
        confidence: Detection confidence.
        color: BGR color tuple.
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID:{person_id} ({confidence:.2f})"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # Label background
    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )


def draw_action_panel(
    frame: np.ndarray,
    predictions: list[tuple[str, float]],
) -> None:
    """Draw action recognition results as a semi-transparent panel.

    Args:
        frame: Image to draw on (modified in-place).
        predictions: List of (label, score) tuples.
    """
    if not predictions:
        return

    panel_h = 32 + 24 * len(predictions)
    panel_w = 380
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame, "Action Recognition", (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1, cv2.LINE_AA,
    )
    for i, (label, score) in enumerate(predictions):
        bar_len = int(score * 180)
        y = 46 + i * 24
        # Score bar
        cv2.rectangle(frame, (10, y - 10), (10 + bar_len, y + 4), (0, 180, 0), -1)
        text = f"{label}: {score:.3f}"
        cv2.putText(
            frame, text, (10, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )


def draw_status_bar(
    frame: np.ndarray,
    frame_idx: int,
    n_persons: int,
    buffer_size: int,
    window_size: int,
) -> None:
    """Draw a status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    status = (
        f"Frame: {frame_idx}  |  Persons: {n_persons}  |  "
        f"Buffer: {buffer_size}/{window_size}  |  Press 'q' to quit"
    )
    cv2.putText(
        frame, status, (8, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main visualization loop
# ---------------------------------------------------------------------------

def print_prediction(frame_idx: int, predictions: list) -> None:
    """Print recognition results to terminal (English only)."""
    print(f"\n[Frame {frame_idx}] Action recognition results:")
    for i, (label, score) in enumerate(predictions, 1):
        print(f"  {i}. {label}: {score:.4f}")


def run_visualize(args: argparse.Namespace) -> None:
    """Main visualization entry point."""

    print("[Visualize] Loading YOLO26n-pose model...")
    pose_estimator = PoseEstimator(
        model_path=args.pose_model,
        device=args.device,
        conf_threshold=args.pose_conf,
    )

    print("[Visualize] Loading STGCN++ model...")
    action_recognizer = ActionRecognizer(
        config_path=args.action_config,
        checkpoint_path=args.action_checkpoint,
        device=args.device,
    )

    converter = SkeletonConverter(
        window_size=args.window_size,
        stride=args.stride,
        max_persons=args.max_persons,
    )

    print("[Visualize] Initialization complete.")

    # Open video source
    if args.video:
        source = args.video
        print(f"[Visualize] Processing video: {source}")
        cap = cv2.VideoCapture(source)
    else:
        source = args.camera
        print(f"[Visualize] Opening camera: {source}")
        cap = cv2.VideoCapture(source)
        # Set camera resolution to native instead of OpenCV's 640x480 default
        if args.cam_width and args.cam_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Visualize] Camera resolution set to {actual_w}x{actual_h}")

    if not cap.isOpened():
        print(f"[Error] Cannot open video source: {source}")
        sys.exit(1)

    frame_idx = 0
    latest_predictions: list[tuple[str, float]] = []
    total_recognitions = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    break  # End of video
                print("[Visualize] Camera read failed.")
                break

            # ---- Data processing (reuses same pipeline logic) ----
            # Use tracking mode for persistent person IDs
            pose_result = pose_estimator.predict_track(frame)

            # Feed into the same skeleton converter
            should_infer = converter.add_frame(pose_result)

            predictions = None
            if should_infer:
                data = converter.get_pose_results()
                if data is not None:
                    pose_results_list, img_shape = data
                    predictions = action_recognizer.recognize_top_k(
                        pose_results_list, img_shape, top_k=args.top_k,
                    )

            if predictions is not None:
                latest_predictions = predictions
                total_recognitions += 1
                print_prediction(frame_idx, predictions)

            # ---- Enhanced rendering ----
            display = frame.copy()
            n_persons = pose_result.keypoints.shape[0]

            for p in range(n_persons):
                pid = int(pose_result.track_ids[p]) if pose_result.track_ids is not None else p
                color = get_person_color(pid)

                # Draw skeleton
                draw_skeleton(
                    display,
                    pose_result.keypoints[p],
                    pose_result.keypoint_scores[p],
                    color,
                    kpt_threshold=args.kpt_threshold,
                )

                # Draw bounding box with ID
                draw_bbox_with_id(
                    display,
                    pose_result.boxes[p],
                    pid,
                    float(pose_result.box_scores[p]),
                    color,
                )

            # Draw action recognition panel
            draw_action_panel(display, latest_predictions)

            # Draw status bar
            draw_status_bar(
                display, frame_idx, n_persons,
                converter.buffer_size, converter.window_size,
            )

            cv2.imshow("Action Detection - Skeleton Visualizer", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n[Visualize] Done. Processed {frame_idx} frames, "
          f"{total_recognitions} recognitions.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Action Detection Visualizer with Skeleton + Tracking IDs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="Video file path")
    input_group.add_argument("--camera", type=int, help="Camera device ID")

    parser.add_argument("--pose-model", type=str, default="yolo26n-pose.pt",
                        help="YOLO pose model path")
    parser.add_argument("--action-config", type=str, default="configs/stgcnpp_config.py",
                        help="MMAction2 config file path")
    parser.add_argument("--action-checkpoint", type=str,
                        default="models/stgcnpp_ntu60_xsub_2d.pth",
                        help="MMAction2 checkpoint path")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Inference device (cpu or cuda:0)")
    parser.add_argument("--window-size", type=int, default=48,
                        help="Sliding window size (frames)")
    parser.add_argument("--stride", type=int, default=16,
                        help="Window sliding stride")
    parser.add_argument("--pose-conf", type=float, default=0.5,
                        help="Pose detection confidence threshold")
    parser.add_argument("--kpt-threshold", type=float, default=0.3,
                        help="Keypoint confidence threshold for rendering")
    parser.add_argument("--max-persons", type=int, default=2,
                        help="Max tracked persons")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K action recognition results")
    parser.add_argument("--cam-width", type=int, default=720,
                        help="Camera capture width (ignored for video files)")
    parser.add_argument("--cam-height", type=int, default=1280,
                        help="Camera capture height (ignored for video files)")

    args = parser.parse_args()

    # Validate checkpoint exists
    if not Path(args.action_checkpoint).exists():
        print(f"[Error] STGCN++ checkpoint not found: {args.action_checkpoint}")
        print("  Run: uv run python -m src.download_models")
        sys.exit(1)

    if args.video and not Path(args.video).exists():
        print(f"[Error] Video file not found: {args.video}")
        sys.exit(1)

    run_visualize(args)


if __name__ == "__main__":
    main()
