"""整合流水线：视频/摄像头 -> 姿态检测 -> 骨架转换 -> 动作识别。

将 YOLO26n-pose 和 MMAction2 STGCN++ 串联起来，实现端到端的动作检测。
"""

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .action_recognizer import ActionRecognizer
from .pose_estimator import PoseEstimator
from .skeleton_converter import SkeletonConverter


class ActionDetectionPipeline:
    """动作检测流水线。

    Args:
        pose_model_path: YOLO 姿态模型路径。
        action_config_path: MMAction2 配置文件路径。
        action_checkpoint_path: MMAction2 模型权重路径。
        device: 推理设备。
        window_size: 滑动窗口大小（帧数）。
        stride: 窗口滑动步长。
        pose_conf_threshold: 姿态检测置信度阈值。
        max_persons: 最大跟踪人数。
        top_k: 返回 Top-K 个动作识别结果。
    """

    def __init__(
        self,
        pose_model_path: str = "yolo26n-pose.pt",
        action_config_path: str = "configs/stgcnpp_config.py",
        action_checkpoint_path: str = "models/stgcnpp_ntu60_xsub_2d.pth",
        device: str = "cpu",
        window_size: int = 48,
        stride: int = 16,
        pose_conf_threshold: float = 0.5,
        max_persons: int = 2,
        top_k: int = 5,
    ) -> None:
        self.device = device
        self.top_k = top_k

        print("[Pipeline] 正在加载 YOLO26n-pose 模型...")
        self.pose_estimator = PoseEstimator(
            model_path=pose_model_path,
            device=device,
            conf_threshold=pose_conf_threshold,
        )

        print("[Pipeline] 正在加载 STGCN++ 模型...")
        self.action_recognizer = ActionRecognizer(
            config_path=action_config_path,
            checkpoint_path=action_checkpoint_path,
            device=device,
        )

        self.converter = SkeletonConverter(
            window_size=window_size,
            stride=stride,
            max_persons=max_persons,
        )

        print("[Pipeline] 初始化完成。")

    def process_video(
        self,
        video_path: str,
        callback: Optional[Callable] = None,
        show: bool = False,
    ) -> list[dict]:
        """处理视频文件。

        Args:
            video_path: 视频文件路径。
            callback: 每次识别完成时的回调函数，接收
                (frame_idx, predictions) 参数。
            show: 是否显示带标注的视频窗口。

        Returns:
            所有识别结果列表，每个元素为
            {'frame_idx': int, 'predictions': list[tuple[str, float]]}。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        results = []
        frame_idx = 0
        latest_predictions = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                predictions = self._process_frame(frame)
                if predictions is not None:
                    latest_predictions = predictions
                    result = {
                        "frame_idx": frame_idx,
                        "predictions": predictions,
                    }
                    results.append(result)
                    if callback:
                        callback(frame_idx, predictions)

                if show:
                    display_frame = self._draw_annotations(
                        frame, latest_predictions
                    )
                    cv2.imshow("Action Detection", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()

        return results

    def process_camera(
        self,
        camera_id: int = 0,
        callback: Optional[Callable] = None,
        cam_width: Optional[int] = None,
        cam_height: Optional[int] = None,
    ) -> None:
        """处理摄像头实时输入。

        Args:
            camera_id: 摄像头设备 ID。
            callback: 每次识别完成时的回调函数。
            cam_width: 摄像头分辨率宽度。None 则使用摄像头默认值。
            cam_height: 摄像头分辨率高度。None 则使用摄像头默认值。
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")

        if cam_width is not None and cam_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

        frame_idx = 0
        latest_predictions = []

        try:
            print("[Pipeline] 按 'q' 退出摄像头模式...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Pipeline] 摄像头读取失败。")
                    break

                predictions = self._process_frame(frame)
                if predictions is not None:
                    latest_predictions = predictions
                    if callback:
                        callback(frame_idx, predictions)

                display_frame = self._draw_annotations(frame, latest_predictions)
                cv2.imshow("Action Detection - Camera", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_idx += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _process_frame(
        self, frame: np.ndarray
    ) -> Optional[list[tuple[str, float]]]:
        """处理单帧，返回识别结果（如果窗口已满）。

        Args:
            frame: BGR 格式的图像帧。

        Returns:
            如果触发了推理则返回预测结果，否则返回 None。
        """
        # 1. 姿态检测
        pose_result = self.pose_estimator.predict(frame)

        # 2. 添加到滑动窗口
        should_infer = self.converter.add_frame(pose_result)

        if not should_infer:
            return None

        # 3. 获取转换后的数据
        data = self.converter.get_pose_results()
        if data is None:
            return None

        pose_results, img_shape = data

        # 4. 动作识别
        predictions = self.action_recognizer.recognize_top_k(
            pose_results, img_shape, top_k=self.top_k
        )

        return predictions

    def _draw_annotations(
        self,
        frame: np.ndarray,
        predictions: list[tuple[str, float]],
    ) -> np.ndarray:
        """在帧上绘制动作识别标注。

        Args:
            frame: 原始帧。
            predictions: (标签, 置信度) 列表。

        Returns:
            带标注的帧。
        """
        display = frame.copy()

        # 绘制半透明背景
        if predictions:
            overlay = display.copy()
            cv2.rectangle(overlay, (5, 5), (400, 30 + 25 * len(predictions)),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        # 绘制预测结果
        for i, (label, score) in enumerate(predictions):
            text = f"{label}: {score:.3f}"
            y_pos = 28 + i * 25
            cv2.putText(
                display,
                text,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # 绘制缓冲区进度
        buf_size = self.converter.buffer_size
        win_size = self.converter.window_size
        progress_text = f"Buffer: {buf_size}/{win_size}"
        h = display.shape[0]
        cv2.putText(
            display,
            progress_text,
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return display

    def reset(self) -> None:
        """重置流水线状态。"""
        self.converter.reset()
