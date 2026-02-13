"""MMAction2 骨架动作识别封装模块。

使用 STGCN++ 进行基于骨架的动作识别推理。
"""

from pathlib import Path
from typing import Optional

import numpy as np

# NTU RGB+D 60 动作类别标签
NTU60_LABELS = [
    "drink water",
    "eat meal/snack",
    "brushing teeth",
    "brushing hair",
    "drop",
    "pickup",
    "throw",
    "sitting down",
    "standing up (from sitting)",
    "clapping",
    "reading",
    "writing",
    "tear up paper",
    "wear jacket",
    "take off jacket",
    "wear a shoe",
    "take off a shoe",
    "wear on glasses",
    "take off glasses",
    "put on a hat/cap",
    "take off a hat/cap",
    "cheer up",
    "hand waving",
    "kicking something",
    "reach into pocket",
    "hopping (one foot jumping)",
    "jump up",
    "make a phone call/answer phone",
    "playing with phone/tablet",
    "typing on a keyboard",
    "pointing to something with finger",
    "taking a selfie",
    "check time (from watch)",
    "rub two hands together",
    "nod head/bow",
    "shake head",
    "wipe face",
    "salute",
    "put the palms together",
    "cross hands in front",
    "sneeze/cough",
    "staggering",
    "falling",
    "touch head (headache)",
    "touch chest (stomachache/heart pain)",
    "touch back (backache)",
    "touch neck (neckache)",
    "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm",
    "punching/slapping other person",
    "kicking other person",
    "pushing other person",
    "pat on back of other person",
    "point finger at the other person",
    "hugging other person",
    "giving something to other person",
    "touch other person's pocket",
    "handshaking",
    "walking towards each other",
    "walking apart from each other",
]


class ActionRecognizer:
    """STGCN++ 骨架动作识别器。

    Args:
        config_path: MMAction2 配置文件路径。
        checkpoint_path: 模型权重路径。
        device: 推理设备，如 'cpu' 或 'cuda:0'。
        labels: 动作类别标签列表，默认使用 NTU60 标签。
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cpu",
        labels: Optional[list[str]] = None,
    ) -> None:
        self.device = device
        self.labels = labels or NTU60_LABELS

        # 延迟导入避免顶层导入时的注册问题
        from mmaction.apis import init_recognizer

        self.model = init_recognizer(config_path, checkpoint_path, device=device)

    def recognize(
        self,
        pose_results: list[dict],
        img_shape: tuple[int, int],
    ) -> list[tuple[str, float]]:
        """对一段骨架序列进行动作识别。

        Args:
            pose_results: 每帧的姿态结果列表，每个 dict 包含
                'keypoints' (N, 17, 2) 和 'keypoint_scores' (N, 17)。
            img_shape: 原始图像尺寸 (H, W)。

        Returns:
            排序后的 (动作标签, 置信度) 列表，按置信度降序排列。
        """
        from mmaction.apis import inference_skeleton

        result = inference_skeleton(
            self.model,
            pose_results,
            img_shape,
        )

        # result.pred_score 是一个 Tensor，shape (num_classes,)
        scores = result.pred_score.cpu().numpy()

        # 按分数降序排列
        sorted_indices = np.argsort(scores)[::-1]

        predictions = []
        for idx in sorted_indices:
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            predictions.append((label, float(scores[idx])))

        return predictions

    def recognize_top_k(
        self,
        pose_results: list[dict],
        img_shape: tuple[int, int],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """返回 Top-K 动作识别结果。

        Args:
            pose_results: 每帧的姿态结果列表。
            img_shape: 原始图像尺寸 (H, W)。
            top_k: 返回前 K 个结果。

        Returns:
            Top-K 的 (动作标签, 置信度) 列表。
        """
        all_predictions = self.recognize(pose_results, img_shape)
        return all_predictions[:top_k]
