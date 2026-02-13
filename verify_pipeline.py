"""端到端验证脚本：使用合成数据测试整个流水线。

不需要真实视频，通过生成随机帧来验证各模块连接正常。
"""

import numpy as np
import sys


def main():
    print("=" * 60)
    print("端到端流水线验证")
    print("=" * 60)

    # 1. 验证 YOLO26n-pose
    print("\n[1/4] 验证 YOLO26n-pose 姿态检测...")
    from src.pose_estimator import PoseEstimator

    pose_est = PoseEstimator(
        model_path="yolo26n-pose.pt",
        device="cpu",
        conf_threshold=0.3,
    )
    # 生成一帧 480x640 的随机图像
    fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pose_result = pose_est.predict(fake_frame)
    print(f"  检测到 {pose_result.keypoints.shape[0]} 个人")
    print(f"  关键点 shape: {pose_result.keypoints.shape}")
    print(f"  图像尺寸: {pose_result.img_shape}")
    print("  [OK] YOLO26n-pose 正常工作。")

    # 2. 验证骨架转换
    print("\n[2/4] 验证骨架数据转换...")
    from src.skeleton_converter import SkeletonConverter

    converter = SkeletonConverter(window_size=8, stride=4, max_persons=2)

    # 模拟 8 帧数据
    for i in range(8):
        from src.pose_estimator import PoseResult as PR

        mock_result = PR(
            keypoints=np.random.rand(1, 17, 2).astype(np.float32) * 640,
            keypoint_scores=np.random.rand(1, 17).astype(np.float32),
            boxes=np.array([[100, 100, 300, 400]], dtype=np.float32),
            box_scores=np.array([0.9], dtype=np.float32),
            img_shape=(480, 640),
        )
        ready = converter.add_frame(mock_result)

    assert ready, "窗口应该已满"
    data = converter.get_pose_results()
    assert data is not None
    pose_results, img_shape = data
    print(f"  窗口帧数: {len(pose_results)}")
    print(f"  每帧关键点 shape: {pose_results[0]['keypoints'].shape}")
    print(f"  图像尺寸: {img_shape}")
    print("  [OK] 骨架数据转换正常工作。")

    # 3. 验证 MMAction2 STGCN++
    print("\n[3/4] 验证 MMAction2 STGCN++ 动作识别...")
    from src.action_recognizer import ActionRecognizer

    recognizer = ActionRecognizer(
        config_path="configs/stgcnpp_config.py",
        checkpoint_path="models/stgcnpp_ntu60_xsub_2d.pth",
        device="cpu",
    )

    # 使用之前转换器生成的数据进行推理
    predictions = recognizer.recognize_top_k(pose_results, img_shape, top_k=5)
    print("  Top-5 预测结果:")
    for i, (label, score) in enumerate(predictions, 1):
        print(f"    {i}. {label}: {score:.4f}")
    print("  [OK] STGCN++ 动作识别正常工作。")

    # 4. 总结
    print("\n[4/4] 流水线集成验证...")
    print("  [OK] 所有组件均可正常工作。")

    print("\n" + "=" * 60)
    print("验证通过！可以使用以下命令运行：")
    print("  uv run python -m src.main --video <视频路径>")
    print("  uv run python -m src.main --camera 0")
    print("=" * 60)


if __name__ == "__main__":
    main()
