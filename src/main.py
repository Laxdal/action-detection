"""命令行入口：动作检测原型。

用法:
    # 处理视频文件
    uv run python -m src.main --video video.mp4

    # 使用摄像头
    uv run python -m src.main --camera 0

    # 指定设备和参数
    uv run python -m src.main --video input.mp4 --device cuda:0 --window-size 64 --top-k 3
"""

import argparse
import sys
from pathlib import Path


def print_prediction(frame_idx: int, predictions: list) -> None:
    """打印识别结果回调。"""
    print(f"\n[Frame {frame_idx}] 动作识别结果:")
    for i, (label, score) in enumerate(predictions, 1):
        print(f"  {i}. {label}: {score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Action Detection: YOLO26n-pose + MMAction2 STGCN++",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 输入源
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video", type=str, help="视频文件路径"
    )
    input_group.add_argument(
        "--camera", type=int, help="摄像头设备 ID"
    )

    # 模型路径
    parser.add_argument(
        "--pose-model",
        type=str,
        default="yolo26n-pose.pt",
        help="YOLO 姿态模型路径",
    )
    parser.add_argument(
        "--action-config",
        type=str,
        default="configs/stgcnpp_config.py",
        help="MMAction2 配置文件路径",
    )
    parser.add_argument(
        "--action-checkpoint",
        type=str,
        default="models/stgcnpp_ntu60_xsub_2d.pth",
        help="MMAction2 模型权重路径",
    )

    # 推理参数
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="推理设备 (cpu 或 cuda:0)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=48,
        help="滑动窗口大小（帧数）",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="窗口滑动步长",
    )
    parser.add_argument(
        "--pose-conf",
        type=float,
        default=0.5,
        help="姿态检测置信度阈值",
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=2,
        help="最大跟踪人数",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回 Top-K 动作识别结果",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不显示可视化窗口（仅视频模式有效）",
    )
    parser.add_argument(
        "--cam-width",
        type=int,
        default=720,
        help="摄像头分辨率宽度（仅摄像头模式有效）",
    )
    parser.add_argument(
        "--cam-height",
        type=int,
        default=1280,
        help="摄像头分辨率高度（仅摄像头模式有效）",
    )

    args = parser.parse_args()

    # 检查模型文件是否存在
    action_checkpoint = Path(args.action_checkpoint)
    if not action_checkpoint.exists():
        print(f"[错误] 找不到 STGCN++ 权重文件: {action_checkpoint}")
        print("请先下载权重文件:")
        print(
            "  uv run python -m src.download_models"
        )
        sys.exit(1)

    # 导入放在这里，避免顶层导入时间过长
    from .pipeline import ActionDetectionPipeline

    pipeline = ActionDetectionPipeline(
        pose_model_path=args.pose_model,
        action_config_path=args.action_config,
        action_checkpoint_path=args.action_checkpoint,
        device=args.device,
        window_size=args.window_size,
        stride=args.stride,
        pose_conf_threshold=args.pose_conf,
        max_persons=args.max_persons,
        top_k=args.top_k,
    )

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[错误] 找不到视频文件: {video_path}")
            sys.exit(1)

        print(f"[Main] 正在处理视频: {video_path}")
        results = pipeline.process_video(
            str(video_path),
            callback=print_prediction,
            show=not args.no_show,
        )
        print(f"\n[Main] 处理完成，共 {len(results)} 次识别。")

    elif args.camera is not None:
        print(f"[Main] 正在打开摄像头: {args.camera}")
        pipeline.process_camera(
            camera_id=args.camera,
            callback=print_prediction,
            cam_width=args.cam_width,
            cam_height=args.cam_height,
        )


if __name__ == "__main__":
    main()
