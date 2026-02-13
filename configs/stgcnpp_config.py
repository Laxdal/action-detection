"""STGCN++ 推理配置（基于 COCO 2D 关键点，NTU60 XSub）。

此配置用于骨架动作识别推理，使用 COCO 17-keypoint 布局。
模型在 NTU RGB+D 60 数据集上预训练，支持 60 种动作类别。
"""

_base_ = []

default_scope = "mmaction"

model = dict(
    type="RecognizerGCN",
    backbone=dict(
        type="STGCN",
        gcn_adaptive="init",
        gcn_with_res=True,
        tcn_type="mstcn",
        graph_cfg=dict(layout="coco", mode="spatial"),
    ),
    cls_head=dict(type="GCNHead", num_classes=60, in_channels=256),
)

test_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(
        type="UniformSampleFrames",
        clip_len=100,
        num_clips=1,
        test_mode=True,
    ),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="PackActionInputs"),
]
