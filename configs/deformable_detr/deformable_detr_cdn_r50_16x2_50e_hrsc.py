# -*- coding: utf-8 -*-
# @Time    : 15/05/2025 01:15
# @Author  : TuanThanh
# @FileName: deformable_detr_cdn_r50_16x2_50e_hrsc2016.py
# @Software: Vscode
angle_version = "oc"
_base_ = [
    "../_base_/datasets/hrsc.py", 
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
model = dict(
    type="RotatedDeformableDETR",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    bbox_head=dict(
        type="RotatedDeformableDETRHead",
        num_query=250, # Kept from your DOTA config, reference HRSC uses 300
        num_classes=1,  # Changed to 1 for HRSC (typically single class: ship)
        in_channels=2048,
        sync_cls_avg_factor=True,
        # two stage
        as_two_stage=False,
        two_stage_type="no",
        # dn training
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_angle_noise_range=15,
        dn_labelbook_size=100, # Assuming this should be related to num_classes, but keeping your original for now. May need adjustment if HRSC has 1 class.
                               # For single class, dn_labelbook_size might be less critical or could be set to 1.
        transformer=dict(
            type="RotatedDeformableDetrTransformer",
            two_stage_num_proposals=250, # Kept from your DOTA config
            dino_style=True,
            encoder=dict(
                type="DetrTransformerEncoder",
                num_layers=6,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=dict(
                        type="MultiScaleDeformableAttention", embed_dims=256
                    ),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=("self_attn", "norm", "ffn", "norm"),
                ),
            ),
            decoder=dict(
                type="RotatedDeformableDetrTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                query_dim=4,
                d_model=256,
                dec_layer_dropout_prob=None,
                transformerlayers=dict(
                    type = "DeformableTransformerDecoderLayer", # Kept from your DOTA config
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(type="MultiScaleDeformableAttention", embed_dims=256),
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True, offset=-0.5
        ),
        bbox_coder=dict(
            type="DeltaXYWHAOBBoxCoder",
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(0.1, 0.1, 0.2, 0.2, 0.1),
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=2.0),
        reg_decoded_bbox=True,
        loss_iou=dict(type="RotatedIoULoss", loss_weight=5.0), # Kept 5.0 from DOTA, HRSC ref uses 8.0
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="Rotated_HungarianAssigner",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="RBBoxL1Cost", weight=2.0, box_format="xywha"),
            iou_cost=dict(type="RotatedIoUCost", iou_mode="iou", weight=5.0), # Kept 5.0 from DOTA, HRSC ref uses 8.0
        )
    ),
    test_cfg=dict(max_per_img=100), # Added from HRSC reference file
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RResize", img_scale=(800, 512)), # HRSC images are often smaller, e.g. (800, 512) is common for HRSC. Adjusted from (1024,1024). You may need to check what ../_base_/datasets/hrsc.py uses or expects.
    dict(
        type="RRandomFlip",
        flip_ratio=[0.25, 0.25, 0.25], # Kept from DOTA, HRSC often just uses horizontal flip_ratio=0.5
        direction=["horizontal", "vertical", "diagonal"], # Kept from DOTA
        version=angle_version,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(800, 512), # Adjusted to match train_pipeline for HRSC
        flip=False,
        transforms=[
            dict(type="RResize"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2, # Often batch size is adjusted based on dataset and GPU memory
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline, filter_empty_gt=False, version=angle_version), # Kept filter_empty_gt from DOTA
    val=dict(pipeline=test_pipeline, version=angle_version),
    test=dict(pipeline=test_pipeline, version=angle_version),
)
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=1e-4, # Kept from DOTA, HRSC ref uses 1e-4
    weight_decay=0.00001, # Kept from DOTA, HRSC ref uses 0.0001
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),
            "sampling_offsets": dict(lr_mult=0.1),
            "reference_points": dict(lr_mult=0.1),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)) # HRSC ref uses max_norm=0.1, but keeping yours
# learning policy
lr_config = dict(policy="step", step=[40]) # HRSC ref uses step=[30] or similar for shorter schedules like 36 epochs. Yours is 50 epochs.
runner = dict(type="EpochBasedRunner", max_epochs=50) # HRSC ref for 1x schedule usually means 36 epochs. This is 50 epochs.
find_unused_parameters = True
work_dir = "work_dirs/hrsc2016_refine/" # Updated work_dir