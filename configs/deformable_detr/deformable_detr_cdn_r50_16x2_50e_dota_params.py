# -*- coding: utf-8 -*-
# @Time    : 25/02/2025 16:19
# @Author  : TuanThanh
# @FileName: deformable_detr_cdn_r50_16x2_50e_dota.py
# @Software: Vscode
angle_version = "oc"
_base_ = [
    "../_base_/datasets/dotav1.py",
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
        num_query=300,
        num_classes=15,
        in_channels=2048,
        sync_cls_avg_factor=True,
        random_refpoints_xy=False,
        # use refine
        with_box_refine=False,
        # two stage
        as_two_stage=True,
        two_stage_type="standard",
        two_stage_add_query_num=0,
        num_patterns=0,
        # dn training
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_angle_noise_range=15,
        dn_labelbook_size=100,
        
        # frm_cfgs=[
        #     dict(in_channels=256, featmap_strides=[16, 32, 64, 128]),
        #     dict(in_channels=256, featmap_strides=[16, 32, 64, 128])
        # ],
        transformer=dict(
            type="RotatedDeformableDetrTransformer",
            dino_style=True,
            # use_dab=False,
            num_feature_levels=4, # Giảm thành số lượng feature levels xuống 4
            two_stage_num_proposals=300,
            
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
                # use_dab=False,
                query_dim=4,
                d_model=256,
                dec_layer_dropout_prob=None,
                transformerlayers=dict(
                    type = "DeformableTransformerDecoderLayer",
                    # type="DetrTransformerDecoderLayer",
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
                    batch_first=False,
                    d_model=256, d_ffn=1024,
                    dropout=0.1, activation="relu",
                    n_levels=4, n_heads=8, n_points=4, # tăng n_points lên 8 để tăng độ chính xác
                    use_deformable_box_attn=False,
                    box_attn_type='roi_align',
                    key_aware_type=None,
                    decoder_sa_type='sa',
                    module_seq=['sa', 'ca', 'ffn'],
                ),
            ),
        ),
        positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, normalize=True, offset=-0.5, temperature=15000),
        bbox_coder=dict(
            type="DeltaXYWHAOBBoxCoder",
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(0.1, 0.1, 0.2, 0.2, 0.1),
        ),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type="L1Loss", loss_weight=2.0),
        reg_decoded_bbox=True,
        # loss_iou=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)
        loss_iou=dict(type="RotatedIoULoss", loss_weight=5.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="Rotated_HungarianAssigner",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="RBBoxL1Cost", weight=2.0, box_format="xywha"),
            iou_cost=dict(type="RotatedIoUCost", iou_mode="iou", weight=5.0),
            # iou_cost=dict(type='GaussianIoUCost', iou_mode='iou', weight=5.0)
        )
    ),
    test_cfg=dict(),
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RResize", img_scale=(1024, 1024)),
    dict(
        type="RRandomFlip",
        flip_ratio=[0.25, 0.25, 0.25],
        direction=["horizontal", "vertical", "diagonal"],
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
        img_scale=(1024, 1024),
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
# Gán test pipeline cho validation và test
data = dict(
    train=dict(pipeline=train_pipeline, filter_empty_gt=False, version=angle_version),
    val=dict(pipeline=test_pipeline, version=angle_version),
    test=dict(pipeline=test_pipeline, version=angle_version),
)
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=2e-4,
    weight_decay=0.05,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),
            "sampling_offsets": dict(lr_mult=0.2),
            "reference_points": dict(lr_mult=0.2),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# lr_config = dict(policy="step", step=[40])
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing', 
    min_lr_ratio=1e-5,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001
)

runner = dict(type="EpochBasedRunner", max_epochs=15)
find_unused_parameters = True
work_dir = "work_dirs/new_refine/"
