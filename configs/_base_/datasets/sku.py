dataset_type = "SKUDataset"
data_root = "/media/lhbac13/AO2-FIX/data/SKU110K/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RResize", img_scale=(768, 768)),
    dict(type="RRandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(768, 768),
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
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file="/media/lhbac13/AO2-FIX/data/SKU110K/SKU110K-R-Json/sku110k-r_train.json",
        img_prefix=data_root + "images/",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file="/media/lhbac13/AO2-FIX/data/SKU110K/SKU110K-R-Json/sku110k-r_val.json",
        img_prefix=data_root + "images/",
        pipeline=train_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file="/media/lhbac13/AO2-FIX/data/SKU110K/SKU110K-R-Json/sku110k-r_val.json",
        img_prefix=data_root + "images/",
        pipeline=test_pipeline,
    ),
)
