# -*- coding: utf-8 -*-
# @Time    : 25/02/2025 16:19
# @Author  : TuanThanh
# @FileName: deformable_detr_twostage_refine_cdn_r50_16x2_50e_dota.py
# @Software: Vscode
_base_ = "deformable_detr_refine_cdn_r50_16x2_50e_dota.py"
model = dict(
    bbox_head=dict(
        as_two_stage=True,
        two_stage_type="standard",
        frm_cfgs=[
            dict(in_channels=256, featmap_strides=[8, 16, 32, 64, 128]),
            dict(in_channels=256, featmap_strides=[8, 16, 32, 64, 128]),
        ],
    )
)
