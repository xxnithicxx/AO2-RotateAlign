# -*- coding: utf-8 -*-
# @Time    : 25/02/2025 16:19
# @Author  : TuanThanh
# @FileName: deformable_detr_refine_cdn_r50_16x2_50e_dota.py
# @Software: Vscode
_base_ = "deformable_detr_cdn_r50_16x2_50e_dota.py"
model = dict(
    bbox_head=dict(
        with_box_refine=True,
    )
)
