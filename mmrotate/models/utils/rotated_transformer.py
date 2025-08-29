# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
import random
from typing import Sequence, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmcv.cnn import (
    build_activation_layer,
    build_conv_layer,
    build_norm_layer,
    xavier_init,
)
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import (
    BaseTransformerLayer,
    TransformerLayerSequence,
    build_transformer_layer_sequence,
)
from torch.nn.init import normal_

from .builder import ROTATED_TRANSFORMER
from mmdet.models.utils import Transformer
from mmdet.models.utils.transformer import inverse_sigmoid
# from mmrotate.core import obb2poly, poly2obb
# # from mmrotate.core import obb2xyxy
# from mmdet.core import bbox_cxcywh_to_xyxy

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    warnings.warn(
        "`MultiScaleDeformableAttention` in MMCV has been moved to "
        "`mmcv.ops.multi_scale_deform_attn`, please update your MMCV"
    )
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def obb2poly_tr(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x = rboxes[..., 0]
    y = rboxes[..., 1]
    w = rboxes[..., 2]
    h = rboxes[..., 3]
    a = rboxes[..., 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=2)


def bbox_cxcywh_to_xyxy_tr(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [
        (cx - 0.5 * w),
        (cy - 0.5 * h),
        (cx + 0.5 * w),
        (cy - 0.5 * h),
        (cx - 0.5 * w),
        (cy + 0.5 * h),
        (cx + 0.5 * w),
        (cy + 0.5 * h),
    ]
    return torch.cat(bbox_new, dim=-1)


@ROTATED_TRANSFORMER.register_module()
class RotatedDeformableDetrTransformer(Transformer):
    """Implements the DeformableDETR transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(
        self,
        dino_style=False,
        d_model=256,
        as_two_stage=False,
        num_feature_levels=5,
        num_patterns=0,
        two_stage_num_proposals=300,  # num_queries
        # two stage
        two_stage_type="no",  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
        two_stage_pat_embed=0,
        two_stage_add_query_num=0,
        two_stage_learn_wh=False,
        two_stage_keep_all_tokens=True,
        # init query
        learnable_tgt_init=True,
        decoder_query_perturber=None,
        random_refpoints_xy=False,
        # evo of #anchors
        dec_layer_number=None,
        rm_self_attn_layers=None,
        # for dn
        embed_init_tgt=False,
        use_detached_boxes_dec_out=False,
        **kwargs
    ):
        super(RotatedDeformableDetrTransformer, self).__init__(**kwargs)
        # Thiết lập các tham số cơ bản
        self.dino_style = dino_style
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        
        self.embed_dims = self.encoder.embed_dims  # Lấy embed_dims từ encoder
        self.d_model = d_model
        self.random_refpoints_xy = random_refpoints_xy

        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.decoder_query_perturber = decoder_query_perturber
        self.embed_init_tgt = embed_init_tgt

        # Các tham số liên quan đến two stage
        self.two_stage_type = two_stage_type
        self.two_stage_num_proposals = two_stage_num_proposals
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens

        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(two_stage_type)

        self.init_layers()
        
        # evo of #anchors
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != 'no' or num_patterns == 0:
                assert dec_layer_number[0] == two_stage_num_proposals, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({two_stage_num_proposals})"
            else:
                assert dec_layer_number[0] == two_stage_num_proposals * num_patterns, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({two_stage_num_proposals}) * num_patterns({num_patterns})"
        
        self.init_weights()
        
        self.rm_self_attn_layers = rm_self_attn_layers
        # if rm_self_attn_layers is not None:
        #     print("Removing the self-attn in {} decoder layers".format(rm_self_attn_layers))
        #     for lid, dec_layer in enumerate(self.decoder.layers):
        #         if lid in rm_self_attn_layers:
        #             dec_layer.rm_self_attn_modules()
        
    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )

        if (self.two_stage_type != "no" and self.embed_init_tgt) or (self.two_stage_type == "no"):
            self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None
            
        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            if self.two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(
                    torch.Tensor(self.two_stage_pat_embed, self.d_model)
                )
                nn.init.normal_(self.pat_embed_for_2stage)

            if self.two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, self.d_model)

            if self.two_stage_learn_wh:
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None
                
            self.pos_trans = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points = nn.Linear(self.embed_dims, 2)
            self.init_ref_points(self.two_stage_num_proposals) # init self.refpoint_embed

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_init(self.reference_points, distribution="uniform", bias=0.0)
        if self.num_feature_levels > 1 and self.level_embeds is not None:
            normal_(self.level_embeds)  
            
        #TODO: add init for two_stage_learn_wh
        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05)))

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 5)

        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2]
            )
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    # TODO: add input hw to gen_encoder_output_proposals
    def gen_encoder_output_proposals(self, memory:Tensor, memory_padding_mask: Tensor, spatial_shapes: Tensor, learnedwh=None):
        """Generate proposals from encoded memory.

        Args:
            memory (Tensor) : The output of encoder,
                has shape (bs, num_key, embed_dim).  num_key is
                equal the number of points on feature map from
                all level.
            memory_padding_mask (Tensor): Padding mask for memory.
                has shape (bs, num_key).
            spatial_shapes (Tensor): The shape of all feature maps.
                has shape (num_level, 2).
            learnedwh (Tensor): The learned wh for proposals.

        Returns:
            tuple: A tuple of feature map and bbox prediction.

                - output_memory (Tensor): The input of decoder,  \
                    has shape (bs, num_key, embed_dim).  num_key is \
                    equal the number of points on feature map from \
                    all levels.
                - output_proposals (Tensor): The normalized proposal \
                    after a inverse sigmoid, has shape \
                    (bs, num_keys, 4).
        """

        N, S, C = memory.shape
        proposals = []
        _cur = 0
        # 生成网格一样的proposals
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H * W)].view(
                N, H, W, 1
            )
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(
                N, 1, 1, 2
            )
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            
            # TODO: add learnedwh, if learnedwh is None, use default value.
            if learnedwh is not None:
                wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0 ** lvl)
            else:
                wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
                
            angle = torch.zeros_like(mask_flatten_)
            proposal = torch.cat((grid, wh, angle), -1).view(N, -1, 5)
            # proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = (
            (output_proposals[..., :4] > 0.01) & (output_proposals[..., :4] < 0.99)
        ).all(-1, keepdim=True)
        # 反sigmoid函数 inversigmoid
        output_proposals[..., :4] = torch.log(
            output_proposals[..., :4] / (1 - output_proposals[..., :4])
        )
        # output_proposals = output_proposals.masked_fill(
        #     memory_padding_mask.unsqueeze(-1), float('inf'))
        # output_proposals = output_proposals.masked_fill(
        #     ~output_proposals_valid, float('inf'))
        output_proposals[..., :4] = output_proposals[..., :4].masked_fill(
            memory_padding_mask.unsqueeze(-1), 10000
        )
        output_proposals[..., :4] = output_proposals[..., :4].masked_fill(
            ~output_proposals_valid, 10000
        )

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            # TODO: check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals, num_pos_feats=128, temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)
        return pos

    def forward(
        self,
        mlvl_feats,
        mlvl_masks,
        query_embed,
        mlvl_pos_embeds,
        bbox_coder=None,
        reg_branches=None,
        cls_branches=None,
        first_stage=False,
        refpoint_embed=None,
        tgt=None,
        attn_mask=None,
        **kwargs
    ):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.


        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            # pos_embed.shape = [2, 256, 128, 128]
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # [bs, w*h, c]
            feat = feat.flatten(2).transpose(1, 2)
            # [bs, w*h]
            mask = mask.flatten(1)
            # [bs, w*h]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.num_feature_levels > 1 and self.level_embeds is not None:
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
            
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)
        # multi-scale reference points
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2
        )  # (H*W, bs, embed_dims)
        # 21760 = 128*128+64*64+32*32+16*16 query的个数
        # memory是编码后的每个query和keys在多层featuremap中对应的特征 一维特征 256
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs
        )
        ####################################################
        # End of encoder
        # memory.shape = (bs, hw, embed_dims)
        # mask_flatten.shape = (bs, hw)
        # lvl_pos_embed_flatten.shape = (hw, bs, embed_dims)
        # spatial_shapes.shape = (num_levels, 2)
        ####################################################
        
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        # TODO: Fix two stage
        if self.as_two_stage: # self.two_stage_type == "standard":
            # TODO: add input hw to gen_encoder_output_proposals
            # OPTIONAL
            if self.two_stage_learn_wh:
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None

            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, input_hw
            )

            enc_outputs_class = cls_branches[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact_angle = reg_branches[self.decoder.num_layers](output_memory) + output_proposals # (bs, hw, 5)

            if first_stage:
                return enc_outputs_coord_unact_angle

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1] # (bs, nq)

            # OPTION 2: Use method from DINO
            # gather boxes
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unact_angle, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 5))  # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 5)).sigmoid()  # sigmoid
            # gather tgt
            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            else:
                tgt_ = tgt_undetach.detach()
                
            # TODO: chỉnh sửa query_pos và query
            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1) # (bs, nq+cdn, 5)
                tgt = torch.cat([tgt, tgt_], dim=1) # (bs, nq+cdn, d_model)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_
            
            reference_points = refpoint_embed[..., :4].sigmoid()
            init_reference_out = reference_points
            
            # Prepare query_pos and query
            if self.dino_style:
                query_pos = None
                query = tgt
            else:
                pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(refpoint_embed[..., :4])))
                query_pos, query = torch.split(pos_trans_out, c, dim=2) 

        else:
            # TODO: query_embed sẽ tương ứng self.tgt_embed, thay thế tgt_ bằng query
            # OPTION 1
            """query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(refpoint_embed).sigmoid()
            init_reference_out = reference_points"""

            # OPTION 2
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            refpoint_embed_ = self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(self.two_stage_num_proposals, 1) # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat
                
            reference_points = refpoint_embed[..., :4].sigmoid()
            query = tgt
            init_reference_out = reference_points

        ####################################################
        # End of preparing tgt
        # tgt.shape = (bs, nq+cdn, d_model)
        # refpoint_embed.shape = (bs, nq+cdn, 5)
        # query.shape = (bs, nq+cdn, d_model)
        # reference_points.shape = (bs, nq+cdn, 4) sigmoid
        ####################################################
        
        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        # inter_references_inputs [nb_dec, bs, num_query, num_vertices, points]
        inter_states, inter_references, inter_references_inputs = self.decoder(
            query=query,
            key=None,
            value=memory,  # memory: hw, bs, embed_dims
            pos = lvl_pos_embed_flatten, # hw, bs, embed_dims
            query_pos=query_pos,  #
            key_padding_mask=mask_flatten,  # memory_key_padding_mask
            reference_points=reference_points,  # refpoints_sigmoid
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            bbox_coder=bbox_coder, 
            tgt_mask=attn_mask,
            **kwargs
        )
        
        ###################################################
        # End of decoder
        # inter_states.shape = (num_layers, nq+cdn, bs, d_model)
        # inter_references.shape = (num_layers, bs, nq+cdn, 4)
        ###################################################
        inter_references_out = inter_references
        if self.as_two_stage:
            return inter_states, init_reference_out, \
                   inter_references_out, enc_outputs_class, \
                   enc_outputs_coord_unact_angle, inter_references_inputs
        return inter_states, init_reference_out, \
               inter_references_out, None, None, inter_references_inputs


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RotatedDeformableDetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(
        self,
        *args,
        d_model=256,
        return_intermediate=False,
        query_dim=4,
        dec_layer_dropout_prob=None,
        decoder_query_perturber=None,
        dec_layer_number=None,
        **kwargs
    ):

        super(RotatedDeformableDetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.d_model = d_model

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_scale = None
        
        self.decoder_query_perturber = decoder_query_perturber
        
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == self.num_layers

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == self.num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        valid_ratios=None,
        reg_branches=None,
        cls_branches=None,
        bbox_coder=None,
        **kwargs
    ):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query # (nq, bs, embed_dims)
        intermediate = []
        intermediate_reference_points = []
        intermediate_reference_points_input = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                                         torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                                         valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            # print('output.shape', output.shape) # [1, 250, 256] [bs, num_query, embed_dims]
            # print('reference_points.shape', reference_points.shape) # [1, 250, 2] [bs, num_query, points]
            # print('reference_points_input.shape', reference_points_input.shape) # [1, 250, 4, 2] [bs, num_query, num_vertices, points]

            if reg_branches is not None:
                # tmp = obb2xyxy(reg_branches[lid](output), version='le90')
                tmp = reg_branches[lid](output)
                    if reference_points.shape[-1] == 4:
                        new_reference_points = tmp[..., :4] + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    else:
                        assert reference_points.shape[-1] == 2
                        new_reference_points = tmp
                        new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    
                    reference_points = new_reference_points.detach()

                output = output.permute(1, 0, 2)
                if self.return_intermediate:
                    intermediate.append(output) # (nq, bs, embed_dims)
                    intermediate_reference_points.append(reference_points) # (bs, nq, 4)
                    
        else: # OPTION 2: use conditional query
            for lid, layer in enumerate(self.layers):
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :] # (nq, bs, nlevel, 4)
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            

                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # (nq, bs, 2*embed_dims)
                raw_query_pos = self.ref_point_head(query_sine_embed)  # (nq, bs, embed_dims)
                pos_scale = self.query_scale(output) if self.query_scale is not None else 1
                query_pos = pos_scale * raw_query_pos
                
                dropflag = False
                if self.dec_layer_dropout_prob is not None:
                    prob = random.random()
                    if prob < self.dec_layer_dropout_prob[lid]:
                        dropflag = True
                        
                if not dropflag:
                    output = layer(
                        tgt = output,
                        tgt_query_pos = query_pos,
                        tgt_query_sine_embed = query_sine_embed,
                        tgt_key_padding_mask = None,
                        tgt_reference_points = reference_points_input,

                        memory = kwargs['value'],
                        memory_key_padding_mask = kwargs['key_padding_mask'],
                        memory_level_start_index = kwargs['level_start_index'],
                        memory_spatial_shapes = kwargs['spatial_shapes'],
                        memory_pos = kwargs['pos'],
                        
                        self_attn_mask = kwargs['tgt_mask'],
                        cross_attn_mask = None
                    )
 
                if reg_branches is not None:
                    # tmp = obb2xyxy(reg_branches[lid](output), version='le90')
                    tmp = reg_branches[lid](output)
                    if reference_points.shape[-1] == 4:
                        new_reference_points = tmp[..., :4] + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    else:
                        assert reference_points.shape[-1] == 2
                        new_reference_points = tmp
                        new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    
                    # giảm số lượng truy vấn (object queries) ở các decoder layers sau để tập trung vào các truy vấn có tiềm năng phát hiện vật thể tốt hơn.
                    if self.dec_layer_number is not None and lid != self.num_layers - 1:
                        nq_now = new_reference_points.shape[0]
                        select_number = self.dec_layer_number[lid + 1]
                        if nq_now != select_number:
                            class_unselected = cls_branches[lid](output)
                            topk_proposals = torch.topk(class_unselected.max(-1)[0], select_number, dim=0)[1] # new_nq, bs
                            new_reference_points = torch.gather(new_reference_points, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid 
                    
                    reference_points = new_reference_points.detach()

                if self.return_intermediate:
                    intermediate.append(output) # (nq, bs, embed_dims)
                    intermediate_reference_points.append(reference_points.transpose(0, 1)) # (bs, nq, 4)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points

def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

@TRANSFORMER_LAYER.register_module()
class DeformableTransformerDecoderLayer(BaseTransformerLayer):
    def __init__(self,
                attn_cfgs=None,
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                operation_order=None,
                norm_cfg=dict(type='LN'),
                init_cfg=None,
                batch_first=False,
                d_model=256, d_ffn=1024,
                dropout=0.1, activation="relu",
                n_levels=4, n_heads=8, n_points=4,
                use_deformable_box_attn=False,
                box_attn_type='roi_align',
                key_aware_type=None,
                decoder_sa_type='sa',
                module_seq=['sa', 'ca', 'ffn'],
                **kwargs
                 ):
        super().__init__(attn_cfgs=attn_cfgs, ffn_cfgs=ffn_cfgs, operation_order=operation_order, norm_cfg=norm_cfg, init_cfg=init_cfg, batch_first=batch_first)
        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']
        # cross attention
        # if use_deformable_box_attn:
        #     self.cross_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
        # else:
        #     self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.cross_attn = MultiScaleDeformableAttention(self.embed_dims, n_heads, n_levels, n_points, batch_first=batch_first)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        # assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        # if decoder_sa_type == 'ca_content':
        #     self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        # self attention
        if self.self_attn is not None:
            if self.decoder_sa_type == 'sa':
                q = k = self.with_pos_embed(tgt, tgt_query_pos)
                tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_label':
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_content':
                tgt2 = self.self_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                            tgt_reference_points.transpose(0, 1).contiguous(),
                            memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))

        return tgt

    def forward_ca(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        # cross attention
        if self.key_aware_type is not None:

            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        # tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
        #                        tgt_reference_points.transpose(0, 1).contiguous(),
        #                        memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
        tgt2 = self.cross_attn(query=tgt, value=memory, query_pos=tgt_query_pos,
                               key_padding_mask=memory_key_padding_mask, reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
                               spatial_shapes=memory_spatial_shapes, level_start_index=memory_level_start_index,)
                               
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):

        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':
                tgt = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            else:
                raise ValueError('unknown funcname {}'.format(funcname))

        return tgt
=======
            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_reference_points_input.append(reference_points_input)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points), torch.stack(intermediate_reference_points_input)

        return output, reference_points, reference_points_input # Return the last reference points for input IOU
>>>>>>> origin/feature/thinh/matching_degree_loss
