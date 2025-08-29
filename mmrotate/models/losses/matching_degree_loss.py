import warnings
import torch
import torch.nn as nn

from mmcv.ops import box_iou_rotated
from mmdet.models.losses.utils import weighted_loss

from ..builder import ROTATED_LOSSES
from ...core.bbox.iou_calculators.builder import ROTATED_IOU_CALCULATORS


def matching_degree(sa, fa, alpha=0.5, gamma=1.0, eps=1e-6, **kwargs):
    """Matching degree.

    This loss computes the matching degree as:
        md = α * sa + (1 - α) * fa - |sa - fa|^γ
    where:
        - sa is the spatial alignment (e.g., input IoU)
        - fa is the feature alignment (e.g., output IoU)

    The score will be clamped to a minimum value of `eps` to avoid log(0) in the
    loss calculation. 
    
    Args:
        sa (torch.Tensor): Spatial alignment value.
        fa (torch.Tensor): Feature alignment value.
        alpha (float): Weighting factor between sa and fa.
        gamma (float): Exponent for the uncertainty penalty.
        eps (float): Small value to avoid log(0).
        
    Returns:
        torch.Tensor: Score tensor.
    """
    # Compute regression uncertainty as the absolute difference between sa and fa.
    u = torch.abs(sa - fa)
    
    # Compute matching degree.
    md = alpha * sa + (1 - alpha) * fa - (u ** gamma)
    
    return md


@weighted_loss
def matching_degree_loss(pred, target, alpha=0.5, gamma=1.0, eps=1e-6, reduction='mean', **kwargs):
    """Matching degree loss.

    This loss computes the matching degree as:
        md = α * sa + (1 - α) * fa - |sa - fa|^γ
    where:
        - sa is the spatial alignment (e.g., input IoU)
        - fa is the feature alignment (e.g., output IoU)
    Both values must be passed in via **kwargs.

    The loss is defined as the negative log of the matching degree.
    
    Args:
        pred (torch.Tensor): Predicted matching degree (for compatibility, not directly used).
        target (torch.Tensor): Target value (for compatibility, not directly used).
        alpha (float): Weighting factor between sa and fa.
        gamma (float): Exponent for the uncertainty penalty.
        eps (float): Small value to avoid log(0).
        reduction (str): Reduction method ('none', 'mean', 'sum').
        **kwargs: Must contain 'sa' and 'fa', which are torch.Tensors.
        
    Returns:
        torch.Tensor: Loss tensor.
    """
    # TODO: Remove this test case
    # Add 'sa' and 'fa' to kwargs
    kwargs['sa'] = torch.tensor([0.8]) # bs
    kwargs['fa'] = torch.tensor([0.85])
    
    if 'sa' not in kwargs or 'fa' not in kwargs:
        raise ValueError("matching_degree_loss requires 'sa' and 'fa' in kwargs")
    sa = kwargs['sa']
    fa = kwargs['fa']
    
    # Compute regression uncertainty as the absolute difference between sa and fa.
    u = torch.abs(sa - fa)
    
    # Compute matching degree.
    md = alpha * sa + (1 - alpha) * fa - (u ** gamma)
    md = md.clamp(min=eps)
    
    # Compute loss as the negative logarithm of the matching degree.
    loss = -torch.log(md)
    
    # Apply the reduction.
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


@ROTATED_LOSSES.register_module()
class MatchingDegreeLoss(nn.Module):
    """Matching Degree Loss Module.

    This module integrates the matching degree loss into the training
    pipeline. The matching degree is computed from the spatial alignment
    (sa) and feature alignment (fa) provided via kwargs.
    
    Args:
        alpha (float): Weighting factor between spatial and feature alignment.
        gamma (float): Exponent for the uncertainty penalty term.
        eps (float): Small constant to avoid log(0).
        reduction (str): Reduction method ('none', 'mean', 'sum').
        loss_weight (float): Weight of the loss.
    """
    def __init__(self,
                 alpha=0.5,
                 gamma=1.0,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(MatchingDegreeLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.
        
        Args:
            pred (torch.Tensor): The prediction (not directly used in loss computation).
            target (torch.Tensor): The target (not directly used in loss computation).
            weight (torch.Tensor, optional): Weight for each prediction.
            avg_factor (int, optional): Average factor for loss averaging.
            reduction_override (str, optional): Override the reduction method.
            **kwargs: Additional keyword arguments, must include 'sa' and 'fa'.
            
        Returns:
            torch.Tensor: Loss tensor.
        """
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * matching_degree_loss(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            eps=self.eps,
            reduction=reduction,
            **kwargs)
        return loss


@ROTATED_IOU_CALCULATORS.register_module()
class RBboxOverlaps2DIgnoreRot(object):
    """2D Overlaps (IoU/GIoU/IoF) Calculator ignoring rotation angle."""

    def __call__(self,
                 bboxes1,
                 bboxes2,
                 mode='iou',
                 is_aligned=False,
                 version='oc'):
        """
        Args:
            bboxes1 (torch.Tensor): shape (..., 4|5|6), interpreted as
                <cx, cy, w, h[, a[, score]]>.  Only cx, cy, w, h are used.
            bboxes2 (torch.Tensor): same as bboxes1.
            mode (str): "iou", "iof", or "giou"
            is_aligned (bool): if True, expects bboxes1.numel == bboxes2.numel
        Returns:
            Tensor: IoU matrix of shape
                if is_aligned: original_shape_without_last_dim
                else: original_shape1_without_last_dim + (N2,)
        """
        # Use .numel() == 0 to check for empty tensors (like shape [0, 5])
        bboxes1_is_empty = (bboxes1 is None or bboxes1.numel() == 0)
        bboxes2_is_empty = (bboxes2 is None or bboxes2.numel() == 0)
        
        if bboxes1_is_empty or bboxes2_is_empty:
            # Try to get device and dtype from a valid tensor
            if not bboxes1_is_empty:
                tensor_ref = bboxes1
            elif not bboxes2_is_empty:
                tensor_ref = bboxes2
            else:
                # Both are empty/None, return a default float tensor on CPU
                # Or consider raising an error if caller should guarantee at least one input
                # Returning (0,) seems a safe default for reduction operations later.
                return torch.zeros((0,), dtype=torch.float32, device='cpu')

            # Determine the expected output shape for empty results
            if is_aligned:
                # Output shape is usually like input shape but without the last dim.
                # If inputs are empty, output should be a 1D empty tensor.
                output_shape = (0,)
            else:
                # Output shape is typically (..., N1, N2) -> (..., 0, N2) if bboxes1 is empty
                # or (..., N1, 0) if bboxes2 is empty.
                # We need N2 if bboxes1 is empty, or N1 if bboxes2 is empty.

                if bboxes1_is_empty:
                    # Need N2 from bboxes2 to determine shape (0, N2)
                    # Flatten bboxes2 to get total count N2 safely
                    tmp_bboxes2 = bboxes2.contiguous().view(-1, bboxes2.shape[-1])
                    N2 = tmp_bboxes2.size(0)
                    # Original leading dimensions of bboxes1 are unknown/irrelevant (size 0)
                    # Return shape (0, N2)
                    output_shape = (0, N2)
                else: # bboxes2 must be empty
                    # Need N1 from bboxes1 to determine shape (N1, 0)
                    orig_shape1 = list(bboxes1.shape)
                    tmp_bboxes1 = bboxes1.contiguous().view(-1, bboxes1.shape[-1])
                    N1 = tmp_bboxes1.size(0)
                    # Preserve original leading dimensions of bboxes1
                    output_shape = list(orig_shape1[:-1]) + [0] # e.g. (*batch_dims, N1, 0) -> reshaped later? No,overlaps is flat (N1,0) here
                    output_shape = (N1, 0) # The view logic later handles reshaping
                    
            # Create an empty tensor with the determined shape
            return torch.zeros(output_shape, dtype=tensor_ref.dtype, device=tensor_ref.device)
        
        # flatten all but last dim
        orig_shape1 = list(bboxes1.shape)
        orig_shape2 = list(bboxes2.shape)
        D1 = orig_shape1[-1]
        D2 = orig_shape2[-1]

        x1 = bboxes1.contiguous().view(-1, D1)
        x2 = bboxes2.contiguous().view(-1, D2)

        # drop score dim if present
        if x1.size(-1) == 6:
            x1 = x1[..., :5]
        if x2.size(-1) == 6:
            x2 = x2[..., :5]

        # now drop angle if present (keep only first 4 dims)
        x1 = x1[..., :4]
        x2 = x2[..., :4]

        # append zero-angle
        zeros1 = x1.new_zeros((x1.size(0), 1))
        zeros2 = x2.new_zeros((x2.size(0), 1))
        x1 = torch.cat([x1, zeros1], dim=-1)
        x2 = torch.cat([x2, zeros2], dim=-1)

        # compute rotated IoU (with zero-angle boxes ⇒ axis-aligned IoU)
        overlaps = box_iou_rotated(x1, x2, mode=mode)

        # reshape back
        if is_aligned:
            # returns a vector of length equal to num boxes in x1/x2
            return overlaps.view(*orig_shape1[:-1])
        else:
            # returns a (..., N2) tensor
            return overlaps.view(*orig_shape1[:-1], x2.size(0))

    def __repr__(self):
        return self.__class__.__name__ + '()'