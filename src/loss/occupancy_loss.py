import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_gauss import GaussianRasterizationSettings
from diff_gauss import GaussianRasterizer as GaussianRasterizer3D
from torch.cuda.amp import autocast

from . import OPENOCC_LOSS
from .base_loss import BaseLoss
from .utils.lovasz_softmax import lovasz_softmax


@OPENOCC_LOSS.register_module()
class RenderLoss(BaseLoss):
    def __init__(
        self,
        weight=1.0,
        mask_elems=False,
        kl_loss=False,
    ):
        super().__init__()
        self.input_dict = {
            "render": "render",
            "render_cam": "render_cam",
            "render_cam_depth": "render_cam_depth",
            "render_bev": "render_bev",
            "render_bev_depth": "render_bev_depth",
        }
        self.weight = weight
        self.mask_elems = mask_elems
        self.loss_func = self.loss_render
        self.kl_loss = kl_loss
        return

    def loss_render(
        self,
        render,
        render_cam=None,
        render_cam_depth=None,
        render_bev=None,
        render_bev_depth=None,
    ):
        render_pred_img = render.get("cam", None)
        render_pred_depth = render.get("depth", None)
        render_pred_bev = render.get("bev", None)
        render_pred_bev_depth = render.get("bev_depth", None)
        device = render_pred_img.device

        B, Ncams = render_pred_img.shape[:2]
        dict_loss = {}
        loss_img, loss_depth, loss_bev, loss_bev_depth = 0, 0, 0, 0

        for i in range(B):
            render_gt_bev = render_bev[i]
            render_gt_bev_depth = render_bev_depth[i]
            for j in range(Ncams):
                render_gt_img = render_cam[i][j]
                render_gt_depth = render_cam_depth[i][j]

                if render_pred_img is not None:
                    res = render_pred_img[i, j].flatten(0, 1)
                    gt = render_gt_img.flatten(0, 1)

                    if self.mask_elems:
                        # Merge SurroundOcc and Occ3D for simplicity.
                        mask = (gt.argmax(-1) == 0) & (gt.argmax(-1) == 17)
                        gt = gt[mask]
                        res = res[mask]

                    if self.kl_loss:
                        loss_img_j = (
                            F.kl_div(
                                torch.log_softmax(res, 1),
                                torch.log_softmax(gt, 1),
                                reduction="mean",
                                log_target=True,
                            )
                            * 10
                        )
                    else:
                        loss_img_j = F.l1_loss(res, gt, reduction="mean")
                    loss_img += loss_img_j

                if render_pred_depth is not None:
                    gt = render_gt_depth
                    res = render_pred_depth[i][j]
                    mask_depth = gt > 0.1
                    if self.mask_elems:
                        mask_depth = mask_depth & mask

                    loss_depth_j = F.l1_loss(res[mask_depth] / 50, gt[mask_depth] / 50)
                    loss_depth += loss_depth_j

            if render_pred_bev is not None:
                res = render_pred_bev[i]
                gt = render_gt_bev

                if self.mask_elems:
                    # Merge SurroundOcc and Occ3D for simplicity.
                    mask = (gt.argmax(-1) == 0) & (gt.argmax(-1) == 17)
                    gt = gt[mask]
                    res = res[mask]

                if self.kl_loss:
                    loss_bev_i = (
                        F.kl_div(
                            torch.log_softmax(res, 1),
                            torch.log_softmax(gt, 1),
                            reduction="mean",
                            log_target=True,
                        )
                        * 10
                    )
                else:
                    loss_bev_i = F.l1_loss(res, gt, reduction="mean")
                loss_bev += loss_bev_i

            if render_pred_bev_depth is not None:
                gt = render_gt_bev_depth
                res = render_pred_bev_depth[i]
                mask_depth = gt > 0.1
                if self.mask_elems:
                    mask_depth = mask_depth & mask

                loss_bev_depth_j = F.l1_loss(
                    (res[mask_depth]) / 100, (gt[mask_depth]) / 100
                )
                loss_bev_depth += loss_bev_depth_j

        if render_pred_img is not None:
            if not torch.isnan(loss_img).any():
                add = loss_img / Ncams
            else:
                add = torch.tensor(0.0, device=device)
            dict_loss.update({"img": add})
        if render_pred_depth is not None:
            if not torch.isnan(loss_depth).any():
                add = loss_depth / Ncams
            else:
                add = torch.tensor(0.0, device=device)
            dict_loss.update({"depth": add})
        if render_pred_bev is not None:
            if not torch.isnan(loss_bev).any():
                add = loss_bev
            else:
                add = torch.tensor(0.0, device=device)
            dict_loss.update({"bev": add})
        if render_pred_bev_depth is not None:
            if not torch.isnan(loss_bev_depth).any():
                add = loss_bev_depth
            else:
                add = torch.tensor(0.0, device=device)
            dict_loss.update({"bev_depth": add})
        return dict_loss


@OPENOCC_LOSS.register_module()
class OccupancyLoss(BaseLoss):

    def __init__(
        self,
        weight,
        num_classes,
        multi_loss_weights=dict(),
        manual_class_weight=None,
    ):

        super().__init__()

        self.weight = weight
        self.input_dict = {
            "pred_occ": "pred_occ",
            "occ_label": "occ_label",
            "occ_mask": "occ_mask",
            "out_occ_shapes": "out_occ_shapes",
        }
        self.loss_func = self.loss_voxel

        self.num_classes = num_classes
        self.classes = list(range(num_classes))

        self.loss_voxel_ce_weight = multi_loss_weights.get("loss_voxel_ce_weight", 1.0)
        self.loss_voxel_lovasz_weight = multi_loss_weights.get(
            "loss_voxel_lovasz_weight", 1.0
        )

        self.class_weights = manual_class_weight

    def loss_voxel(self, pred_occ, occ_label, occ_mask=None, out_occ_shapes=None):
        dict_scale = {}
        # Multi-scale
        for scale in range(len(pred_occ)):
            ratio = out_occ_shapes[-1][0] // out_occ_shapes[scale][0]

            occ_pred_s = pred_occ[scale]
            occ_label_s = multiscale_supervision(occ_label.float(), ratio).flatten(
                start_dim=1
            )
            occ_mask_s = multiscale_supervision(occ_mask, ratio).flatten(start_dim=1)
            occ_mask_s[..., 0] = 1  # At least one to backpropagate

            if occ_mask_s is not None:
                occ_mask_s = occ_mask_s.flatten(1)
                occ_pred_s = occ_pred_s.transpose(1, 2)[occ_mask_s][None].transpose(
                    1, 2
                )
                occ_label_s = occ_label_s[occ_mask_s][None]

            loss_dict = {}
            # No Focal Loss
            loss_dict["loss_voxel_ce"] = self.loss_voxel_ce_weight * CE_ssc_loss(
                occ_pred_s,
                occ_label_s,
                self.class_weights.type_as(occ_pred_s),
                ignore_index=255,
            )

            # Lovasz Loss
            loss_dict["loss_voxel_lovasz"] = (
                self.loss_voxel_lovasz_weight
                * lovasz_softmax(
                    torch.softmax(occ_pred_s, dim=1).transpose(1, 2).flatten(0, 1),
                    occ_label_s.flatten(),
                    ignore=255,
                )
            )

            loss_scale = 0.0
            for k, v in loss_dict.items():
                if v.numel() > 0:
                    loss_scale = loss_scale + v
            loss_scale = loss_scale * 0.5 ** (len(pred_occ) - 1 - scale)
            dict_scale[f"scale_{len(pred_occ) - 1 - scale}"] = loss_scale
        return dict_scale


def inverse_sigmoid(x, sign="A"):
    x = x.to(torch.float32)
    while x >= 1 - 1e-5:
        x = x - 1e-5

    while x < 1e-5:
        x = x + 1e-5

    return -torch.log((1 / x) - 1)


# Function to perform 3D majority pooling
def multiscale_supervision(labels, ratio):
    # Not exactly the pooling implemented in Surroundocc.
    B, H, W, Z = labels.shape

    if ratio == 1:
        return labels
    device = labels.device

    meshgrid = torch.stack(
        torch.meshgrid(
            torch.arange(H), torch.arange(W), torch.arange(Z), indexing="ij"
        ),
        dim=-1,
    )
    coords = (meshgrid // ratio).flatten(0, -2).long()
    labels = labels.flatten(start_dim=1)

    new_labels = torch.zeros(
        (B, H // ratio, W // ratio, Z // ratio), device=device, dtype=labels.dtype
    )
    for i in range(B):
        new_labels[i, coords[:, 0], coords[:, 1], coords[:, 2]] = labels[i]
    return new_labels


def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    # from IPython import embed
    # embed()
    # exit()
    with autocast(False):
        loss = criterion(pred, target.long())

    return loss


from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from mmcv.ops import softmax_focal_loss as _softmax_focal_loss
from mmdet.models.losses.utils import weight_reduce_loss


# This method is only for debugging
def py_sigmoid_focal_loss(
    pred, target, weight=None, gamma=2.0, alpha=0.25, reduction="mean", avg_factor=None
):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = (
        F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        * focal_weight
    )
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight

    loss = loss.sum(-1).mean()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_focal_loss_with_prob(
    pred, target, weight=None, gamma=2.0, alpha=0.25, reduction="mean", avg_factor=None
):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.
    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    num_classes = pred.size(1)
    target = F.one_hot(target, num_classes=num_classes + 1)
    target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(pred, target, reduction="none") * focal_weight

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(
    pred,
    target,
    weight=None,
    cls_weight=None,
    gamma=2.0,
    alpha=0.25,
    reduction="mean",
    avg_factor=None,
):
    r"""A wrapper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(
        pred.contiguous(), target.contiguous(), gamma, alpha, cls_weight, "none"
    )
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight
    loss = loss.sum(-1).mean()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def softmax_focal_loss(
    pred,
    target,
    weight=None,
    cls_weight=None,
    gamma=2.0,
    alpha=0.25,
    reduction="mean",
    avg_factor=None,
):
    r"""A wrapper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _softmax_focal_loss(
        pred.contiguous(), target.contiguous(), gamma, alpha, cls_weight, "none"
    )
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight
    loss = loss.mean()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class CustomFocalLoss(nn.Module):

    def __init__(
        self,
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        reduction="mean",
        loss_weight=1.0,
        activated=False,
    ):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(CustomFocalLoss, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(
        self,
        pred,
        target,
        target_xyz,
        weight=None,
        avg_factor=None,
        ignore_index=255,
        reduction_override=None,
    ):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        target_xy = target_xyz[..., :2]  # b, n, 2
        target_dist = torch.norm(target_xy, 2, -1)  # b, n
        target_dist_max = target_dist.max()
        c = target_dist / target_dist_max + 1  # b, n
        c = c.flatten()

        # weight_mask = weight[None, None, :] * c[..., None] # b, n, c
        # weight_mask = weight_mask.flatten(0, 1)

        pred = pred.transpose(1, 2).flatten(0, 1)  # BN, C
        target = target.flatten(0, 1)

        num_classes = pred.size(1)

        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    assert False
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss
        else:
            calculate_loss_func = softmax_focal_loss

        loss_cls = self.loss_weight * calculate_loss_func(
            pred,
            target.to(torch.long),
            c,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor,
        )

        return loss_cls
