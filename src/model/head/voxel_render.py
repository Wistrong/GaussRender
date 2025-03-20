import os
import pdb
import random
import sys

import numpy as np
import torch
from einops import rearrange, repeat
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from torch import nn

from ..utils.gaussian_utils import GaussianPrediction


@MODELS.register_module()
class Decoder(BaseModule):
    def __init__(
        self, in_dim, segm_dim, scaling_dim, rotation_dim, opacity_dim, num_layer=2
    ):
        super().__init__()

        self.segm_dim = segm_dim
        self.opacity_dim = opacity_dim
        self.scaling_dim = scaling_dim
        self.rotation_dim = rotation_dim
        self.out_dim = 3 + segm_dim + opacity_dim + scaling_dim + rotation_dim

        layers_coarse = (
            [nn.Linear(in_dim, in_dim), nn.ReLU()]
            + [nn.Linear(in_dim, in_dim), nn.ReLU()] * (num_layer - 1)
            + [nn.Linear(in_dim, self.out_dim)]
        )
        self.mlp_coarse = nn.Sequential(*layers_coarse)
        self.init(self.mlp_coarse)

    def init(self, layers):
        init_method = "xavier"
        if init_method:
            for layer in layers:
                if not isinstance(layer, torch.nn.Linear):
                    continue
                if init_method == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                elif init_method == "xavier":
                    torch.nn.init.xavier_uniform_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)

    def forward(self, feats, opacity_shift, scaling_shift):
        parameters = self.mlp_coarse(feats).float()
        B, _, C = parameters.shape
        parameters = parameters.view(B, -1, 1, C).flatten(1, 2)
        offset, colors_precomp, opacity, scaling, rotation = torch.split(
            parameters,
            [3, self.segm_dim, self.opacity_dim, self.scaling_dim, self.rotation_dim],
            dim=-1,
        )
        opacity = opacity + opacity_shift
        scaling = scaling + scaling_shift
        offset = torch.sigmoid(offset) * 2 - 1.0

        B = opacity.shape[0]
        colors_precomp = colors_precomp.view(B, -1, self.segm_dim)
        opacity = opacity.view(B, -1, self.opacity_dim)
        scaling = scaling.view(B, -1, self.scaling_dim)
        rotation = rotation.view(B, -1, self.rotation_dim)
        offset = offset.view(B, -1, 3)
        return offset, colors_precomp, scaling, rotation, opacity


@MODELS.register_module()
class Voxel2Gaussians(BaseModule):
    def __init__(
        self,
        occsize,
        voxel_size,
        segm_dim,
        in_c,
        mask_elements,
        transfer_colors,
        transfer_opacity,
        overwrite_scales,
        overwrite_rotations,
        overwrite_opacity,
        decoder_kwargs={"num_layer": 2},
        dataset_tag=None,
        gaussian_scale=None,
    ):
        super().__init__()
        # 2DGS model
        self.scaling_dim, self.rotation_dim = 2, 4
        self.scaling_dim = 3
        self.opacity_dim = 1
        self.out_dim = (
            segm_dim + self.scaling_dim + self.rotation_dim + self.opacity_dim
        )

        self.decoder = Decoder(
            in_c,
            segm_dim,
            self.scaling_dim,
            self.rotation_dim,
            self.opacity_dim,
            **decoder_kwargs,
        )

        # parameters initialization
        self.opacity_shift = -1.0
        self.scaling_shift = -1.0

        self.occsize = occsize

        self.mask_elements = mask_elements
        self.transfer_colors = transfer_colors
        self.transfer_opacity = transfer_opacity
        self.overwrite_scales = overwrite_scales
        self.overwrite_rotations = overwrite_rotations
        self.overwrite_opacity = overwrite_opacity
        self.voxel_size = voxel_size
        self.dataset_tag = dataset_tag

        self.gaussian_scale = gaussian_scale
        return

    def get_offseted_pt(self, occ_xyz, offset, mask=None):
        """Offset is in [-1,1]."""
        B = offset.shape[0]
        half_cell_size = 0.5 * self.voxel_size

        group_centers = occ_xyz
        if mask is not None:
            centers = group_centers.squeeze(0)[mask] + offset * half_cell_size
        else:
            centers = (
                group_centers.unsqueeze(-2).expand(B, -1, 1, -1).reshape(offset.shape)
                + offset * half_cell_size
            )
        return centers

    def forward(self, voxel_feats, metas, occ_pred=None):
        """Given a volume feature, estimate gaussians and render them in 2D images."""
        num_classes = occ_pred.shape[1]

        B = occ_pred.shape[0]
        device = occ_pred.device
        dtype = occ_pred.dtype

        # Remove empty voxels
        if self.mask_elements:
            # Remove 0,17,255.
            argmx = occ_pred.argmax(1)
            # SurroundOcc
            if self.dataset_tag in ["surroundocc", "kitti360"]:
                empty_cells = (argmx != 0) & (argmx != 255)
            # Occ3D
            elif self.dataset_tag == "occ3d":
                empty_cells = (argmx != 17) & (argmx != 255)
            else:
                return NotImplementedError(
                    f"Unsupported dataset_tag: {self.dataset_tag}"
                )
            mask = rearrange(empty_cells, "b x y z -> (b x y z)")
        else:
            mask = torch.ones_like(occ_pred[:, 0].flatten()).bool()

        if voxel_feats is not None:
            voxel_feats = rearrange(voxel_feats, "b c x y z -> b (x y z) c")
            voxel_feats = voxel_feats.flatten(0, 1)[mask].unsqueeze(0)

            (
                offsets,
                semantics,
                scaling,
                rotations,
                opacity,
            ) = self.decoder(
                voxel_feats, self.opacity_shift, self.scaling_shift
            )  # (B,Npts,in_c) -> (B,Npts*K,out_c)
            opacity = torch.sigmoid(opacity)
            rotations = torch.nn.functional.normalize(rotations, dim=-1)
            scaling = torch.nn.functional.sigmoid(scaling) * 0.63 + 0.01
        else:
            B, C, X, Y, Z = occ_pred.shape
            offsets = torch.zeros((B, X * Y * Z, 3), device=device, dtype=dtype)
            semantics = None  # We will transfer colors
            scaling = (
                torch.ones((B, X * Y * Z, 3), device=device, dtype=dtype)
                * self.gaussian_scale
            )
            rotations = torch.zeros((B, X * Y * Z, 4), device=device, dtype=dtype)
            rotations[:, 0] = 1
            opacity = torch.ones((B, X * Y * Z), device=device, dtype=dtype)

        if self.transfer_colors or voxel_feats is None:
            semantics = rearrange(occ_pred, "b c x y z -> (b x y z) c")[mask]
            semantics = semantics.unsqueeze(0)

        if self.transfer_opacity or voxel_feats is None:
            out_soft = occ_pred.softmax(1)
            # SurroundOcc
            if self.dataset_tag in ["surroundocc", "kitti360"]:
                out_soft = out_soft[:, 0:1]
            # Occ3D
            elif self.dataset_tag == "occ3d":
                out_soft = out_soft[:, 17:18]
            else:
                raise NotImplementedError
            sharpen_factor = 2  # Adjust as needed
            out_soft = torch.where(
                out_soft < 0.5,
                out_soft**sharpen_factor,
                1 - (1 - out_soft) ** sharpen_factor,
            )

            opacity = rearrange(
                rearrange(1 - out_soft, "b 1 x y z -> (b x y z) 1")[mask].unsqueeze(0),
            )

        if self.overwrite_opacity:
            opacity = torch.ones_like(opacity)
            # SurroundOcc
            if self.dataset_tag in ["surroundocc", "kitti360"]:
                index_empty = torch.where(semantics.argmax(-1) == 0, True, False)
            # Occ3D
            elif self.dataset_tag == "occ3d":
                index_empty = torch.where(semantics.argmax(-1) == 17, True, False)
            else:
                raise NotImplementedError(
                    f"Unsupported dataset_tag: {self.dataset_tag}"
                )
            opacity[index_empty] = 0.0
            opacity[~index_empty] = 1.0

        # Merge all semantics
        g = semantics.shape[1]
        new_semantics = torch.full((B, g, 19), -10_000, device=device, dtype=dtype)
        new_semantics[:, :, :num_classes] = semantics
        semantics = new_semantics

        # Prevent the rendering of empty classes: done a priori.
        if self.mask_elements:
            # SurroundOcc & Kitti360: empty class is 0.
            if self.dataset_tag in ["surroundocc", "kitti360"]:
                new_semantics[..., 0] = -10_000
            # Occ3D: empty class is 17.
            elif self.dataset_tag == "occ3d":
                new_semantics[..., 17] = -10_000

        # convert to local positions
        offsets = torch.zeros_like(offsets)
        means = self.get_offseted_pt(metas["occ_xyz"].flatten(1, 3), offsets, mask)

        if self.overwrite_scales:
            scaling = torch.ones_like(scaling) * self.gaussian_scale

        if self.overwrite_rotations:
            rotations = torch.zeros_like(rotations)
            rotations[..., 0] = 1.0

        return GaussianPrediction(
            means=means,
            semantics=semantics,
            opacities=opacity,
            scales=scaling,
            rotations=rotations,
        )
