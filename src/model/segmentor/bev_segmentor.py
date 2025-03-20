import pdb

import torch
from mmseg.models import SEGMENTORS, build_backbone

from .base_segmentor import CustomBaseSegmentor


@SEGMENTORS.register_module()
class BEVSegmentor(CustomBaseSegmentor):

    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        voxel_size=None,
        pc_range=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)
        self.voxel_size = voxel_size
        self.pc_range = pc_range

        if hasattr(self, "aggregator") and (hasattr(self.aggregator, "renderer")):
            self.renderer = self.aggregator.renderer
        if hasattr(self, "head") and (hasattr(self.head, "renderer")):
            self.renderer = self.head.renderer

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        assert len(img_feats_backbone) == len(self.img_backbone_out_indices)
        if hasattr(self, "img_neck"):
            img_feats = self.img_neck(img_feats_backbone)
        else:
            img_feats = img_feats_backbone

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return {"ms_img_feats": img_feats_reshaped}

    def forward_extra_img_backbone(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)

        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W)
            )
        return img_feats_backbone_reshaped

    def forward(
        self,
        imgs=None,
        metas=None,
        points=None,
        **kwargs,
    ):
        """Forward training function."""
        results = {
            "imgs": imgs,
            "metas": metas,
            "points": points,
            "voxel_size": self.voxel_size,
            "pc_range": self.pc_range,
        }
        results.update(kwargs)
        # Backbone
        outs = self.extract_img_feat(**results)
        results.update(outs)

        # Lifter
        if hasattr(self, "lifter"):
            outs = self.lifter(**results)
            results.update(outs)

        # Encoder
        if hasattr(self, "encoder"):
            outs = self.encoder(**results)
            results.update(outs)

        # Head
        outs = self.head(**results)
        results.update(outs)

        # Eventually aggregator: tpvformer
        if hasattr(self, "aggregator"):
            outs = self.aggregator(**results)
            results.update(outs)
        return results
