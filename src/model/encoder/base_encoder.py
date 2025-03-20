from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class BaseEncoder(BaseModule):
    """Further encode 3D representations.
    image backbone -> neck -> lifter -> encoder -> segmentor
    """

    def __init__(self, init_cfg=None, **kwargs):
        super().__init__(init_cfg)

    def forward(self, representation, ms_img_feats=None, metas=None, **kwargs):
        pass
