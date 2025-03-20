_base_ = ["data_select.py"]
model_tag = "tpvformer"

# ========= model config ===============
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
_num_cams_ = 6

num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
num_classes = _base_.num_classes

point_cloud_range = _base_.point_cloud_range
voxel_size = _base_.voxel_size

find_unused_parameters = False
load_from = "ckpts/r101_dcn_fcos3d_pretrain.pth"

xmin, ymin, zmin, xmax, ymax, zmax = _base_.point_cloud_range
vx = _base_.voxel_size
out_occ_shapes = [
    [int((xmax - xmin) / vx), int((ymax - ymin) / vx), int((zmax - zmin) / vx)]
]

scale_h = 2
scale_w = 2
scale_z = 2
tpv_h_ = out_occ_shapes[0][0] // scale_h
tpv_w_ = out_occ_shapes[0][1] // scale_w
tpv_z_ = out_occ_shapes[0][2] // scale_z
grid_size = [tpv_h_ * scale_h, tpv_w_ * scale_w, tpv_z_ * scale_z]


img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
_base_.train_pipeline[4] = dict(type="NormalizeMultiviewImage", **img_norm_cfg)
_base_.val_pipeline[3] = dict(type="NormalizeMultiviewImage", **img_norm_cfg)
_base_.test_pipeline[3] = dict(type="NormalizeMultiviewImage", **img_norm_cfg)
train_dataset_config = dict(pipeline=_base_.train_pipeline)
val_dataset_config = dict(pipeline=_base_.val_pipeline)
test_dataset_config = dict(pipeline=_base_.test_pipeline)


model = dict(
    type="BEVSegmentor",
    pc_range=point_cloud_range,
    voxel_size=voxel_size,
    use_grid_mask=True,
    img_backbone_out_indices=[1, 2, 3],
    img_backbone=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    head=dict(
        type="TPVFormerHead",
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        num_feature_levels=_num_levels_,
        num_cams=_num_cams_,
        embed_dims=_dim_,
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=tpv_h_,
            col_num_embed=tpv_w_,
        ),
        encoder=dict(
            type="TPVFormerEncoder",
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_layers=3,
            pc_range=point_cloud_range,
            num_points_in_pillar=num_points_in_pillar,
            return_intermediate=False,
            transformerlayers=dict(
                type="TPVFormerLayer",
                attn_cfgs=[
                    dict(
                        type="TPVCrossViewHybridAttention",
                        embed_dims=_dim_,
                        num_levels=1,
                    ),
                    dict(
                        type="TPVImageCrossAttention",
                        deformable_attention=dict(
                            type="TPVMSDeformableAttention3D",
                            embed_dims=_dim_,
                            num_points=num_points,
                            num_z_anchors=num_points_in_pillar,
                            num_levels=_num_levels_,
                            floor_sampling_offset=False,
                            tpv_h=tpv_h_,
                            tpv_w=tpv_w_,
                            tpv_z=tpv_z_,
                        ),
                        embed_dims=_dim_,
                        tpv_h=tpv_h_,
                        tpv_w=tpv_w_,
                        tpv_z=tpv_z_,
                    ),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=(
                    "self_attn",
                    "norm",
                    "cross_attn",
                    "norm",
                    "ffn",
                    "norm",
                ),
            ),
        ),
    ),
    aggregator=dict(
        type="TPVAggregator",
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        nbr_classes=num_classes,
        occ_shape=out_occ_shapes[-1],
        voxel_size=voxel_size,
        in_dims=_dim_,
        hidden_dims=2 * _dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z,
    ),
)
