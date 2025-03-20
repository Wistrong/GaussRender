_base_ = [
    "../_base_/loss_render.py",
    "../_base_/model_tpvformer.py",
]

model = dict(
    aggregator=dict(
        render=True,
        pre_render_kwargs=dict(
            mask_elements=False,
            transfer_colors=True,
            transfer_opacity=False,
            overwrite_scales=True,
            overwrite_rotations=False,
            overwrite_opacity=False,
            dataset_tag=_base_.dataset_tag,
        ),
        render_kwargs=dict(
            with_cam_rendering=True,
            with_bev_rendering=True,
            with_depth_rendering=True,
            with_bev_depth_rendering=True,
            inspect=False,
            pc_range=_base_.model.get("pc_range"),
            voxel_size=_base_.model.get("voxel_size"),
            render_ncam=None,
            num_classes=_base_.num_classes,
            dataset_tag=_base_.dataset_tag,
            render_gt_mode=None,
            cam_idx=None,
            gaussian_scale=None,
        ),
    )
)
