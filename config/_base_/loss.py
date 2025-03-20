print_freq = 250
work_dir = None
max_epochs = 20
inspect = False

# ================== training ========================
lr = 2e-4
optimizer = dict(
    optimizer=dict(
        type="AdamW",
        lr=lr,
        weight_decay=1e-4,
    ),
    paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.1)}),
)

# =================== Loss ==========================
loss = dict(
    type="MultiLoss",
    w_occupancy=True,
    loss_cfgs=[
        dict(
            type="OccupancyLoss",
            weight=1.0,
            num_classes=None,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0, loss_voxel_lovasz_weight=1.0
            ),
            manual_class_weight=None,
        )
    ],
)

loss_input_convertion = dict(pred_occ="pred_occ")
