_base_ = ["./path.py"]
data_root = f"{_base_.folderpath}/data/nuscenes/"
occ_path = f"{_base_.folderpath}/data/surroundocc_nuscenes/samples"

batch_size = 1

num_classes = 17
dataset_tag = "surroundocc"


point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
voxel_size = 0.5

train_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",
        to_float32=True,
    ),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path),
    dict(type="ResizeCropFlipImage"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage"),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

val_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",
        to_float32=True,
    ),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage"),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

data_aug_conf = dict(
    resize_lim=(1.0, 1.0),
    final_dim=(896, 1600),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=False,
)

train_dataset_config = dict(
    type="SurrNuScenesDataset",
    data_root=data_root,
    imageset=f"{_base_.folderpath}/data/nuscenes_infos_train_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    is_mini=False,
    phase="train",
)
train_loader = dict(batch_size=batch_size, num_workers=8, shuffle=True)

val_dataset_config = dict(
    type="SurrNuScenesDataset",
    data_root=data_root,
    imageset=f"{_base_.folderpath}/data/nuscenes_infos_val_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=val_pipeline,
    is_mini=False,
    phase="val",
)
val_loader = dict(batch_size=batch_size, num_workers=8, shuffle=False)

test_dataset_config = val_dataset_config.copy()
test_pipeline = val_pipeline.copy()
test_loader = val_loader.copy()
