_base_ = ["./path.py"]
data_root = f"{_base_.folderpath}/data/sscbench-kitti"
occ_path = f"{_base_.folderpath}/data/sscbench-kitti/preprocess/labels"
batch_size = 1

num_classes = 19
dataset_tag = "kitti360"
point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
voxel_size = 0.2

data_aug_conf = dict(
    resize_lim=(1.0, 1.0),
    final_dim=(376, 1408),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=376,
    W=1408,
    rand_flip=False,
)

train_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",
        to_float32=True,
    ),
    dict(type="LoadOccupancyKITTI360", occ_path=occ_path),
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
    dict(type="LoadOccupancyKITTI360", occ_path=occ_path),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage"),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

train_dataset_config = dict(
    type="KITTI360Dataset",
    data_root=data_root,
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase="train",
)
train_loader = dict(batch_size=batch_size, num_workers=8, shuffle=True)

val_dataset_config = dict(
    type="KITTI360Dataset",
    data_root=data_root,
    data_aug_conf=data_aug_conf,
    pipeline=val_pipeline,
    phase="val",
)
val_loader = dict(batch_size=batch_size, num_workers=8, shuffle=False)

test_dataset_config = val_dataset_config.copy()
test_pipeline = val_pipeline.copy()
test_dataset_config["phase"] = "test"
test_loader = val_loader.copy()
