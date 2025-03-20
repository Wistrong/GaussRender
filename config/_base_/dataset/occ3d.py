_base_ = ["./surroundocc.py"]
occ_path = f"{_base_.folderpath}/data/gts/"

num_classes = 18
dataset_tag = "occ3d"

point_cloud_range = [-40, -40, -1, 40, 40, 5.4]
voxel_size = 0.4

_base_.train_pipeline[1]["type"] = "LoadOccupancyOcc3dNuscenes"
_base_.val_pipeline[1]["type"] = "LoadOccupancyOcc3dNuscenes"
_base_.train_pipeline[1]["occ_path"] = occ_path
_base_.val_pipeline[1]["occ_path"] = occ_path

_base_.train_pipeline[-1]["use_ego"] = True
_base_.val_pipeline[-1]["use_ego"] = True

train_dataset_config = dict(
    type="Occ3dNuScenesDataset",
    imageset=f"{_base_.folderpath}/data/bevdetv2-nuscenes_infos_train.pkl",
    pipeline=_base_.train_pipeline,
)

val_dataset_config = dict(
    type="Occ3dNuScenesDataset",
    imageset=f"{_base_.folderpath}/data/bevdetv2-nuscenes_infos_val.pkl",
    pipeline=_base_.val_pipeline,
)

test_dataset_config = val_dataset_config.copy()
test_pipeline = _base_.val_pipeline.copy()
test_loader = _base_.val_loader.copy()
