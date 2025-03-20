from mmengine.registry import Registry

OPENOCC_DATASET = Registry("openocc_dataset")
OPENOCC_DATAWRAPPER = Registry("openocc_datawrapper")
OPENOCC_TRANSFORMS = Registry("openocc_transforms")

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .kitti_360 import KITTI360Dataset
from .occ3d_nuscenes import Occ3dNuScenesDataset
from .pipeline import *
from .surr_nuscenes import SurrNuScenesDataset
from .utils import CustomDistributedSampler, custom_collate_fn_temporal


def get_dataloader(
    train_dataset_config,
    val_dataset_config,
    train_loader,
    val_loader,
    dist=False,
    iter_resume=False,
    train_sampler_config=dict(shuffle=True, drop_last=True),
    val_sampler_config=dict(shuffle=False, drop_last=False),
    val_only=False,
):
    if val_only:
        val_wrapper = OPENOCC_DATASET.build(val_dataset_config)

        val_sampler = None
        if dist:
            val_sampler = DistributedSampler(val_wrapper, **val_sampler_config)

        val_dataset_loader = DataLoader(
            dataset=val_wrapper,
            batch_size=val_loader["batch_size"],
            collate_fn=custom_collate_fn_temporal,
            shuffle=False,
            sampler=val_sampler,
            num_workers=val_loader["num_workers"],
            pin_memory=True,
        )

        return None, val_dataset_loader

    train_wrapper = OPENOCC_DATASET.build(train_dataset_config)
    val_wrapper = OPENOCC_DATASET.build(val_dataset_config)

    train_sampler = val_sampler = None
    if dist:
        if iter_resume:
            train_sampler = CustomDistributedSampler(
                train_wrapper, **train_sampler_config
            )
        else:
            train_sampler = DistributedSampler(train_wrapper, **train_sampler_config)
        val_sampler = DistributedSampler(val_wrapper, **val_sampler_config)

    train_dataset_loader = DataLoader(
        dataset=train_wrapper,
        batch_size=train_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        shuffle=False if dist else train_loader["shuffle"],
        sampler=train_sampler,
        num_workers=train_loader["num_workers"],
        pin_memory=True,
    )
    val_dataset_loader = DataLoader(
        dataset=val_wrapper,
        batch_size=val_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        shuffle=False,
        sampler=val_sampler,
        num_workers=val_loader["num_workers"],
        pin_memory=True,
    )

    return train_dataset_loader, val_dataset_loader


def get_dataset(
    train_dataset_config,
    val_dataset_config,
):
    train_wrapper = OPENOCC_DATASET.build(train_dataset_config)
    val_wrapper = OPENOCC_DATASET.build(val_dataset_config)
    return train_wrapper, val_wrapper
