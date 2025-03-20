import glob
import os
import pdb
import time
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.sscbench.helpers import (
    compute_CP_mega_matrix,
    compute_local_frustums,
    vox2pix,
)

from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS


@OPENOCC_DATASET.register_module()
class KITTI360Dataset(Dataset):

    def __init__(
        self,
        data_root=None,
        data_aug_conf=None,
        pipeline=None,
        phase="train",
        split=None,
        return_keys=[
            "img",
            "projection_mat",
            "image_wh",
            "lidar2cam",
            "ref2lidar",
            "occ_label",
            "occ_xyz",
            "occ_cam_mask",
            "cam_intrinsic",
            "sequence",
            "sample_idx",
            "monoscene",
        ],
    ):
        splits = {
            "train": [
                "2013_05_28_drive_0004_sync",
                "2013_05_28_drive_0000_sync",
                "2013_05_28_drive_0010_sync",
                "2013_05_28_drive_0002_sync",
                "2013_05_28_drive_0003_sync",
                "2013_05_28_drive_0005_sync",
                "2013_05_28_drive_0007_sync",
            ],
            "val": ["2013_05_28_drive_0006_sync"],
            "test": ["2013_05_28_drive_0009_sync"],
            "trainval": [
                "2013_05_28_drive_0004_sync",
                "2013_05_28_drive_0000_sync",
                "2013_05_28_drive_0010_sync",
                "2013_05_28_drive_0002_sync",
                "2013_05_28_drive_0003_sync",
                "2013_05_28_drive_0005_sync",
                "2013_05_28_drive_0007_sync",
                "2013_05_28_drive_0006_sync",
            ],
        }
        if split is None:
            split = phase
        self.split = split
        self.sequences = splits[split]

        self.data_path = self.root = data_root
        self.label_root = os.path.join(data_root, "preprocess", "labels")
        self.transxy = np.eye(4)
        self.custom_init()

        self.data_aug_conf = data_aug_conf
        self.test_mode = phase != "train"
        self.pipeline = []
        for t in pipeline:
            self.pipeline.append(OPENOCC_TRANSFORMS.build(t))

        self.return_keys = return_keys

    @staticmethod
    def read_calib():
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        P = np.array(
            [
                552.554261,
                0.000000,
                682.049453,
                0.000000,
                0.000000,
                552.554261,
                238.769549,
                0.000000,
                0.000000,
                0.000000,
                1.000000,
                0.000000,
            ]
        ).reshape(3, 4)

        cam2velo = np.array(
            [
                0.04307104361,
                -0.08829286498,
                0.995162929,
                0.8043914418,
                -0.999004371,
                0.007784614041,
                0.04392796942,
                0.2993489574,
                -0.01162548558,
                -0.9960641394,
                -0.08786966659,
                -0.1770225824,
            ]
        ).reshape(3, 4)
        C2V = np.concatenate([cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
        V2C = np.linalg.inv(C2V)
        V2C = V2C[:3, :]

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = P
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = V2C
        return calib_out

    def custom_init(self):
        start_time = time.time()
        self.keyframes = []
        self.frame2scan = {}

        for sequence in self.sequences:
            calib = self.read_calib()
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            glob_path = os.path.join(
                self.root, "data_2d_raw", sequence, "voxels", "*.bin"
            )
            seq_img_paths = glob.glob(glob_path)
            seq_img_paths = sorted(seq_img_paths)

            for seq_img_path in seq_img_paths:
                filename = os.path.basename(seq_img_path)
                frame_id = os.path.splitext(filename)[0]

                current_img_path = os.path.join(
                    self.root,
                    "data_2d_raw",
                    sequence,
                    "image_00/data_rect",
                    frame_id + ".png",
                )

                self.frame2scan.update(
                    {str(sequence) + "_" + frame_id: len(self.keyframes)}
                )

                self.keyframes.append(
                    {
                        "frame_id": frame_id,
                        "sequence": sequence,
                        "img_path": current_img_path,
                        "timestamp": int(frame_id),
                        "T_velo_2_cam": T_velo_2_cam,
                        "P": P,
                        "proj_matrix": proj_matrix,
                        "voxel_path": seq_img_path,
                    }
                )
            print(sequence)
        print("Preprocess time: --- %s seconds ---" % (time.time() - start_time))

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def __getitem__(self, index):
        info = deepcopy(self.keyframes[index])
        input_dict = self.get_data_info(info)

        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()
        for t in self.pipeline:
            input_dict = t(input_dict)

        return_dict = {
            k: input_dict[k] for k in self.return_keys if k in input_dict.keys()
        }
        return return_dict

    def get_data_info(self, info):
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = info["P"][:3, :3]
        lidar2img = intrinsic @ info["T_velo_2_cam"]

        input_dict = dict(
            token=info["frame_id"],
            sequence=info["sequence"],
            sample_idx=info["frame_id"],
            cam_intrinsic=np.expand_dims(intrinsic, axis=0),
            occ_path="",
            timestamp=info["timestamp"],
            img_filename=[info["img_path"]],
            pts_filename="",
            ego2lidar=np.eye(4),
            lidar2cam=np.expand_dims(info["T_velo_2_cam"], axis=0),
            lidar2img=np.expand_dims(lidar2img, axis=0),
            ego2img=np.expand_dims(lidar2img, axis=0),
        )

        return input_dict

    def __len__(self):
        return len(self.keyframes)
