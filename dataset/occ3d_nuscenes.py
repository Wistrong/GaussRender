import os
import pdb
from copy import deepcopy

import mmengine
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from .nuscenes import BaseNuscenes
from .utils import get_img2global, get_lidar2global


@OPENOCC_DATASET.register_module()
class Occ3dNuScenesDataset(BaseNuscenes):
    def __init__(
        self,
        data_root=None,
        imageset=None,
        data_aug_conf=None,
        pipeline=None,
        phase="train",
        is_mini=False,
        nusc=None,
    ):
        super().__init__(data_root, imageset, data_aug_conf, pipeline, phase)
        data = mmengine.load(imageset)
        self.scene_infos = data["infos"]
        if is_mini:
            self.scene_infos = self.sub_select_data(self.scene_infos, phase, 0.20)
        self.nusc = nusc

    def __getitem__(self, index):
        if self.overfit:
            index = 0
        info = deepcopy(self.scene_infos[index])
        input_dict = self.get_data_info(info)

        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()
        for t in self.pipeline:
            input_dict["nusc"] = self.nusc
            input_dict = t(input_dict)

        return_dict = {
            "img": input_dict["img"],
            "pts_filename": input_dict["pts_filename"],
            "cam_intrinsic": input_dict["cam_intrinsic"],
            "lidar2cam": input_dict["lidar2cam"],
            "projection_mat": input_dict["projection_mat"],
            "image_wh": input_dict["image_wh"],
            "occ_label": input_dict["occ_label"],
            "occ_xyz": input_dict["occ_xyz"],
            "occ_cam_mask": input_dict["occ_cam_mask"],
            "cam2ego": input_dict["cam2ego"],
            "ref2lidar": input_dict["ref2lidar"],
            "ego2lidar": input_dict["ego2lidar"],
            "idx_tag": "/".join(
                [info["scene_name"].split("-")[1], str(info["frame_idx"])]
            ),
            "token": input_dict["sample_idx"],
            "lidar_points": input_dict.get("lidar_points", None),
        }
        return return_dict

    def get_data_info(self, info):
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        img2lidar_rts = []
        cam_intrinsics = []
        cam2ego_rts = []
        ego2image_rts = []

        lidar2ego_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"]).T
        ego2lidar = np.linalg.inv(lidar2ego)

        for cam_type in self.sensor_types:
            info_cam = info["cams"][cam_type]
            image_paths.append(info_cam["data_path"])

            cam2ego = np.eye(4)
            cam2ego[:3, :3] = Quaternion(
                info_cam["sensor2ego_rotation"]
            ).rotation_matrix
            cam2ego[:3, 3] = np.array(info_cam["sensor2ego_translation"]).T

            ego2global = np.eye(4)
            ego2global[:3, :3] = Quaternion(
                info_cam["ego2global_rotation"]
            ).rotation_matrix
            ego2global[:3, 3] = np.array(info_cam["ego2global_translation"]).T

            cam2lidar = np.eye(4)
            cam2lidar[:3, :3] = info_cam["sensor2lidar_rotation"]
            cam2lidar[:3, 3] = np.array(info_cam["sensor2lidar_translation"]).T

            cam2img = np.asarray(info_cam["cam_intrinsic"])
            viewpad = np.eye(4)
            viewpad[:3, :3] = cam2img

            cam2global = ego2global @ cam2ego
            lidar2img = viewpad @ np.linalg.inv(cam2lidar)
            lidar2cam = np.linalg.inv(cam2lidar)
            img2lidar = np.linalg.inv(lidar2img)
            ego2img = viewpad @ np.linalg.inv(cam2ego)

            lidar2img_rts.append(lidar2img)
            lidar2cam_rts.append(lidar2cam)
            img2lidar_rts.append(img2lidar)
            cam_intrinsics.append(viewpad)
            cam2ego_rts.append(cam2ego)
            ego2image_rts.append(ego2img)

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"]).T

        lidar2global = ego2global @ lidar2ego

        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            occ_path=info["occ_path"],
            timestamp=info["timestamp"] / 1e6,
            ego2global=ego2global,
            lidar2global=lidar2global,
            img_filename=image_paths,
            lidar2img=np.asarray(lidar2img_rts),
            lidar2cam=np.asarray(lidar2cam_rts),
            img2lidar=np.asarray(img2lidar_rts),
            cam_intrinsic=np.asarray(cam_intrinsics),
            ego2lidar=ego2lidar,
            cam2ego=np.asarray(cam2ego_rts),
            ego2img=np.asarray(ego2image_rts),
            token=info["token"],
        )

        return input_dict

    def __len__(self):
        return len(self.scene_infos)
