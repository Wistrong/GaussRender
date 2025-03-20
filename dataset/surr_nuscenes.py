import os
from copy import deepcopy

import mmengine
import numpy as np
from pyquaternion import Quaternion

from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from .nuscenes import BaseNuscenes
from .utils import get_img2global, get_lidar2global


@OPENOCC_DATASET.register_module()
class SurrNuScenesDataset(BaseNuscenes):
    def __init__(
        self,
        data_root=None,
        imageset=None,
        data_aug_conf=None,
        pipeline=None,
        phase="train",
        is_mini=False,
    ):
        super().__init__(data_root, imageset, data_aug_conf, pipeline, phase)
        data = mmengine.load(imageset)
        self.scene_infos = data["infos"]
        self.keyframes = data["metadata"]
        if is_mini:
            self.keyframes = self.sub_select_data(self.keyframes, phase, 0.20)
        self.keyframes = sorted(
            self.keyframes, key=lambda x: x[0] + "{:0>3}".format(str(x[1]))
        )

    def get_data_info(self, info):
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        img2lidar_rts = []
        cam_intrinsics = []
        cam2ego_rts = []
        ego2image_rts = []

        lidar2ego_r = Quaternion(
            info["data"]["LIDAR_TOP"]["calib"]["rotation"]
        ).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info["data"]["LIDAR_TOP"]["calib"]["translation"]).T
        ego2lidar = np.linalg.inv(lidar2ego)

        lidar2global = get_lidar2global(
            info["data"]["LIDAR_TOP"]["calib"], info["data"]["LIDAR_TOP"]["pose"]
        )
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(
            info["data"]["LIDAR_TOP"]["pose"]["rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.asarray(
            info["data"]["LIDAR_TOP"]["pose"]["translation"]
        ).T

        for cam_type in self.sensor_types:
            image_paths.append(
                os.path.join(self.data_path, info["data"][cam_type]["filename"])
            )

            img2global, cam2global = get_img2global(
                info["data"][cam_type]["calib"], info["data"][cam_type]["pose"], True
            )
            lidar2img = np.linalg.inv(img2global) @ lidar2global
            lidar2cam = np.linalg.inv(cam2global) @ lidar2global
            img2lidar = np.linalg.inv(lidar2global) @ img2global

            cam2ego_r = Quaternion(
                info["data"][cam_type]["calib"]["rotation"]
            ).rotation_matrix
            cam2ego = np.eye(4)
            cam2ego[:3, :3] = cam2ego_r
            cam2ego[:3, 3] = np.array(info["data"][cam_type]["calib"]["translation"]).T

            intrinsic = info["data"][cam_type]["calib"]["camera_intrinsic"]
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic

            lidar2img_rts.append(lidar2img)
            lidar2cam_rts.append(lidar2cam)
            img2lidar_rts.append(img2lidar)
            cam_intrinsics.append(viewpad)
            cam2ego_rts.append(cam2ego)
            ego2image_rts.append(np.linalg.inv(img2global) @ ego2global)

        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=os.path.join(
                self.data_path, info["data"]["LIDAR_TOP"]["filename"]
            ),
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
        )

        return input_dict

    def __getitem__(self, index):
        if self.overfit:
            index = 0
        scene_token, index = self.keyframes[index]
        info = deepcopy(self.scene_infos[scene_token][index])
        input_dict = self.get_data_info(info)

        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()
        for t in self.pipeline:
            input_dict = t(input_dict)

        return_dict = {
            "img": input_dict["img"],  # Ncam,3,H,W
            "image_wh": input_dict["image_wh"],  # Ncam,2
            "pts_filename": input_dict["pts_filename"],
            "cam_intrinsic": input_dict["cam_intrinsic"],  # Ncam,4,4
            "lidar2cam": input_dict["lidar2cam"],  # Ncam,4,4
            "projection_mat": input_dict["projection_mat"],  # Ncam,4,4
            "occ_label": input_dict["occ_label"],  # X,Y,Z
            "occ_xyz": input_dict["occ_xyz"],  # X,Y,Z,3
            "occ_cam_mask": input_dict["occ_cam_mask"],  # X,Y,Z
            "ref2lidar": input_dict["ref2lidar"],  # 4,4
            # Optional
            "cam2ego": input_dict["cam2ego"],
            "ego2lidar": input_dict["ego2lidar"],
            "sample_id": "__".join([scene_token, str(index)]),
        }

        return return_dict

    def __len__(self):
        return len(self.keyframes)
