import json
import math
import os
import pdb
import re

import mmcv
import numpy as np
import torch
from numpy import random
from PIL import Image

from dataset.sscbench.helpers import (
    compute_CP_mega_matrix,
    compute_local_frustums,
    vox2pix,
)

from . import OPENOCC_TRANSFORMS


@OPENOCC_TRANSFORMS.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __init__(
        self,
    ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if "img" in results:
            if isinstance(results["img"], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results["img"]]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
            else:
                imgs = np.ascontiguousarray(results["img"].transpose(2, 0, 1))
            results["img"] = torch.from_numpy(imgs)
        return results

    def __repr__(self):
        return self.__class__.__name__


@OPENOCC_TRANSFORMS.register_module()
class NuScenesAdaptor(object):
    def __init__(self, num_cams, use_ego=False):
        self.num_cams = num_cams
        self.projection_key = "ego2img" if use_ego else "lidar2img"
        self.use_ego = use_ego
        return

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict[self.projection_key])
        )
        input_dict["ref2lidar"] = (
            np.float32(input_dict["ego2lidar"])
            if self.use_ego
            else np.eye(4, dtype=np.float32)
        )
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        return input_dict


@OPENOCC_TRANSFORMS.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_configs = results.get("aug_configs")
        if aug_configs is None:
            return results
        resize, resize_dims, crop, flip, rotate = aug_configs
        for key in ["img"]:
            imgs = results[key]
            N = len(imgs)
            new_imgs = []
            for i in range(N):
                img = Image.fromarray(np.uint8(imgs[i]))
                img, ida_mat = self._img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                mat = np.eye(4)
                mat[:3, :3] = ida_mat
                new_imgs.append(np.array(img).astype(np.float32))
                if key == "img":
                    results["lidar2img"][i] = mat @ results["lidar2img"][i]
                    results["ego2img"][i] = mat @ results["ego2img"][i]
                    if "cam_intrinsic" in results:
                        results["cam_intrinsic"][i][:3, :3] *= resize

            results[key] = new_imgs
            if key == "img":
                results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat


@OPENOCC_TRANSFORMS.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, int_divide=False, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.int_divide = int_divide

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(
                img if not self.int_divide else img / 255,
                self.mean,
                self.std,
                self.to_rgb,
            )
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(
        self,
        to_float32=False,
        color_type="unchanged",
    ):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1
        )
        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


class LoadOccupancyBase(object):
    def __init__(self, occ_path):
        self.occ_path = occ_path

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([xxx, yyy, zzz], dim=-1).numpy()
        return xyz  # x, y, z, 3

    def __call__(self, results):
        return NotImplementedError

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancySurroundOcc(LoadOccupancyBase):
    def __init__(self, occ_path):
        super().__init__(occ_path)
        self.occ_path = occ_path

        xyz = self.get_meshgrid([-50, -50, -5.0, 50, 50, 3.0], [200, 200, 16], 0.5)
        self.xyz = np.concatenate(
            [xyz, np.ones_like(xyz[..., :1])], axis=-1
        )  # x, y, z, 4

    def __call__(self, results):
        label_file = os.path.join(
            self.occ_path, results["pts_filename"].split("/")[-1] + ".npy"
        )
        label = np.load(label_file)

        # Class 0 should be ignored.
        # https://github.com/weiyithu/SurroundOcc/blob/d346e8ce476817dfd8492226e7b92660955bf89c/projects/mmdet3d_plugin/datasets/pipelines/loading.py#L30
        label[:, 3][label[:, 3] == 0] = 255

        # 0 is the new by default class, representing empty. (cf. SurroundOcc repo.)
        new_label = np.zeros((200, 200, 16), dtype=np.int64)
        new_label[label[:, 0], label[:, 1], label[:, 2]] = label[:, 3]

        results["occ_label"] = new_label
        results["occ_cam_mask"] = new_label != 255

        occ_xyz = self.xyz[..., :3]
        results["occ_xyz"] = occ_xyz
        return results


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancyOcc3dNuscenes(LoadOccupancyBase):
    def __init__(self, occ_path, **kwargs):
        super().__init__(occ_path)
        self.occ_path = occ_path

        xyz = self.get_meshgrid([-40, -40, -1.0, 40, 40, 5.4], [200, 200, 16], 0.4)
        self.xyz = np.concatenate(
            [xyz, np.ones_like(xyz[..., :1])], axis=-1
        )  # x, y, z, 4

    def __call__(self, results):
        label_file = os.path.join(
            self.occ_path, "/".join(results["occ_path"].split("/")[-2:]), "labels.npz"
        )
        file = np.load(label_file)
        labels, mask_camera = (
            file["semantics"],
            file["mask_camera"],
        )

        results["occ_label"] = labels
        results["occ_cam_mask"] = mask_camera

        occ_xyz = self.xyz[..., :3]
        results["occ_xyz"] = occ_xyz
        return results


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancyKITTI360(object):

    def __init__(self, occ_path, semantic=False):
        self.occ_path = occ_path
        self.semantic = semantic

        xyz = self.get_meshgrid(
            [0.0, -25.6, -2.0, 51.2, 25.6, 4.4], [256, 256, 32], 0.2
        )
        self.xyz = np.concatenate(
            [xyz, np.ones_like(xyz[..., :1])], axis=-1
        )  # x, y, z, 4

    def get_meshgrid(self, ranges, grid, reso):
        xxx = (torch.arange(grid[0], dtype=torch.float) + 0.5) * reso + ranges[0]
        yyy = (torch.arange(grid[1], dtype=torch.float) + 0.5) * reso + ranges[1]
        zzz = (torch.arange(grid[2], dtype=torch.float) + 0.5) * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([xxx, yyy, zzz], dim=-1).numpy()
        return xyz  # x, y, z, 3

    def __call__(self, results):
        occ_xyz = self.xyz[..., :3]
        results["occ_xyz"] = occ_xyz

        ## read occupancy label
        label_path = os.path.join(
            self.occ_path, results["sequence"], "{}_1_1.npy".format(results["token"])
        )
        label = np.load(label_path).astype(np.int64)

        results["occ_cam_mask"] = label != 255
        results["occ_label"] = label
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadKitti360Monoscene(object):
    def __init__(
        self,
        vox_origin,
        voxel_size,
        scale_3ds,
        img_W,
        img_H,
        scene_size,
        output_scale,
        split,
        label_root,
        frustum_size,
    ):
        self.vox_origin = np.array(vox_origin)
        self.voxel_size = np.array(voxel_size)
        self.scale_3ds = scale_3ds
        self.img_W = img_W
        self.img_H = img_H
        self.scene_size = scene_size
        self.output_scale = np.array(output_scale)
        self.split = split
        self.label_root = label_root
        self.frustum_size = frustum_size
        return

    def __call__(self, results):
        T_velo_2_cam = results["lidar2cam"][0]
        cam_k = results["cam_intrinsic"][0, :3, :3]
        sequence = results["sequence"]
        frame_id = results["sample_idx"]
        data = {}

        for scale_3d in self.scale_3ds:
            projected_pix, fov_mask, pix_z = vox2pix(
                T_velo_2_cam,
                cam_k,
                self.vox_origin,
                self.voxel_size * scale_3d,
                self.img_W,
                self.img_H,
                self.scene_size,
            )
            data["projected_pix_{}".format(scale_3d)] = projected_pix
            data["fov_mask_{}".format(scale_3d)] = fov_mask
            data["pix_z_{}".format(scale_3d)] = pix_z

        target_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
        target = np.load(target_1_path)
        target_8_path = os.path.join(self.label_root, sequence, frame_id + "_1_8.npy")
        target_1_8 = np.load(target_8_path)
        CP_mega_matrix = compute_CP_mega_matrix(target_1_8)
        data["CP_mega_matrix"] = CP_mega_matrix

        # Compute the masks, each indicate the voxels of a local frustum
        if self.split != "test":
            projected_pix_output = data["projected_pix_{}".format(self.output_scale)]
            pix_z_output = data["pix_z_{}".format(self.output_scale)]
            frustums_masks, frustums_class_dists = compute_local_frustums(
                projected_pix_output,
                pix_z_output,
                target,
                self.img_W,
                self.img_H,
                dataset="kitti",
                n_classes=19,
                size=self.frustum_size,
            )
        else:
            frustums_masks = None
            frustums_class_dists = None
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v)
        results.update({"monoscene": data})
        return results
