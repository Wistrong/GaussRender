import math
import os
import pdb
import random
import sys

import numpy as np
import torch
from diff_gauss import GaussianRasterizationSettings
from diff_gauss import GaussianRasterizer as GaussianRasterizer3D
from einops import rearrange, repeat
from torch import nn

from src.utils.camera import MiniCam

from ..utils.utils import get_frequencies

C0 = 0.28209479177387814


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def covariance_from_scaling_rotation(scaling, rotation, c2ws):
    L = build_scaling_rotation(scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


def modify_extrinsics(extrinsics, phi, theta, tx, tz):
    """
    Modify camera extrinsics with given rotations and translations.

    Args:
        extrinsics (torch.Tensor): Input extrinsics of shape (B, Ncam, 4, 4).
        phi (torch.Tensor): Rotation angles around the X-axis, shape (B, Ncam).
        theta (torch.Tensor): Rotation angles around the Y-axis, shape (B, Ncam).
        tx (torch.Tensor): Translation along the Y-axis, shape (B, Ncam).
        tz (torch.Tensor): Translation along the Z-axis, shape (B, Ncam).

    Returns:
        torch.Tensor: Modified extrinsics of shape (B, Ncam, 4, 4).
    """
    B, ncam = extrinsics.shape[:2]
    dtype = extrinsics.dtype
    device = extrinsics.device

    # Convert angles to radians
    phi = phi * torch.pi / 180.0  # (B, Ncam)
    theta = theta * torch.pi / 180.0  # (B, Ncam)

    # Compute cosine and sine for rotation matrices
    cos_phi = torch.cos(phi)  # (B, Ncam)
    sin_phi = torch.sin(phi)  # (B, Ncam)
    cos_theta = torch.cos(theta)  # (B, Ncam)
    sin_theta = torch.sin(theta)  # (B, Ncam)

    # Define rotation matrices in batch
    # Rotation around X-axis (phi)
    looking_down = torch.zeros(B, ncam, 4, 4, dtype=dtype, device=device)
    looking_down[:, :, 0, 0] = 1
    looking_down[:, :, 1, 1] = cos_phi
    looking_down[:, :, 1, 2] = -sin_phi
    looking_down[:, :, 2, 1] = sin_phi
    looking_down[:, :, 2, 2] = cos_phi
    looking_down[:, :, 3, 3] = 1

    # Rotation around Y-axis (theta)
    turning_left = torch.zeros(B, ncam, 4, 4, dtype=dtype, device=device)
    turning_left[:, :, 0, 0] = cos_theta
    turning_left[:, :, 0, 2] = sin_theta
    turning_left[:, :, 1, 1] = 1
    turning_left[:, :, 2, 0] = -sin_theta
    turning_left[:, :, 2, 2] = cos_theta
    turning_left[:, :, 3, 3] = 1

    # Perform matrix multiplication in batch: R_x * R_y * extrinsics
    extrinsics = looking_down @ turning_left @ extrinsics

    # Apply translations along the Y-axis (tx) and Z-axis (tz)
    extrinsics[:, :, 1, 3] += tx  # Translate along Y-axis
    extrinsics[:, :, 2, 3] += tz  # Translate along Z-axis

    return extrinsics


class Renderer(nn.Module):
    def __init__(
        self,
        sh_degree=3,
        white_background=False,
        radius=1,
        with_cam_rendering=False,
        with_bev_rendering=False,
        with_depth_rendering=False,
        with_bev_depth_rendering=True,
        inspect=False,
        pc_range=None,
        voxel_size=None,
        render_ncam=1,
        dataset_tag=None,
        num_render_cls=19,
        num_classes=None,
        render_gt_mode=None,
        gaussian_scale=None,
        cam_idx=None,
    ):
        super().__init__()

        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius
        self.num_classes = num_classes

        self.setup_functions()
        self.with_cam_rendering = with_cam_rendering
        self.with_bev_rendering = with_bev_rendering
        self.with_depth_rendering = with_depth_rendering
        self.with_bev_depth_rendering = with_bev_depth_rendering
        self.inspect = inspect

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        if render_ncam is None and cam_idx is not None:
            render_ncam = len(cam_idx)
        else:
            render_ncam = 1
        self.render_ncam = render_ncam

        # SurroundOcc
        self.num_render_cls = num_render_cls
        bg_color = torch.zeros((self.num_render_cls), dtype=torch.float32)
        self.dataset_tag = dataset_tag
        if dataset_tag in ["surroundocc", "kitti360"]:
            bg_color[0] = 1.0
        # Occ3d
        elif dataset_tag == "occ3d":
            bg_color[17] = 1.0

        if gaussian_scale is None:
            if dataset_tag in ["occ3d"]:
                self.gaussian_scale = 0.27
            elif dataset_tag == "surroundocc":
                self.gaussian_scale = 0.25
            elif dataset_tag == "kitti360":
                self.gaussian_scale = 0.1
        else:
            self.gaussian_scale = gaussian_scale

        if render_gt_mode is None:
            if dataset_tag in ["surroundocc"]:
                render_gt_mode = "trajectory"
            elif dataset_tag in ["occ3d", "kitti360"]:
                render_gt_mode = "sensor"
        assert render_gt_mode in [
            None,
            "elevated",
            "sensor",
            "random",
            "stereo",
            "trajectory",
        ]

        self.render_gt_mode = render_gt_mode
        assert dataset_tag in ["surroundocc", "kitti360", "occ3d"]
        self.register_buffer("bg_color", bg_color)
        self.cam_idx = cam_idx

    def setup_functions(self):

        self.scaling_activation = torch.nn.functional.softplus

        self.opacity_activation = torch.sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def set_rasterizer(self, cam, scaling_modifier=1.0, device="cuda"):
        # Set up rasterization configuration
        tanfovx = math.tan(cam.FoVx * 0.5)
        tanfovy = math.tan(cam.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform,
            sh_degree=self.sh_degree,
            campos=cam.camera_center,
            prefiltered=False,
            debug=False,
        )
        return GaussianRasterizer3D(raster_settings=raster_settings)

    def get_params(
        self,
        position_lr_init=0.00016,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
    ):
        l = [
            {"params": [self._xyz], "lr": position_lr_init, "name": "xyz"},
            {"params": [self._features_dc], "lr": feature_lr, "name": "f_dc"},
            {
                "params": [self._features_rest],
                "lr": feature_lr / 20.0,
                "name": "f_rest",
            },
            {"params": [self._opacity], "lr": opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": rotation_lr, "name": "rotation"},
        ]
        return l

    def get_scaling(self, _scaling):
        return self.scaling_activation(_scaling)

    def get_rotation(self, _rotation):
        return self.rotation_activation(_rotation)

    def get_opacity(self, _opacity):
        return self.opacity_activation(_opacity)

    def get_covariance(self, _scaling, _rotation, c2ws):
        return covariance_from_scaling_rotation(
            self.get_scaling(_scaling), self.get_rotation(_rotation), c2ws
        )

    def get_render_matrix(self, metas, red=1):
        Ncams = self.render_ncam
        intrins = metas["cam_intrinsic"].float()
        lidar2cam = metas["lidar2cam"].float()
        ref2lidar = metas["ref2lidar"].float()
        B = lidar2cam.shape[0]
        device = intrins.device
        dtype = intrins.dtype

        ################################
        # Get cameras matrices
        ################################
        # Select indices
        if self.dataset_tag in ["surroundocc", "occ3d"]:
            cam_pool = [
                "CAM_FRONT",
                "CAM_FRONT_LEFT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]
            if self.cam_idx is None:
                weights = [1 / 3, 1 / 6, 1 / 6, 1 / 3, 1 / 6, 1 / 6]
                random_idx = random.choices(
                    list(range(len(cam_pool))), weights=weights, k=Ncams
                )
            else:
                random_idx = self.cam_idx
        elif self.dataset_tag == "kitti360":
            random_idx = [0] * Ncams
        else:
            raise NotImplementedError
        ref2cams = lidar2cam[:, random_idx] @ ref2lidar
        intrins_cams = intrins[:, random_idx]

        # Modify cameras
        if self.render_gt_mode == "elevated":
            random_theta = torch.rand((Ncams), device=device, dtype=dtype) * 10 - 5
            ref2cams = modify_extrinsics(
                ref2cams,
                torch.tensor([20], device=device, dtype=dtype),
                random_theta,
                torch.tensor([8], device=device, dtype=dtype),
                0,
            )
        elif self.render_gt_mode == "sensor":
            pass
        elif self.render_gt_mode == "trajectory":
            xmin, ymin, zmin, xmax, ymax, zmax = self.pc_range
            rnd_factor = random.uniform(0, 1)
            ref2cams[:, :, :3, -1] += torch.tensor(
                [
                    (ymax - abs(ymin)) / 2 * rnd_factor,
                    2.0,
                    -(xmax - abs(xmin)) * 0.7 * rnd_factor,
                ],
                device=device,
                dtype=dtype,
            )
        elif self.render_gt_mode == "random":
            random_phi = torch.rand((Ncams), device=device) * 20
            random_theta = torch.rand((Ncams), device=device) * 20 - 10
            random_tz = torch.rand((Ncams), device=device) * 20 - 10
            ref2cams = modify_extrinsics(
                ref2cams, random_phi, random_theta, random_phi * 8 / 20, random_tz
            )
        elif self.render_gt_mode == "stereo":
            # The idea is to double the number of cameras and shift the half to the left or right.
            ref2cams_stereo = modify_extrinsics(
                ref2cams,
                torch.zeros((Ncams), device=device),
                torch.zeros((Ncams), device=device),
                torch.rand((Ncams), device=device) * 2 - 1,
                torch.rand((Ncams), device=device) * 2 - 1,
            )
            intrins_cams = intrins_cams.repeat(1, 2, 1, 1)
            ref2cams = torch.cat([ref2cams, ref2cams_stereo], dim=1)
            Ncams = Ncams * 2
        else:
            raise NotImplementedError

        # BEV
        xmin, ymin, zmin, xmax, ymax, zmax = self.pc_range
        ref2cams_bev = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, xmax - xmin],
                [0, 0, 0, 1.0],
            ],
            device=device,
            dtype=dtype,
        ).repeat(B, 1, 1, 1)
        if self.dataset_tag in ["surroundocc", "occ3d"]:
            SIZE = 400
            intrins_bev = torch.tensor(
                [
                    [SIZE, 0, SIZE / 2, 0],
                    [0, SIZE, SIZE / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                device=device,
                dtype=dtype,
            ).repeat(B, 1, 1, 1)
        elif self.dataset_tag == "kitti360":
            SIZE = 400
            intrins_bev = torch.tensor(
                [
                    [SIZE, 0, 0, 0],
                    [0, SIZE, 200, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                device=device,
                dtype=dtype,
            ).repeat(B, 1, 1, 1)

        ###############################
        # Transform to minicam object.
        ###############################
        znear, zfar = 0.1, 100.0
        cam_to_render = [[]]
        bev_cam_to_render = []
        for i in range(B):
            # CAMERA
            for j in range(Ncams):
                intrin_cam = intrins_cams[i, j]
                ref2cam = ref2cams[i, j]
                # Cam to render
                if self.dataset_tag in ["surroundocc", "occ3d"]:
                    width, height = 1600 // red, 900 // red
                elif self.dataset_tag == "kitti360":
                    width, height = 1408 // red, 376 // red
                else:
                    raise NotImplementedError
                intrin_cam[:2, :3] = intrin_cam[:2, :3] / red
                cam = MiniCam(ref2cam, intrin_cam, width, height, znear, zfar)
                cam.to_device(device)
                cam_to_render[i].append(cam)

            # BEV
            ref2cam = ref2cams_bev[i].squeeze(0)
            intrin_bev = intrins_bev[i].squeeze(0)
            size = 400
            bev_cam = MiniCam(ref2cam, intrin_bev, size, size, znear, zfar)
            bev_cam.to_device(device)
            bev_cam_to_render.append(bev_cam)

        return {
            # CAMERA
            "intrins": intrins_cams,
            "ref2cams": ref2cams,
            "cam_to_render": cam_to_render,
            # BEV
            "intrins_bev": intrins_bev,
            "ref2cams_bev": ref2cams_bev,
            "bev_cam_to_render": bev_cam_to_render,
        }

    def render_gt(
        self,
        data,
        dict_valid,
    ):
        occ_label = data["occ_label"]
        intrins_cam = data["render"]["intrins"]
        cam_to_render = data["render"]["cam_to_render"]
        bev_cam_to_render = data["render"]["bev_cam_to_render"]

        B, Ncams = intrins_cam.shape[:2]
        device = intrins_cam.device
        dtype = intrins_cam.dtype
        gaussian_scale = self.gaussian_scale

        # Gaussian rendering
        render_cam = [[]]
        render_cam_depth = [[]]
        render_bev = []
        render_bev_depth = []

        # Render CAM
        for i in range(B):
            # Get GT means
            # occ_label of shape: B,X,Y,Z
            occ_xyz = torch.stack(torch.where(torch.ones_like(occ_label[i])), dim=-1)

            # Visibility mask: remove ignore and empty labels.
            occ_label = occ_label.flatten().to(torch.int64)

            if "lidar_points" in data.keys() and data["lidar_points"][0] is not None:
                pts = data["lidar_points"][i]
                xmin, ymin, zmin, xmax, ymax, zmax = self.pc_range
                pts_label = pts[:, 3]
                pts = (
                    pts[:, :3]
                    - torch.tensor([xmin, ymin, zmin], device=occ_label.device)
                ) / self.voxel_size - 0.5

                X, Y, Z = data["occ_label"][i].shape
                lidar_mask = (
                    (pts[:, 0] >= 0)
                    & (pts[:, 1] >= 0)
                    & (pts[:, 2] >= 0)
                    & (pts[:, 0] < X)
                    & (pts[:, 1] < Y)
                    & (pts[:, 2] < Z)
                )
                pts = pts[lidar_mask].int()

                occ_label = torch.full_like(occ_label, 255)
                occ_label[pts[:, 0] * Y * Z + pts[:, 1] * Z + pts[:, 2]] = pts_label[
                    lidar_mask
                ].long()

            if self.dataset_tag in ["surroundocc", "kitti360"]:
                empty_class = 0
                valid_base = (occ_label != empty_class) & (occ_label != 255)
            elif self.dataset_tag == "occ3d":
                empty_class = 17
                valid_base = (occ_label != empty_class) & (occ_label != 255)
            else:
                raise NotImplementedError

            for j in range(Ncams):
                if dict_valid is not None and len(dict_valid["cam"]) > 0:
                    valid = dict_valid["cam"][i][j] & valid_base
                else:
                    valid = valid_base

                xmin, ymin, zmin, xmax, ymax, zmax = self.pc_range
                means = (occ_xyz[valid] + 0.5) * self.voxel_size + torch.tensor(
                    [xmin, ymin, zmin], device=device
                )

                # Get GT semantics
                g = valid.sum().item()

                # Merge Occ3D with 18 classes, SurroundOcc with 17 classes, SSCKitti360 with 19 classes.
                num_render_cls = self.num_render_cls
                semantics = torch.zeros(g, num_render_cls, device=device, dtype=dtype)

                # One Hot encoding
                semantics[torch.arange(g), occ_label[valid]] = 1.0

                # Get other parameters
                opacities = torch.ones((g, 1), device=device, dtype=dtype)
                rotations = torch.zeros((g, 4), device=device, dtype=dtype)
                rotations[:, 0] = 1.0
                scales = torch.full((g, 3), gaussian_scale, device=device, dtype=dtype)

                out_render = self.render_img(
                    cam_to_render[i][j],
                    means,
                    semantics,
                    opacities,
                    scales,
                    rotations,
                    device,
                )
                render_cam[i].append(out_render["image"])
                render_cam_depth[i].append(out_render["depth"])

            # Render BEV
            if dict_valid is not None and len(dict_valid["bev"]) > 0:
                valid = dict_valid["bev"][i] & valid_base
            else:
                valid = valid_base
            xmin, ymin, zmin, xmax, ymax, zmax = self.pc_range
            means = (occ_xyz[valid] + 0.5) * self.voxel_size + torch.tensor(
                [xmin, ymin, zmin], device=device
            )
            g = valid.sum().item()
            num_render_cls = self.num_render_cls
            semantics = torch.zeros(g, num_render_cls, device=device, dtype=dtype)
            semantics[torch.arange(g), occ_label[valid]] = 1.0
            opacities = torch.ones((g, 1), device=device, dtype=dtype)
            rotations = torch.zeros((g, 4), device=device, dtype=dtype)
            rotations[:, 0] = 1.0
            scales = torch.full((g, 3), gaussian_scale, device=device, dtype=dtype)

            out_render = self.render_img(
                bev_cam_to_render[i],
                means,
                semantics,
                opacities,
                scales,
                rotations,
                device,
            )
            render_bev.append(out_render["image"])
            render_bev_depth.append(out_render["depth"])

        return (
            torch.stack([torch.stack([o for o in render_cam[i]]) for i in range(B)]),
            torch.stack(
                [torch.stack([o for o in render_cam_depth[i]]) for i in range(B)]
            ),
            torch.stack([torch.stack([o for o in render_bev[i]]) for i in range(B)]),
            torch.stack(
                [torch.stack([o for o in render_bev_depth[i]]) for i in range(B)]
            ),
        )

    def render_img(
        self,
        cam,
        centers,
        colors_precomp,
        opacity,
        scales,
        rotations,
        device,
        cov3D_precomp=None,
    ):
        rasterizer = self.set_rasterizer(cam, device=device)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                centers,
                dtype=centers.dtype,
                requires_grad=True,
                device=device,
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        assert scales.shape[-1] == 3

        rendered_image, rendered_depth, *_ = rasterizer(
            means3D=centers,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3Ds_precomp=cov3D_precomp,
            extra_attrs=None,
        )
        return {
            f"image": rendered_image[: self.num_classes].permute(1, 2, 0),
            f"depth": rendered_depth.permute(1, 2, 0),
        }

    def render_bev(self, gaussians, cam, index_b, valid_gauss, intrin, ref2cam, metas):
        device = gaussians.means.device
        # Rendering
        valid = valid_gauss

        semantics = gaussians.semantics[index_b][valid]
        scales = gaussians.scales[index_b][valid]
        rotations = gaussians.rotations[index_b][valid]
        opacities = gaussians.opacities[index_b][valid]
        means = gaussians.means[index_b][valid]

        semantics = self.prepare_semantics(semantics)

        inspect = self.inspect
        # fmt: off
        frame = self.render_img(cam,means,semantics,opacities,scales,rotations,device,)
        if inspect:
            torch.save(frame["image"].detach().cpu(), "./inspect/rendered_bev_image.pth")
            torch.save(frame["depth"].detach().cpu(), "./inspect/rendered_bev_depth.pth")
        # fmt: on
        return frame, valid

    def render_cam(self, gaussians, cam, index_b, valid_gauss, intrin, ref2cam, metas):
        device = gaussians.means.device
        # Filter only rays where means is visible from a camera.
        pts = ref2cam @ (
            torch.cat(
                [
                    gaussians.means[index_b],
                    torch.ones_like(gaussians.means[index_b, :, :1]),
                ],
                dim=1,
            )
            .transpose(0, 1)
            .contiguous()
        )
        z_valid = pts[2] > 0.0

        # From cam to pixel
        pts = (intrin @ pts)[:3]
        z = pts[2]
        pts = pts[:2]

        # Normalize z
        pts = pts / z

        x_valid = (pts[0] > +0.5) & (pts[0] < cam.image_width - 0.5)
        y_valid = (pts[1] > +0.5) & (pts[1] < cam.image_height - 0.5)
        valid = x_valid & y_valid & z_valid & valid_gauss

        semantics = gaussians.semantics[index_b][valid]
        scales = gaussians.scales[index_b][valid]
        rotations = gaussians.rotations[index_b][valid]
        opacities = gaussians.opacities[index_b][valid]
        means = gaussians.means[index_b][valid]

        semantics = self.prepare_semantics(semantics)

        inspect = self.inspect
        # fmt: off
        frame = self.render_img(cam,means,semantics,opacities,scales,rotations,device) 
        
        if inspect:
            torch.save(frame["image"].detach().cpu(), "./inspect/rendered_image.pth")
            torch.save(frame["depth"].detach().cpu(), "./inspect/rendered_depth.pth")
        # fmt: on
        return frame, valid

    def prepare_semantics(self, semantics):
        # Apply softmax such that output is between 0 and 1.
        semantics = torch.nn.functional.softmax(semantics, dim=-1)
        return semantics

    def filter_using_entropy(self, index_b, valid, gaussians, metas):
        preds = gaussians.semantics[index_b][valid]
        target = metas["occ_label"][index_b].flatten()[valid]
        losses = torch.nn.functional.cross_entropy(preds, target, reduction="none")
        M = max(1, int(preds.shape[0] * 0.5))
        # Get indices of top M hardest elements
        top_indices = torch.topk(losses, M, largest=True).indices
        valid_filter = torch.zeros_like(valid)
        global_indices = torch.nonzero(valid, as_tuple=True)[0]
        top_global_indices = global_indices[top_indices]
        valid_filter[top_global_indices] = True
        return valid_filter

    def filter_by_colors(self, metas, use_saved_classes=False):
        valid_colors = []
        labels = metas["occ_label"]
        B = labels.shape[0]
        device = labels.device

        for i in range(B):
            colors_precomp = labels[i].flatten()
            valid_color = torch.ones_like(colors_precomp, dtype=torch.bool)
            valid_colors.append(valid_color)
        valid_colors = torch.stack(valid_colors, dim=0)
        return valid_colors

    def forward(self, gaussians, metas):
        """Gaussians are in either ego or lidar coordinate system.
        Render_mat are always in lidar coordinate system, so if gaussians
        are in ego, we should apply a transform."""
        (
            intrins,
            ref2cams,
            intrins_bev,
            ref2cams_bev,
            cam_to_render,
            bev_cam_to_render,
        ) = (
            metas["render"]["intrins"],
            metas["render"]["ref2cams"],
            metas["render"]["intrins_bev"],
            metas["render"]["ref2cams_bev"],
            metas["render"]["cam_to_render"],
            metas["render"]["bev_cam_to_render"],
        )
        B, Ncams = intrins.shape[:2]

        dict_out = {
            "cam": [[]] if self.with_cam_rendering else None,
            "depth": ([[]] if self.with_depth_rendering else None),
            "bev": [] if self.with_bev_rendering else None,
            "bev_depth": [] if self.with_bev_depth_rendering else None,
            "valid": {"cam": [[]], "bev": []},
        }

        for i in range(B):
            # We remove the ignore label during rendering since it is ignored during optimization.
            valid_gauss = metas["occ_label"][i].flatten() != 255

            if self.with_cam_rendering or self.with_depth_rendering:
                for j in range(Ncams):
                    intrin = intrins[i][j]
                    ref2cam = ref2cams[i][j]
                    frame_cam, valid_cam = self.render_cam(
                        gaussians,
                        cam_to_render[i][j],
                        i,
                        valid_gauss,
                        intrin,
                        ref2cam,
                        metas,
                    )
                    if self.with_cam_rendering:
                        dict_out["cam"][i].append(frame_cam["image"])
                    if self.with_depth_rendering:
                        dict_out["depth"][i].append(frame_cam["depth"])
                    dict_out["valid"]["cam"][i].append(valid_cam)

            if self.with_bev_rendering or self.with_bev_depth_rendering:
                frame_bev, valid_bev = self.render_bev(
                    gaussians,
                    bev_cam_to_render[i],
                    i,
                    valid_gauss,
                    intrins_bev[i][0],
                    ref2cams_bev[i][0],
                    metas,
                )
                if self.with_bev_rendering:
                    dict_out["bev"].append(frame_bev["image"])
                if self.with_bev_depth_rendering:
                    dict_out["bev_depth"].append(frame_bev["depth"])
                dict_out["valid"]["bev"].append(valid_bev)

        if self.with_cam_rendering:
            dict_out["cam"] = torch.stack(
                [torch.stack([o for o in dict_out["cam"][i]]) for i in range(B)]
            )
        if self.with_depth_rendering:
            dict_out["depth"] = torch.stack(
                [torch.stack([o for o in dict_out["depth"][i]]) for i in range(B)]
            )

        if self.with_bev_rendering:
            dict_out["bev"] = torch.stack(dict_out["bev"])
        if self.with_bev_depth_rendering:
            dict_out["bev_depth"] = torch.stack(dict_out["bev_depth"])
        dict_out["gaussians"] = gaussians
        return dict_out
