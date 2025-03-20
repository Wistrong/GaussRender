import math
import pdb

import torch


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getProjectionMatrix(znear, zfar, K, W, H):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    top = znear * cy / fy
    bottom = -znear * (H - cy) / fy
    right = znear * (W - cx) / fx
    left = -znear * cx / fx

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = -(right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


class MiniCam:
    def __init__(self, w2c, intrins, width, height, znear, zfar):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.znear = znear
        self.zfar = zfar
        self.FoVx = focal2fov(intrins[0, 0], width)
        self.FoVy = focal2fov(intrins[1, 1], height)

        # w2c[:3, :3] = w2c[:3, :3].T

        self.world_view_transform = w2c.transpose(0, 1)
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, K=intrins, W=width, H=height
            )
            .transpose(0, 1)
            .to(w2c.device)
            .to(intrins.dtype)
        )  # Handle intrins

        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def to_device(self, device):
        self.world_view_transform = self.world_view_transform.to(device).to(
            torch.float32
        )
        self.projection_matrix = self.projection_matrix.to(device).to(torch.float32)
        self.camera_center = self.camera_center.to(device).to(torch.float32)
        self.full_proj_transform = self.full_proj_transform.to(device).to(torch.float32)
