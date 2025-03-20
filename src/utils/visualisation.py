import os
import pdb

import numpy as np
import torch

colors = np.array(
    [
        [0, 0, 0, 255],  # 0
        [255, 120, 50, 255],
        [255, 192, 203, 255],
        [255, 255, 0, 255],
        [0, 150, 245, 255],
        [0, 255, 255, 255],
        [200, 180, 0, 255],
        [255, 0, 0, 255],
        [255, 240, 150, 255],
        [135, 60, 0, 255],
        [160, 32, 240, 255],  # 10
        [255, 0, 255, 255],
        [139, 137, 137, 255],
        [75, 0, 75, 255],
        [150, 240, 80, 255],
        [230, 230, 250, 255],
        [0, 175, 0, 255],
        [0, 255, 127, 255],  #  17
        [255, 99, 71, 255],
        [0, 191, 255, 255],
        [255, 255, 255, 255],
    ]
).astype(np.uint8)


# Function to generate 3D corners
def get_3d_corners(H, W, camera_intrins, c2w, depth_coeff=1):
    # Image corners in 2D (pixel coordinates)
    corners_2d = np.array([[0, 0], [W, 0], [W, H], [0, H]])

    # Add homogeneous coordinate to the 2D corners (make them 3D points in the camera space)
    corners_2d_homogeneous = np.hstack([corners_2d, np.ones((4, 1))])

    # Invert the camera intrinsics to project image points into normalized camera space
    camera_intrins_inv = np.linalg.inv(camera_intrins)

    # Convert 2D corners to 3D ray directions in the camera space
    rays_camera_space = (camera_intrins_inv @ corners_2d_homogeneous.T).T

    # Normalize the rays (set depth = 1 for simplicity)
    rays_camera_space /= rays_camera_space[:, 2][:, None]
    rays_camera_space *= depth_coeff

    # Convert the rays to world coordinates using the c2w matrix
    rays_world_space = (c2w @ np.hstack([rays_camera_space, np.ones((4, 1))]).T).T

    return rays_world_space[:, :3]  # Return only the 3D coordinates


def inspect_results(input_imgs, data, result_dict, tag=""):
    base_path = f"inspect/results/{tag}"
    os.makedirs(base_path, exist_ok=True)

    # # Fast inspection
    resize_factor = 0.3
    new_h = int(input_imgs.shape[3] * resize_factor)
    new_w = int(input_imgs.shape[4] * resize_factor)

    # Resize
    downsampled_imgs = torch.nn.functional.interpolate(
        input_imgs.squeeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    np.save(f"{base_path}/imgs.npy", downsampled_imgs.cpu().numpy())

    # For Mayavi
    mayavi_gt_occ = data["occ_label"].detach().cpu().numpy()[..., None]
    mayavi_gt_occ = np.concatenate(
        [mayavi_gt_occ, data["occ_xyz"].cpu().numpy()], axis=-1
    )
    np.save(f"{base_path}/gt_occ.npy", mayavi_gt_occ)

    mayavi_pred_occ = result_dict["pred_occ"][-1].detach().cpu().argmax(1).numpy()
    mayavi_pred_occ = mayavi_pred_occ.reshape(*mayavi_gt_occ.shape[:-1])
    mayavi_pred_occ = np.concatenate(
        [mayavi_pred_occ[..., None], data["occ_xyz"].cpu().numpy()], axis=-1
    )
    np.save(f"{base_path}/pred_occ.npy", mayavi_pred_occ)
    np.save(
        f"{base_path}/intrins.npy",
        result_dict["metas"]["cam_intrinsic"].cpu().numpy(),
    )
    if "cam2ego" in result_dict["metas"].keys():
        np.save(
            f"{base_path}/extrin.npy",
            result_dict["metas"]["cam2ego"].cpu().numpy(),
        )
