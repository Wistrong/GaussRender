import argparse
import glob
import math
import os
import pdb

import mayavi.mlab as mlab
import numpy as np

# Define colors
colors = np.array(
    [
        [0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier
        [255, 192, 203, 255],  # bicycle
        [255, 255, 0, 255],  # bus
        [0, 150, 245, 255],  # car
        [0, 255, 255, 255],  # construction_vehicle
        [200, 180, 0, 255],  # motorcycle
        [255, 0, 0, 255],  # pedestrian
        [255, 240, 150, 255],  # traffic_cone
        [135, 60, 0, 255],  # trailer
        [160, 32, 240, 255],  # truck
        [255, 0, 255, 255],  # driveable_surface
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk
        [150, 240, 80, 255],  # terrain
        [230, 230, 250, 255],  # manmade
        [0, 175, 0, 255],  # vegetation
        [0, 255, 127, 255],  # ego car
        [255, 99, 71, 255],
        [0, 191, 255, 255],
    ]
).astype(np.uint8)


def visualize_voxels(step_folder, dataset, output_folder, is_gt):
    """
    Visualize voxel data using Mayavi.

    Args:
        step_folder (str): Path to the step folder containing voxel data.
        dataset (str): Name of the dataset (occ3d or surroundocc).
        output_folder (str): Path to save the rendered images.
    """
    # Load voxel data
    file_path = os.path.join(step_folder, "gt_occ.npy" if is_gt else "pred_occ.npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Voxel data file not found: {file_path}")

    # Get coordinates
    fov_voxels = np.load(file_path)
    fov_voxels = fov_voxels[..., [1, 2, 3, 0]]
    fov_voxels = fov_voxels.reshape(-1, 4)

    # Constants
    if dataset == "occ3d":
        VOXEL_SIZE = 0.4
        empty_cls = 17
    elif dataset == "surroundocc":
        VOXEL_SIZE = 0.5
        empty_cls = 0
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Filter out ego car (label 17)
    filtering = fov_voxels[..., 3] != empty_cls
    fov_voxels = fov_voxels[filtering]
    print(f"Filtered voxels shape: {fov_voxels.shape}")

    # Create Mayavi figure
    figure = mlab.figure(size=(1600, 900), bgcolor=(1, 1, 1))
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=VOXEL_SIZE - 0.05 * VOXEL_SIZE,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )

    # Apply color mapping
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    # Load camera intrinsics and extrinsics
    cam2ego_rts = np.load(os.path.join(step_folder, "extrin.npy"))[0]
    K = np.load(os.path.join(step_folder, "intrins.npy"))[0]
    Ncams = cam2ego_rts.shape[0]

    f = 0.0055  # only define the direction
    cam_type = ["front", "front_right", "front_left", "back", "back_right", "back_left"]

    W = 1600
    H = 900
    # Loop over cameras and set camera positions
    for cam_idx in range(Ncams):
        # Set camera parameters
        scene = figure.scene
        scene.camera.position = cam2ego_rts[cam_idx][..., :3, 3].flatten()
        scene.camera.focal_point = (
            cam2ego_rts[cam_idx] @ np.array([0, 0, f, 1.0]).reshape([4, 1])
        ).flatten()[:3]

        # Field of view
        theta_x = np.degrees(2 * math.atan(W / (2 * K[cam_idx, 0, 0])))
        theta_y = np.degrees(2 * math.atan(H / (2 * K[cam_idx, 1, 1])))
        scene.camera.view_angle = (theta_x + theta_y) / 2
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [0.01, 300.0]
        scene.camera.compute_view_plane_normal()
        scene.render()

        # Render and save the figure
        mlab.draw()
        save_path = os.path.join(output_folder, f"{cam_type[cam_idx]}.png")
        mlab.savefig(save_path)
        # mlab.show()
        print(f"Saved camera view to: {save_path}")
    mlab.close()

    # BEV Rendering
    # Set camera for BEV (top-down view)
    # Resize the figure to be square
    figure = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=VOXEL_SIZE - 0.05 * VOXEL_SIZE,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )

    # Apply color mapping
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    scene = figure.scene
    scene.camera.position = [0, 0, 40]  # High above the scene
    scene.camera.focal_point = [0, 0, 0]  # Look at the center
    scene.camera.view_up = [0, 1, 0]  # Y-axis is up
    scene.camera.view_angle = 90  # Wide-angle view
    scene.camera.clipping_range = [0.01, 300.0]
    scene.camera.compute_view_plane_normal()
    scene.render()

    # Render and save the BEV figure
    mlab.draw()
    save_path = os.path.join(output_folder, "bev.png")
    mlab.savefig(save_path)
    print(f"Saved BEV view to: {save_path}")

    mlab.close()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Visualize voxel data using Mayavi.")
    parser.add_argument(
        "--folder", type=str, help="Path to the root folder containing scenes and steps"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="occ3d",
        help="Name of the dataset (default: occ3d).",
    )
    parser.add_argument("--scene-folder", type=str, default=None)

    # Parse arguments
    args = parser.parse_args()

    # Iterate over scenes and steps
    if args.scene_folder:
        step_folders = glob.glob(os.path.join(args.scene_folder, "*"))
    else:
        step_folders = glob.glob(os.path.join(args.folder, "*/*"))

    # Pred
    for step_folder in step_folders:
        output_folder = os.path.join("rendered", "/".join(step_folder.split("\\")[1:]))
        os.makedirs(output_folder, exist_ok=True)
        visualize_voxels(step_folder, args.dataset, output_folder, False)

    # GT
    for step_folder in step_folders:
        output_folder = os.path.join("rendered", "/".join(step_folder.split("\\")[1:]))
        output_folder = output_folder.replace("_render", "_gt").replace("_std", "_gt")
        os.makedirs(output_folder, exist_ok=True)
        visualize_voxels(step_folder, args.dataset, output_folder, True)


if __name__ == "__main__":
    main()
