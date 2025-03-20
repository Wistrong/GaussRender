import argparse
import os

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_grid_gif_for_scenes(folder_path, fps=2):
    """
    Create a grid GIF for each scene, showing all camera views in a grid layout with BEV images and a legend.
    """
    output_folder = folder_path.replace("rendered", "gifs")
    os.makedirs(output_folder, exist_ok=True)

    camera_views = {
        "FRONT_LEFT": (0, 0),
        "FRONT": (0, 1),
        "FRONT_RIGHT": (0, 2),
        "BACK_LEFT": (1, 0),
        "BACK": (1, 1),
        "BACK_RIGHT": (1, 2),
    }
    prediction_views = {
        "FRONT_LEFT": (2, 0),
        "FRONT": (2, 1),
        "FRONT_RIGHT": (2, 2),
        "BACK_LEFT": (3, 0),
        "BACK": (3, 1),
        "BACK_RIGHT": (3, 2),
    }
    bev_positions = [(0, 3), (1, 3), (2, 3), (3, 3)]  # BEV occupies 2x2 grid

    # Load a font for the text (you may need to specify the path to a TTF font file)
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Adjust font size as needed
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font

    for scene_folder in sorted(os.listdir(folder_path)):
        scene_path = os.path.join(folder_path, scene_folder)
        if not os.path.isdir(scene_path):
            continue

        print(f"Processing scene {scene_folder}...")
        grid_images = []
        frame_folders = sorted(
            os.listdir(scene_path),
            key=lambda x: int(x) if x.isdigit() else float("inf"),
        )

        for enum, frame_folder in enumerate(frame_folders):
            frame_path = os.path.join(scene_path, frame_folder)
            if not os.path.isdir(frame_path):
                continue

            print(f"Processing frame {frame_folder} in scene {scene_folder}...")

            image_path = os.path.join(
                frame_path.replace("rendered", "results"), "imgs.npy"
            )
            images = {}
            if os.path.exists(image_path):
                image_array = np.load(image_path).transpose(0, 2, 3, 1)
                image_array = image_array + np.array(
                    [103.530, 116.280, 123.675]
                ).reshape(1, 1, 1, 3)
                image_array = image_array[..., [2, 1, 0]].astype(np.uint8)
                for i, view in enumerate(
                    [
                        "FRONT",
                        "FRONT_RIGHT",
                        "FRONT_LEFT",
                        "BACK",
                        "BACK_RIGHT",
                        "BACK_LEFT",
                    ]
                ):
                    images[view] = Image.fromarray(image_array[i])
            else:
                print(f"Missing image file: {image_path}")
                images = {
                    view: Image.new("RGB", (480, 640), "black") for view in camera_views
                }

            bev_gt_path = os.path.join(frame_path.replace("_render", "_gt"), "bev.png")
            bev_pred_path = os.path.join(frame_path, "bev.png")
            bev_gt = (
                Image.open(bev_gt_path).convert("RGB")
                if os.path.exists(bev_gt_path)
                else Image.new("RGB", (480, 640), "gray")
            )
            bev_pred = (
                Image.open(bev_pred_path).convert("RGB")
                if os.path.exists(bev_pred_path)
                else Image.new("RGB", (480, 640), "gray")
            )

            image_width, image_height = images["FRONT"].size
            grid_width, grid_height = 3, 4  # Now 4x4 with BEV
            grid_image = Image.new(
                "RGB", (image_width * grid_width + 268 * 2, image_height * grid_height)
            )

            # Draw camera views
            for view, (row, col) in camera_views.items():
                grid_image.paste(images[view], (col * image_width, row * image_height))
                # Add text label
                draw = ImageDraw.Draw(grid_image)
                draw.text(
                    (col * image_width + 20, row * image_height + 20),
                    f"{view}",
                    font=font,
                    fill="white",
                )

            # Draw prediction views
            for view, (row, col) in prediction_views.items():
                path = os.path.join(frame_path, view.lower() + ".png")
                if os.path.exists(image_path):
                    img = Image.open(path).resize((image_width, image_height))
                grid_image.paste(img, (col * image_width, row * image_height))
                # Add text label
                draw = ImageDraw.Draw(grid_image)
                draw.text(
                    (col * image_width + 20, row * image_height + 20),
                    f"Rendered: {view}",
                    font=font,
                    fill="black",
                )

            # Draw BEV images
            grid_image.paste(
                bev_gt.resize((268 * 2, 268 * 2)), (3 * image_width, 0 * image_height)
            )
            draw.text(
                (3 * image_width + 20, 10), "BEV: Ground Truth", font=font, fill="black"
            )

            grid_image.paste(
                bev_pred.resize((268 * 2, 268 * 2)), (3 * image_width, 2 * image_height)
            )
            draw.text(
                (3 * image_width + 20, 2 * image_height + 20),
                "BEV: Prediction",
                font=font,
                fill="black",
            )

            if enum % 10 == 0:
                imageio.mimsave(
                    f"qualitative_{enum}.png", [np.array(grid_image)], fps=fps
                )
            grid_images.append(grid_image)

        gif_path = os.path.join(output_folder, f"{scene_folder}_grid.gif")
        imageio.mimsave(gif_path, [np.array(img) for img in grid_images], fps=fps)
        print(f"GIF saved to: {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a grid GIF including BEV images and a legend."
    )
    parser.add_argument(
        "--rendered-folder", type=str, required=True, help="Path to rendered images."
    )
    parser.add_argument(
        "--fps", type=int, default=2, help="Frames per second for the GIF."
    )
    args = parser.parse_args()
    create_grid_gif_for_scenes(args.rendered_folder, args.fps)
