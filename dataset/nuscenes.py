import json
import os
import pdb
import random
from copy import deepcopy
from pathlib import Path

import mmengine
import numpy as np
from torch.utils.data import Dataset

from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS


class BaseNuscenes(Dataset):
    def __init__(
        self,
        data_root=None,
        imageset=None,
        data_aug_conf=None,
        pipeline=None,
        phase="train",
    ):
        self.data_path = data_root

        self.data_aug_conf = data_aug_conf
        self.overfit = self.data_aug_conf.get("overfit", False)
        self.test_mode = phase != "train"
        self.pipeline = []
        for t in pipeline:
            self.pipeline.append(OPENOCC_TRANSFORMS.build(t))

        self.sensor_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]

    def sub_select_data(self, object, phase, percent=0.20, seed_value=2025):
        # Set the seed for reproducibility
        random.seed(seed_value)

        # Define the output file path
        output_file = Path(f"dataset/indices/selected_indices_{phase}_nuscenes.json")

        # Check if the file already exists
        if output_file.exists():
            # Load the indices from the file
            try:
                with output_file.open("r") as f:
                    selected_indices = json.load(f)
                print(f"Loaded selected indices from {output_file}")
                return [object[i] for i in selected_indices]
            except (IOError, json.JSONDecodeError) as e:
                print(
                    f"Error loading indices from {output_file}: {e}. Generating new indices..."
                )
        else:
            print(f"{output_file} does not exist. Generating new indices...")

        # If the file doesn't exist or there was an error, generate new indices
        N = len(object)
        selected_indices = random.sample(range(N), int(N * percent))

        # Save the selected indices to the JSON file
        try:
            with output_file.open("w") as f:
                json.dump(selected_indices, f)
            print(f"Selected indices have been saved to {output_file}")
        except IOError as e:
            print(f"Error saving selected indices to {output_file}: {e}")

        return [object[i] for i in selected_indices]

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

    def get_data_info(self, info):
        return NotImplementedError("get_data_info is not implemented in BaseNuscenes")

    def __len__(self):
        return NotImplemented("__len__ is not implemented in BaseNuscenes")
