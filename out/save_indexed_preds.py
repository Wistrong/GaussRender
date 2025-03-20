import argparse
import json
import os
import os.path as osp
import pdb
import sys
import warnings

import numpy as np
import torch
from mmengine import Config, DictAction
from tqdm import tqdm

from dataset import OPENOCC_DATASET, get_dataloader
from dataset.utils import custom_collate_fn_temporal
from src.model.utils.utils import (
    modify_config_based_on_dataset,
    modify_file_inplace,
    set_gt_render,
    set_matrix_to_render,
)
from src.utils.setup import init_logger, init_model, resume_from_checkpoint
from src.utils.visualisation import inspect_results

warnings.filterwarnings("ignore")


def main(local_rank, args):
    # load config
    modify_file_inplace("./config/_base_/data_select.py", args.dataset)
    cfg = Config.fromfile(args.py_config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = modify_config_based_on_dataset(cfg, args.dataset)
    cfg.work_dir = args.work_dir

    distributed = False

    # build model
    logger = init_logger(cfg, log_file="/tmp/run.log")
    model, raw_model = init_model(local_rank, distributed, cfg, logger)
    model.eval()

    # resume and load
    cfg.resume_from = ""
    if args.resume_from:
        cfg.resume_from = args.resume_from

    epoch, global_iter, last_iter = resume_from_checkpoint(
        cfg,
        logger,
        0,
        0,
        0,
        raw_model,
        strict_state=args.no_strict_state,
    )

    test_dataset = OPENOCC_DATASET.build(cfg.test_dataset_config)
    with torch.no_grad():
        for i_iter_val in args.indices:
            data = custom_collate_fn_temporal([test_dataset[i_iter_val]])

            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop("img")

            # Render
            if "render" in cfg.loss_input_convertion.keys():
                set_matrix_to_render(distributed, model, data)

            result_dict = model(imgs=input_imgs, metas=data)

            # Render
            if "render" in cfg.loss_input_convertion.keys():
                set_gt_render(distributed, model, data, result_dict["render"]["valid"])

            # Inspect output
            idx_tag = result_dict["metas"]["idx_tag"][0]
            str_std = "std" if "std" in args.work_dir else "render"
            inspect_results(
                input_imgs,
                data,
                result_dict,
                tag=f"{cfg.dataset_tag}/{cfg.model_tag}_{str_std}/{idx_tag}",
            )
            print(f"Done: {i_iter_val}")


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--py-config", required=True)
    parser.add_argument("--work-dir", type=str, default="./out/debug")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",  # Accept one or more integers
        help="List of indices to process (e.g., 1 2 3)",
    )
    parser.add_argument("--no-strict-state", action="store_false", default=True)
    parser.add_argument(
        "--dataset", choices=["occ3d", "surroundocc", "kitti360"], required=True
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()

    main(0, args)
