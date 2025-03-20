import argparse
import json
import os
import os.path as osp
import pdb
import warnings

import numpy as np
import torch
from mmengine import Config, DictAction
from tqdm import tqdm

from dataset import get_dataloader
from src.model.utils.utils import (
    modify_config_based_on_dataset,
    modify_file_inplace,
    set_gt_render,
    set_matrix_to_render,
    set_up_metrics,
)
from src.utils.setup import (
    init_ddp_env,
    init_logger,
    init_model,
    resume_from_checkpoint,
    set_seed,
    set_working_dir,
)
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

    # init DDP
    distributed = init_ddp_env(local_rank, args.gpus, cfg)

    # global settings
    set_seed(args.seed)
    set_working_dir(cfg, args)
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.py_config)))

    # Setup printing
    logger = init_logger(cfg)
    logger.info("Working dir: " + cfg.work_dir)
    logger.info(f"Config:\n{cfg.pretty_text}")

    # build model
    model, raw_model = init_model(local_rank, distributed, cfg, logger)
    model.eval()

    _, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.test_dataset_config,
        cfg.train_loader,
        cfg.test_loader,
        dist=distributed,
        val_only=True,
    )

    epoch = 0
    global_iter = 0
    last_iter = 0

    # resume and load
    cfg.resume_from = ""
    if osp.exists(osp.join(args.work_dir, "latest.pth")):
        cfg.resume_from = osp.join(args.work_dir, "latest.pth")
    if args.resume_from:
        cfg.resume_from = args.resume_from

    logger.info("resume from: " + cfg.resume_from)
    logger.info("work dir: " + args.work_dir)

    epoch, global_iter, last_iter = resume_from_checkpoint(
        cfg,
        logger,
        epoch,
        global_iter,
        last_iter,
        raw_model,
        strict_state=args.no_strict_state,
    )
    logger.info("load from: " + cfg.load_from)
    logger.info("resume from: " + cfg.resume_from)

    # eval
    print_freq = cfg.print_freq

    miou_metric, miou_metric_img, depth_metric, miou_metric_bev = set_up_metrics(
        args.dataset
    )
    os.environ["eval"] = "true"
    mIoU_results = []

    if cfg.inspect > 0:
        os.makedirs(
            os.path.join(os.path.abspath(cfg.work_dir), "inspect"), exist_ok=True
        )

    with torch.no_grad():
        for i_iter_val, data in enumerate(tqdm(val_dataset_loader)):

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

            pred_occ = result_dict["pred_occ"][-1].argmax(1)
            gt_occ = data["occ_label"].flatten(start_dim=1)

            occ_cam_mask = data["occ_cam_mask"].flatten(start_dim=1)
            miou_metric._after_step(pred_occ, gt_occ, occ_cam_mask)

            bool_with_render = False
            bool_with_depth = False
            if "render" in result_dict.keys():
                if result_dict["render"]["cam"] is not None:
                    bool_with_render = True
                    miou_metric_img._after_step(
                        result_dict["render"]["cam"].argmax(-1).flatten(),
                        data["render_cam"].argmax(-1).flatten(),
                    )
                if result_dict["render"]["depth"] is not None:
                    bool_with_depth = True
                    mask_cam_depth = (data["render_cam_depth"] > 0.1).flatten()
                    depth_metric._after_step(
                        result_dict["render"]["depth"].flatten(),
                        data["render_cam_depth"].flatten(),
                        mask_cam_depth,
                    )
                if result_dict["render"]["bev"] is not None:
                    bool_with_bev = True
                    miou_metric_bev._after_step(
                        result_dict["render"]["bev"].argmax(-1).flatten(),
                        data["render_bev"].argmax(-1).flatten(),
                    )

            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info("[EVAL] Iter %5d" % (i_iter_val))

            if i_iter_val % 100 == 0 and i_iter_val > 0 and args.short:
                break

            # Inspect output
            if cfg.inspect > 0:
                idx_tag = result_dict["metas"]["idx_tag"][0]
                inspect_results(
                    input_imgs,
                    data,
                    result_dict,
                    tag=f"{cfg.model_tag}/{cfg.dataset_tag}/{idx_tag}",
                )
                if i_iter_val > cfg.inspect - 1:
                    exit(0)
    if bool_with_render:
        print("\n" * 2)
        logger.info("\nIoU img:")
        miou_metric_img._after_epoch(logger)
        miou_metric_img.reset()

    if bool_with_depth:
        print("\n" * 2)
        l1depth = depth_metric._after_epoch(logger)
        logger.info(f"L1 Depth: {l1depth}")
        depth_metric.reset()

    if bool_with_bev:
        print("\n" * 2)
        logger.info("IoU BeV:")
        miou_metric_bev._after_epoch(logger)
        miou_metric_bev.reset()

    print("\n" * 2)
    logger.info("IoU 3D:")
    miou_metric._after_epoch(logger)
    miou_metric.reset()

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--py-config", required=True)
    parser.add_argument("--work-dir", type=str, default="./out/debug")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument(
        "--dataset", choices=["occ3d", "surroundocc", "kitti360"], required=True
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    # Eval options
    parser.add_argument("--no-strict-state", action="store_false", default=True)
    parser.add_argument("--short", action="store_true")
    args = parser.parse_args()

    if args.gpus is None:
        ngpus = torch.cuda.device_count()
    else:
        ngpus = args.gpus
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)

# python eval.py --py-config config/surroundocc/render.py --work-dir out/debug/ --dataset occ3d --resume-from ckpts/final/occ3d_surr_std.pth --cfg-options model.head.render_kwargs.render_gt_mode=sensor model.head.render_kwargs.cam_idx="[0,1,2,3,4,5]" model.head.pre_render_kwargs.overwrite_opacity=True --no-strict-state --save-list-iou
