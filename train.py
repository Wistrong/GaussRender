import argparse
import glob
import os
import os.path as osp
import pdb
import time
import warnings

import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.optim import build_optim_wrapper
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler

from dataset import get_dataloader
from src.loss import OPENOCC_LOSS
from src.model.utils.utils import (
    get_gpu_memory_usage,
    get_loss,
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

    # Setup DDP env
    distributed = init_ddp_env(local_rank, args.gpus, cfg)

    # global settings
    set_seed(args.seed)

    if local_rank == 0:
        # Setup working dirs
        set_working_dir(cfg, args)
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.py_config)))

        # Setup logger
        from src.utils.logger import WrappedTBWriter

        writer = WrappedTBWriter("selfocc", log_dir=osp.join(cfg.work_dir, "tf"))
        WrappedTBWriter._instance_dict["selfocc"] = writer
    else:
        writer = None

    # Setup printing
    logger = init_logger(cfg)
    logger.info("Working dir: " + cfg.work_dir)
    logger.info(f"Config:\n{cfg.pretty_text}")

    # Building model
    model, raw_model = init_model(local_rank, distributed, cfg, logger)
    model.train()

    # Building dataloader
    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=args.iter_resume,
    )

    # get optimizer, loss, scheduler
    optimizer = build_optim_wrapper(model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    if local_rank == 0:
        loss_func.writer = writer
    max_num_epochs = cfg.max_epochs
    if cfg.get("multisteplr", False):
        scheduler = MultiStepLRScheduler(optimizer, **cfg.multisteplr_config)
    else:
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_dataset_loader) * max_num_epochs,
            lr_min=cfg.optimizer["optimizer"]["lr"] * 1e-3,
            warmup_t=cfg.get("warmup_iters", 500),
            warmup_lr_init=cfg.optimizer["optimizer"]["lr"] / 3,
            t_in_epochs=False,
        )

    # resume and load
    epoch = 0
    global_iter = 0
    last_iter = 0

    # Resuming
    cfg.resume_from = ""
    if args.resume_from:
        cfg.resume_from = args.resume_from

    epoch, global_iter, last_iter = resume_from_checkpoint(
        cfg,
        logger,
        epoch,
        global_iter,
        last_iter,
        raw_model,
        optimizer,
        scheduler,
        train_dataset_loader,
    )
    logger.info("load from: " + cfg.load_from)
    logger.info("resume from: " + cfg.resume_from)

    # training
    print_freq = cfg.print_freq
    first_run = True

    miou_metric, _, _, _ = set_up_metrics(args.dataset)
    best_miou = float("-inf")

    if cfg.inspect > 0:
        os.makedirs(
            os.path.join(os.path.abspath(cfg.work_dir), "inspect"), exist_ok=True
        )

    while epoch < max_num_epochs:
        os.environ["eval"] = "false"
        if hasattr(train_dataset_loader.sampler, "set_epoch"):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_list = []
        time.sleep(2)
        data_time_s = time.time()
        time_s = time.time()

        # Training
        for i_iter, data in enumerate(train_dataset_loader):
            if first_run:
                i_iter = i_iter + last_iter

            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop("img")
            data_time_e = time.time()

            # Render
            if "render" in cfg.loss_input_convertion.keys():
                set_matrix_to_render(distributed, model, data)

            # forward + backward + optimize
            inspect = cfg.inspect
            result_dict = model(imgs=input_imgs, metas=data, global_iter=global_iter)

            # Render
            if "render" in cfg.loss_input_convertion.keys():
                set_gt_render(distributed, model, data, result_dict["render"]["valid"])

            # Inspect output
            if inspect:
                inspect_results(input_imgs, data, result_dict)
                exit(0)

            # Loss
            loss, loss_dict = get_loss(
                result_dict, data, loss_func, cfg, cfg.out_occ_shapes
            )

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.detach().cpu().item())
            scheduler.step_update(global_iter)
            time_e = time.time()

            # Log images
            global_iter += 1

            # Log bar
            if i_iter % print_freq == 0 and local_rank == 0:
                # Log gaussians
                if writer is not None and "render" in result_dict.keys():
                    writer.add_scalar(
                        "Gaussians/Opacity",
                        result_dict["render"]["gaussians"].opacities.mean(),
                        global_iter,
                    )

                lr = optimizer.param_groups[0]["lr"]
                # Inside your logger code
                reserved_memory = get_gpu_memory_usage()

                logger.info(
                    "[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), lr: %.7f, "
                    "time: %.3f (%.3f), GPU Memory Reserved: %.2fMB"
                    % (
                        epoch,
                        i_iter,
                        len(train_dataset_loader),
                        loss.item(),
                        np.mean(loss_list),
                        lr,
                        time_e - time_s,
                        data_time_e - data_time_s,
                        reserved_memory,
                    )
                )
                if writer is not None:
                    writer.add_scalar(
                        "Optimization/Lr", optimizer.param_groups[-1]["lr"], global_iter
                    )
                detailed_loss = []
                for loss_name, loss_value in loss_dict.items():
                    detailed_loss.append(f"{loss_name}: {loss_value:.5f}")
                detailed_loss = ", ".join(detailed_loss)
                logger.info(detailed_loss)
                loss_list = []
            data_time_s = time.time()
            time_s = time.time()

        # Eval
        if epoch % cfg.get("eval_every_epochs", 1) != 0:
            continue
        model.eval()
        os.environ["eval"] = "true"
        val_loss_list = []

        with torch.no_grad():
            for i_iter_val, data in enumerate(val_dataset_loader):
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
                    set_gt_render(
                        distributed, model, data, result_dict["render"]["valid"]
                    )

                loss, loss_dict = get_loss(
                    result_dict, data, loss_func, cfg, cfg.out_occ_shapes
                )

                # Calculate IoU
                pred_occ = result_dict["pred_occ"][-1].argmax(1)

                # Post process for only cam training.
                if not cfg.loss.w_occupancy:
                    B = pred_occ.shape[0]
                    for b in range(B):
                        op_thresh = (
                            result_dict["render"]["gaussians"].opacities[b].squeeze(-1)
                            < 0.1
                        )
                        if cfg.dataset_tag in ["surroundocc", "kitti360"]:
                            empty_class = 0
                        elif cfg.dataset_tag == "occ3d":
                            empty_class = 17
                        pred_occ[b][op_thresh] = empty_class
                gt_occ = data["occ_label"].flatten(start_dim=1)

                occ_cam_mask = data["occ_cam_mask"].flatten(start_dim=1)
                miou_metric._after_step(pred_occ, gt_occ, occ_cam_mask)

                val_loss_list.append(loss.detach().cpu().numpy())
                if i_iter_val % print_freq == 0 and local_rank == 0:
                    logger.info(
                        "[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)"
                        % (epoch, i_iter_val, loss.item(), np.mean(val_loss_list))
                    )
                    detailed_loss = []
                    for loss_name, loss_value in loss_dict.items():
                        detailed_loss.append(f"{loss_name}: {loss_value:.5f}")
                    detailed_loss = ", ".join(detailed_loss)
                    logger.info(detailed_loss)

        logger.info("Current val loss is %.3f" % (np.mean(val_loss_list)))

        miou, iou = miou_metric._after_epoch(logger)
        miou_metric.reset()

        # Save to TensorBoard
        if writer is not None:
            writer.add_scalar("Validation/mIoU", miou, epoch)
            writer.add_scalar("Validation/IoU", iou, epoch)

        # save checkpoint
        current_val_miou = miou
        if local_rank == 0:
            # Create the dictionary to save
            dict_to_save = {
                "state_dict": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "global_iter": global_iter,
            }

            # Save the last checkpoint
            last_checkpoint_file = os.path.join(cfg.work_dir, "last.pth")
            torch.save(dict_to_save, last_checkpoint_file)

            # Save the checkpoint if val/IoU is higher than the previous best
            if current_val_miou > best_miou:
                # Delete previous best model if it exists
                old_best_files = glob.glob(os.path.join(cfg.work_dir, "best_*.pth"))
                for old_file in old_best_files:
                    os.remove(old_file)

                # Update best IoU and save new checkpoint
                new_best_checkpoint_file = os.path.join(
                    cfg.work_dir, f"best_{current_val_miou:.2f}.pth"
                )
                best_miou = current_val_miou
                torch.save(dict_to_save, new_best_checkpoint_file)

        epoch += 1
        first_run = False

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--py-config", default="config/tpv_lidarseg.py")
    parser.add_argument("--work-dir", type=str, default="./out/tpv_lidarseg")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--iter-resume", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset", choices=["occ3d", "surroundocc", "kitti360"], required=True
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=ngpus)
    else:
        main(0, args)
