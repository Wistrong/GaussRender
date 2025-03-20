import os
import os.path as osp
import pdb
import re
import time

import torch
import torch.distributed as dist
from mmengine.logging import MMLogger
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor


def refine_load_from_sd(sd):
    for k in list(sd.keys()):
        if "img_neck." in k:
            del sd[k]
    return sd


def filter_unused_parameters(model, model_tag):
    unused_params_file = f"config/{model_tag}/unused.txt"

    try:
        with open(unused_params_file, "r") as file:
            unused_params = [line.strip()[1:-1] for line in file.readlines()]
    except (FileNotFoundError, ValueError, SyntaxError) as e:
        print(f"Error reading {unused_params_file}: {e}")
        return []

    # Disable the parameters listed in the file
    for name, param in model.named_parameters():
        if name in unused_params:
            param.requires_grad = False  # Disable gradients for this parameter
            # print(f"Disabling {name} (requires_grad=False)")
    return


def set_seed(seed):
    set_random_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def set_working_dir(cfg, args):
    os.makedirs(args.work_dir, exist_ok=True)
    version_dirs = [
        int(re.search(r"\d+", d).group())
        for d in os.listdir(args.work_dir)
        if os.path.isdir(os.path.join(args.work_dir, d)) and re.match(r"version_\d+", d)
    ]
    next_version = max(version_dirs, default=0) + 1
    cfg.work_dir = os.path.join(args.work_dir, f"version_{next_version}")
    os.makedirs(cfg.work_dir, exist_ok=True)


def pass_print(*args, **kwargs):
    pass


def init_ddp_env(local_rank, gpus, cfg):
    if gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        if local_rank == 0:
            print(f"World Size: {hosts}, number of GPUs: {gpus}")
            print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{ip}:{port}",
            world_size=hosts * gpus,
            rank=rank * gpus + local_rank,
        )
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)

        if local_rank != 0:
            import builtins

            builtins.print = pass_print
    else:
        distributed = False
    return distributed


def init_model(local_rank, distributed, cfg, logger):
    model = build_segmentor(cfg.model)
    filter_unused_parameters(model, cfg.model_tag)
    model.init_weights()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of params: {n_parameters}")
    if distributed:
        if cfg.get("syncBN", True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info("converted sync bn.")

        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        model = ddp_model_module(
            model.cuda(local_rank),
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get("find_unused_parameters", False),
        )
        raw_model = model.module
        logger.info("done ddp model")
    else:
        model = model.cuda()
        raw_model = model
    return model, raw_model


def init_logger(cfg, log_file=None):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if log_file is None:
        log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = MMLogger("selfocc", log_file=log_file)
    MMLogger._instance_dict["selfocc"] = logger
    return logger


def resume_from_checkpoint(
    cfg,
    logger,
    epoch,
    global_iter,
    last_iter,
    raw_model,
    optimizer=None,
    scheduler=None,
    train_dataset_loader=None,
    strict_state=True,
):
    if cfg.resume_from:
        if osp.exists(cfg.resume_from):
            ckpt = torch.load(cfg.resume_from, map_location="cpu")
            if "state_dict" in ckpt.keys():
                raw_model.load_state_dict(ckpt["state_dict"], strict=strict_state)
            else:
                raw_model.load_state_dict(ckpt, strict=True)
            if "optimizer" in ckpt.keys() and optimizer is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt.keys() and scheduler is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "epoch" in ckpt.keys():
                epoch = ckpt["epoch"]
            if "global_iter" in ckpt.keys():
                global_iter = ckpt["global_iter"]
            if "last_iter" in ckpt:
                last_iter = ckpt["last_iter"] if "last_iter" in ckpt else 0
            if train_dataset_loader is not None and hasattr(
                train_dataset_loader.sampler, "set_last_iter"
            ):
                train_dataset_loader.sampler.set_last_iter(last_iter)
            logger.info(f"successfully resumed from epoch {epoch}")
        else:
            exit("No checkpoint found. Please check your config.")
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location="cpu")
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        try:
            raw_model.load_state_dict(state_dict, strict=True)
        except:
            raw_model.load_state_dict(refine_load_from_sd(state_dict), strict=False)
    return epoch, global_iter, last_iter
