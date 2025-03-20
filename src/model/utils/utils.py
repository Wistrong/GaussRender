import pdb

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.metrics import L1_Metric, MeanIoU


def modify_file_inplace(file_path, dataset_type):
    # Allow to dynamically change the dataset.
    with open(file_path, "w") as f:
        if dataset_type == "occ3d":
            f.write('_base_ = ["dataset/occ3d.py"]\n')
        elif dataset_type == "surroundocc":
            f.write('_base_ = ["dataset/surroundocc.py"]\n')
        elif dataset_type == "kitti360":
            f.write('_base_ = ["dataset/kitti-360.py"]\n')

        else:
            raise ValueError("Unsupported dataset type!")
    return


def get_frequencies(dataset):
    # fmt: off
    if dataset == "occ3d":
        # From GaussianFormer
        manual_class_weight = torch.tensor([
            1.01552756,1.06897009,1.30013094, 1.07253735,0.94637502,1.10087012,
            1.26960524,1.06258364,1.189019,   1.06217292,1.00595144,0.85706115,
            1.03923299,0.90867526,0.8936431,  0.85486129,0.8527829,0.5])
    elif dataset == "surroundocc":
        # From GaussianFormer
        # empty is the zero class, 17 does not exist.
        manual_class_weight = torch.tensor([
            0.5,1.06897009,1.30013094, 1.07253735,0.94637502,1.10087012,
            1.26960524,1.06258364,1.189019,   1.06217292,1.00595144,0.85706115,
            1.03923299,0.90867526,0.8936431,  0.85486129,0.8527829])
    elif dataset == "kitti360":
        # ? https://github.com/ai4ce/SSCBench/blob/f6828ebe7f87953b0e006f10de19e85eeb1eceee/method/MonoScene/monoscene/data/kitti_360/params.py#L3
        kitti_360_class_frequencies = np.array([
            2264087502,20098728,104972,96297,1149426,4051087,125103,105540713,
            16292249,45297267,14454132,110397082,6766219,295883213,50037503,
            1561069,406330,30516166,1950115])
        # ? https://github.com/ai4ce/SSCBench/blob/f6828ebe7f87953b0e006f10de19e85eeb1eceee/method/MonoScene/monoscene/scripts/train_monoscene.py#L69
        manual_class_weight = torch.from_numpy(1 / np.log(kitti_360_class_frequencies + 0.001))
    else:
        raise NotImplementedError(f"Not implemented for {dataset}")
    # fmt: on
    return manual_class_weight


def modify_config_based_on_dataset(cfg, dataset):
    for loss_cfg in cfg.loss.loss_cfgs:
        if loss_cfg["type"] == "OccupancyLoss":
            loss_cfg["num_classes"] = cfg.num_classes
            # Add 0 to have 19 classes.
            manual_class_weight = get_frequencies(dataset)
            loss_cfg["manual_class_weight"] = manual_class_weight
    return cfg


@torch.no_grad()
def set_matrix_to_render(distributed, my_model, data):
    if distributed:
        renderer = my_model.module.renderer
    else:
        renderer = my_model.renderer
    data.update({"render": renderer.get_render_matrix(data)})


@torch.no_grad()
def set_gt_render(distributed, my_model, data, dict_valid):
    if distributed:
        renderer = my_model.module.renderer
    else:
        renderer = my_model.renderer
    (render_cam, render_cam_depth, render_bev, render_bev_depth) = renderer.render_gt(
        data, dict_valid
    )

    data.update({"render_cam": render_cam})
    data.update({"render_cam_depth": render_cam_depth})
    data.update({"render_bev": render_bev})
    data.update({"render_bev_depth": render_bev_depth})


def set_up_metrics(dataset_tag):
    # fmt: off
    if dataset_tag == "occ3d":
        list_cls = ["other","barrier","bicycle","bus","car","construction_vehicle",
                "motorcycle","pedestrian","traffic_cone","trailer","truck","driveable_surface",
                "other_flat","sidewalk","terrain","manmade","vegetation", "empty"]
        # 17 is the empty class in Occ3d-Nusc.
        empty_label = 17
    elif dataset_tag == "surroundocc":
        list_cls = ["empty","barrier","bicycle","bus","car","construction_vehicle",
                "motorcycle","pedestrian","traffic_cone","trailer","truck","driveable_surface",
                "other_flat","sidewalk","terrain","manmade","vegetation"]
        # 0 is the empty class in SurroundOcc
        empty_label = 0
    elif dataset_tag == "kitti360":
        list_cls =  ["empty","car","bicycle","motorcycle","truck","other-vehicle",
                     "person","road","parking","sidewalk","other-ground","building",
                     "fence","vegetation","terrain","pole","traffic-sign","other-structure",
                     "other-object"]
        # 0 is the empty class in KITTI360
        empty_label = 0
        
    # 3D
    miou_metric = MeanIoU(empty_label=empty_label,label_str=list_cls)
    miou_metric.reset()

    # Rendering
    miou_metric_img = MeanIoU(empty_label=empty_label,label_str=list_cls)
    miou_metric_img.reset()
    depth_metric = L1_Metric()
    depth_metric.reset()
    
    # BeV
    miou_metric_bev = MeanIoU(empty_label=empty_label,label_str=list_cls)
    miou_metric_bev.reset()

    # fmt: on
    return miou_metric, miou_metric_img, depth_metric, miou_metric_bev


# Function to format memory usage nicely
def get_gpu_memory_usage():
    reserved_memory = torch.cuda.memory_reserved() / (1024 * 1024)  # In MB
    return reserved_memory


def get_loss(result_dict, data, loss_func, cfg, out_occ_shapes):
    # ! https://github.com/pmj110119/RenderOcc/issues/30
    loss_input = {
        "occ_label": data["occ_label"],
        # ! No masking during training, can be used during evaluation.
        "occ_mask": (data["occ_label"] != 255).bool(),
        "out_occ_shapes": out_occ_shapes,
        "voxel_size": result_dict["voxel_size"],
        "pc_range": result_dict["pc_range"],
    }
    if "render" in cfg.loss_input_convertion.keys():
        loss_input.update(
            {
                "render_cam": data["render_cam"],
                "render_cam_depth": data["render_cam_depth"],
                "render_bev": data["render_bev"],
                "render_bev_depth": data["render_bev_depth"],
            }
        )
    for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
        if loss_input_val in result_dict.keys():
            loss_input.update({loss_input_key: result_dict[loss_input_val]})
    loss, loss_dict = loss_func(loss_input)
    return loss, loss_dict
