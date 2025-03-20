# Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
import argparse
import copy
import glob
import math
import os
import pdb

import mmengine
import numpy as np
import torch
from prettytable import PrettyTable
from pyquaternion import Quaternion
from torch.utils.cpp_extension import load
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# fmt: off
occ3d_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free']

openocc_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free']
# fmt: on


np.set_printoptions(precision=3, suppress=True)


def trans_matrix(T, R):
    tm = np.eye(4)
    tm[:3, :3] = R.rotation_matrix
    tm[:3, 3] = T
    return tm


# A helper dataset for RayIoU. It is NOT used during training.
class EgoPoseDataset(Dataset):
    def __init__(self, data_infos):
        super(EgoPoseDataset, self).__init__()

        self.data_infos = data_infos
        self.scene_frames = {}

        for info in data_infos:
            scene_name = info["scene_name"]
            if scene_name not in self.scene_frames:
                self.scene_frames[scene_name] = []
            self.scene_frames[scene_name].append(info)

    def __len__(self):
        return len(self.data_infos)

    def get_ego_from_lidar(self, info):
        ego_from_lidar = trans_matrix(
            np.array(info["lidar2ego_translation"]),
            Quaternion(info["lidar2ego_rotation"]),
        )
        return ego_from_lidar

    def get_global_pose(self, info, inverse=False):
        global_from_ego = trans_matrix(
            np.array(info["ego2global_translation"]),
            Quaternion(info["ego2global_rotation"]),
        )
        ego_from_lidar = trans_matrix(
            np.array(info["lidar2ego_translation"]),
            Quaternion(info["lidar2ego_rotation"]),
        )
        pose = global_from_ego.dot(ego_from_lidar)
        if inverse:
            pose = np.linalg.inv(pose)
        return pose

    def __getitem__(self, idx):
        info = self.data_infos[idx]

        ref_sample_token = info["token"]
        ref_lidar_from_global = self.get_global_pose(info, inverse=True)
        ref_ego_from_lidar = self.get_ego_from_lidar(info)

        scene_frame = self.scene_frames[info["scene_name"]]
        ref_index = scene_frame.index(info)

        # NOTE: getting output frames
        output_origin_list = []
        for curr_index in range(len(scene_frame)):
            # if this exists a valid target
            if curr_index == ref_index:
                origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                # transform from the current lidar frame to global and then to the reference lidar frame
                global_from_curr = self.get_global_pose(
                    scene_frame[curr_index], inverse=False
                )
                ref_from_curr = ref_lidar_from_global.dot(global_from_curr)
                origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)

            origin_tf_pad = np.ones([4])
            origin_tf_pad[:3] = origin_tf  # pad to [4]
            origin_tf = np.dot(ref_ego_from_lidar[:3], origin_tf_pad.T).T  # [3]

            # origin
            if np.abs(origin_tf[0]) < 39 and np.abs(origin_tf[1]) < 39:
                output_origin_list.append(origin_tf)

        # select 8 origins
        if len(output_origin_list) > 8:
            select_idx = np.round(
                np.linspace(0, len(output_origin_list) - 1, 8)
            ).astype(np.int64)
            output_origin_list = [output_origin_list[i] for i in select_idx]

        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))  # [T, 3]

        return (ref_sample_token, output_origin_tensor)


dvr = load(
    "dvr",
    sources=["extensions/dvr/dvr.cpp", "extensions/dvr/dvr.cu"],
    verbose=True,
    extra_cuda_cflags=["-allow-unsupported-compiler"],
)
_pc_range = [-40, -40, -1.0, 40, 40, 5.4]
_voxel_size = 0.4


# https://github.com/tarashakhurana/4d-occ-forecasting/blob/ff986082cd6ea10e67ab7839bf0e654736b3f4e2/test_fgbg.py#L29C1-L46C16
def get_rendered_pcds(origin, points, tindex, pred_dist):
    pcds = []

    for t in range(len(origin)):
        mask = tindex == t
        # skip the ones with no data
        if not mask.any():
            continue
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v**2).sum(axis=1, keepdims=True))
        pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
        pcds.append(torch.from_numpy(pred_pts))

    return pcds


def generate_lidar_rays():
    # prepare lidar ray angles
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)

    # nuscenes lidar fov: [0.2107773983152201, -0.5439104895672159] (rad)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    lidar_rays = []
    for pitch_angle in pitch_angles:
        for azimuth_angle in np.arange(0, 360, 1):
            azimuth_angle = np.deg2rad(azimuth_angle)

            x = np.cos(pitch_angle) * np.cos(azimuth_angle)
            y = np.cos(pitch_angle) * np.sin(azimuth_angle)
            z = np.sin(pitch_angle)

            lidar_rays.append((x, y, z))

    return np.array(lidar_rays, dtype=np.float32)


def process_one_sample(
    sem_pred, lidar_rays, output_origin, instance_pred=None, occ_class_names=None
):
    # lidar origin in ego coordinate
    # lidar_origin = torch.tensor([[[0.9858, 0.0000, 1.8402]]])
    T = output_origin.shape[1]
    pred_pcds_t = []

    free_id = len(occ_class_names) - 1
    occ_pred = copy.deepcopy(sem_pred)
    occ_pred[sem_pred < free_id] = 1
    occ_pred[sem_pred == free_id] = 0
    occ_pred = occ_pred.permute(2, 1, 0)
    occ_pred = occ_pred[None, None, :].contiguous().float()

    offset = torch.Tensor(_pc_range[:3])[None, None, :]
    scaler = torch.Tensor([_voxel_size] * 3)[None, None, :]

    lidar_tindex = torch.zeros([1, lidar_rays.shape[0]])

    for t in range(T):
        lidar_origin = output_origin[:, t : t + 1, :]  # [1, 1, 3]
        lidar_endpts = lidar_rays[None] + lidar_origin  # [1, 15840, 3]

        output_origin_render = ((lidar_origin - offset) / scaler).float()  # [1, 1, 3]
        output_points_render = ((lidar_endpts - offset) / scaler).float()  # [1, N, 3]
        output_tindex_render = lidar_tindex  # [1, N], all zeros

        with torch.no_grad():
            pred_dist, _, coord_index = dvr.render_forward(
                occ_pred.cuda(),
                output_origin_render.cuda(),
                output_points_render.cuda(),
                output_tindex_render.cuda(),
                [1, 16, 200, 200],
                "test",
            )
            pred_dist *= _voxel_size

        pred_pcds = get_rendered_pcds(
            lidar_origin[0].cpu().numpy(),
            lidar_endpts[0].cpu().numpy(),
            lidar_tindex[0].cpu().numpy(),
            pred_dist[0].cpu().numpy(),
        )
        coord_index = coord_index[0, :, :].int().cpu()  # [N, 3]

        pred_label = sem_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][
            :, None
        ]  # [N, 1]
        pred_dist = pred_dist[0, :, None].cpu()

        if instance_pred is not None:
            pred_instance = instance_pred[
                coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]
            ][
                :, None
            ]  # [N, 1]
            pred_pcds = torch.cat(
                [pred_label.float(), pred_instance.float(), pred_dist], dim=-1
            )
        else:
            pred_pcds = torch.cat([pred_label.float(), pred_dist], dim=-1)

        pred_pcds_t.append(pred_pcds)

    pred_pcds_t = torch.cat(pred_pcds_t, dim=0)

    return pred_pcds_t.numpy()


def calc_rayiou(pcd_pred_list, pcd_gt_list, occ_class_names):
    thresholds = [1, 2, 4]

    gt_cnt = np.zeros([len(occ_class_names)])
    pred_cnt = np.zeros([len(occ_class_names)])
    tp_cnt = np.zeros([len(thresholds), len(occ_class_names)])

    for pcd_pred, pcd_gt in zip(pcd_pred_list, pcd_gt_list):
        for j, threshold in enumerate(thresholds):
            # L1
            depth_pred = pcd_pred[:, 1]
            depth_gt = pcd_gt[:, 1]
            l1_error = np.abs(depth_pred - depth_gt)
            tp_dist_mask = l1_error < threshold

            for i, cls in enumerate(occ_class_names):
                cls_id = occ_class_names.index(cls)
                cls_mask_pred = pcd_pred[:, 0] == cls_id
                cls_mask_gt = pcd_gt[:, 0] == cls_id

                gt_cnt_i = cls_mask_gt.sum()
                pred_cnt_i = cls_mask_pred.sum()
                if j == 0:
                    gt_cnt[i] += gt_cnt_i
                    pred_cnt[i] += pred_cnt_i

                tp_cls = cls_mask_gt & cls_mask_pred  # [N]
                tp_mask = np.logical_and(tp_cls, tp_dist_mask)
                tp_cnt[j][i] += tp_mask.sum()

    iou_list = []
    for j, threshold in enumerate(thresholds):
        iou_list.append((tp_cnt[j] / (gt_cnt + pred_cnt - tp_cnt[j]))[:-1])

    return iou_list


def main_rayiou(sem_pred_list, sem_gt_list, lidar_origin_list, occ_class_names):
    torch.cuda.empty_cache()

    # generate lidar rays
    lidar_rays = generate_lidar_rays()
    lidar_rays = torch.from_numpy(lidar_rays)

    pcd_pred_list, pcd_gt_list = [], []
    for sem_pred, sem_gt, lidar_origins in tqdm(
        zip(sem_pred_list, sem_gt_list, lidar_origin_list), ncols=50
    ):
        sem_pred = torch.from_numpy(np.reshape(sem_pred, [200, 200, 16]))
        sem_gt = torch.from_numpy(np.reshape(sem_gt, [200, 200, 16]))

        pcd_pred = process_one_sample(
            sem_pred, lidar_rays, lidar_origins, occ_class_names=occ_class_names
        )
        pcd_gt = process_one_sample(
            sem_gt, lidar_rays, lidar_origins, occ_class_names=occ_class_names
        )

        # evalute on non-free rays
        valid_mask = pcd_gt[:, 0].astype(np.int32) != len(occ_class_names) - 1
        pcd_pred = pcd_pred[valid_mask]
        pcd_gt = pcd_gt[valid_mask]

        assert pcd_pred.shape == pcd_gt.shape
        pcd_pred_list.append(pcd_pred)
        pcd_gt_list.append(pcd_gt)

    iou_list = calc_rayiou(pcd_pred_list, pcd_gt_list, occ_class_names)
    rayiou = np.nanmean(iou_list)
    rayiou_0 = np.nanmean(iou_list[0])
    rayiou_1 = np.nanmean(iou_list[1])
    rayiou_2 = np.nanmean(iou_list[2])

    table = PrettyTable(["Class Names", "RayIoU@1", "RayIoU@2", "RayIoU@4"])
    table.float_format = ".3"

    for i in range(len(occ_class_names) - 1):
        table.add_row(
            [occ_class_names[i], iou_list[0][i], iou_list[1][i], iou_list[2][i]],
            divider=(i == len(occ_class_names) - 2),
        )

    table.add_row(["MEAN", rayiou_0, rayiou_1, rayiou_2])

    print(table)

    torch.cuda.empty_cache()

    return {
        "RayIoU": rayiou,
        "RayIoU@1": rayiou_0,
        "RayIoU@2": rayiou_1,
        "RayIoU@4": rayiou_2,
    }, table


def main(args):
    data_infos = mmengine.load(os.path.join(args.data_root, "nuscenes_infos_val.pkl"))[
        "infos"
    ]
    gt_filepaths = sorted(
        glob.glob(os.path.join(args.data_root, args.data_type, "*/*/*.npz"))
    )

    gt_path_root = "data/gts"
    gt_filepaths = sorted(glob.glob(os.path.join(gt_path_root, "*/*/*.npz")))

    # retrieve scene_name
    token2scene = {}
    for gt_path in gt_filepaths:
        token = gt_path.split("/")[-2]
        scene_name = gt_path.split("/")[-3]
        token2scene[token] = scene_name

    for i in range(len(data_infos)):
        scene_name = token2scene[data_infos[i]["token"]]
        data_infos[i]["scene_name"] = scene_name

    lidar_origins = []
    occ_gts = []
    occ_preds = []

    for idx, batch in enumerate(DataLoader(EgoPoseDataset(data_infos), num_workers=8)):
        output_origin = batch[1]
        info = data_infos[idx]
        occ_path = os.path.join(
            gt_path_root, info["scene_name"], info["token"], "labels.npz"
        )
        occ_gt = np.load(occ_path, allow_pickle=True)["semantics"]
        occ_gt = np.reshape(occ_gt, [200, 200, 16]).astype(np.uint8)

        occ_path = os.path.join(args.pred_dir, info["token"] + ".npz")
        occ_pred = np.load(occ_path, allow_pickle=True)["arr_0"]
        occ_pred = np.reshape(occ_pred, [200, 200, 16]).astype(np.uint8)

        lidar_origins.append(output_origin)
        occ_gts.append(occ_gt)
        occ_preds.append(occ_pred)

    if args.data_type == "occ3d":
        occ_class_names = occ3d_class_names
    elif args.data_type == "openocc_v2":
        occ_class_names = openocc_class_names
    else:
        raise ValueError

    result, table = main_rayiou(
        occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names
    )

    log_path = os.path.split(os.path.split(args.pred_dir)[0])[0]

    with open(os.path.join(log_path, "log.txt"), "a") as f:
        f.writelines(str(result) + "\n")
        # pdb.set_trace()
        table = f"metric table:{table}"
        f.writelines(table + "\n")

    return (result, table)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/nuscenes")
    parser.add_argument("--pred_dir", type=str, default="")
    parser.add_argument(
        "--data-type", type=str, choices=["occ3d", "openocc_v2"], default="occ3d"
    )
    args = parser.parse_args()

    torch.random.manual_seed(0)
    np.random.seed(0)

    main(args)
