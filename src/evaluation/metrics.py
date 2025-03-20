import pdb

import numpy as np
import torch
import torch.distributed as dist


class MeanIoU:
    def __init__(
        self,
        label_str,
        empty_label,
        ignore_label=255,
        name="none",
    ):
        self.label_str = label_str
        self.num_classes = len(label_str)
        self.ignore_label = ignore_label
        self.empty_label = empty_label
        self.name = name

    def reset(self) -> None:
        # Add +1 for non empty.
        self.total_seen = torch.zeros(self.num_classes + 1).cuda()
        self.total_correct = torch.zeros(self.num_classes + 1).cuda()
        self.total_positive = torch.zeros(self.num_classes + 1).cuda()

    def _after_step(self, outputs, targets, mask=None):
        if mask is not None:
            outputs = outputs[mask]
            targets = targets[mask]

        # Remove ignore label, i.e 255.
        # In both SurroundOcc and Occ3d we ignore 255.
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        for i in range(self.num_classes):
            self.total_seen[i] += torch.sum(targets == i).item()
            self.total_correct[i] += torch.sum((targets == i) & (outputs == i)).item()
            self.total_positive[i] += torch.sum(outputs == i).item()

            if i == self.empty_label:
                # non empty
                self.total_seen[-1] += torch.sum(targets != i).item()
                self.total_correct[-1] += torch.sum(
                    (targets != i) & (outputs != i)
                ).item()
                self.total_positive[-1] += torch.sum(outputs != i).item()

    def _after_epoch(self, logger):
        if dist.is_initialized():
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)
            dist.barrier()

        ious = []
        precs = []
        recas = []

        # Add last element representing non empty.
        for i in range(self.num_classes + 1):
            # Precision
            if self.total_positive[i] == 0:
                precs.append(0.0)
            else:
                cur_prec = self.total_correct[i] / self.total_positive[i]
                precs.append(cur_prec.item())

            # Recall and IoU
            if self.total_seen[i] == 0:
                ious.append(1)
                recas.append(1)
            else:
                cur_iou = self.total_correct[i] / (
                    self.total_seen[i] + self.total_positive[i] - self.total_correct[i]
                )
                cur_reca = self.total_correct[i] / self.total_seen[i]
                ious.append(cur_iou.item())
                recas.append(cur_reca)

        logger.info(f"Validation per class iou {self.name}:")
        for iou, prec, reca, label_str in zip(ious, precs, recas, self.label_str):
            logger.info("%s : %.2f%%, %.2f, %.2f" % (label_str, iou * 100, prec, reca))

        logger.info("-" * 25)
        # Considering empty as a class:
        # ious: [ ? ... ? iou_empty ? ... ? iou_non_empty ]
        logger.info("mIoU w. empty cls : %.2f" % (np.mean(ious[:-1]) * 100))
        del ious[self.empty_label]
        miou = np.mean(ious[:-1])
        logger.info("mIoU wo. empty cls : %.2f" % (miou * 100))
        iou = ious[-1]
        logger.info("IoU as non-empty IoU: %.2f" % (iou * 100))
        print([" & ".join([f"{round(num * 100, 2)}" for num in ious])])
        return miou * 100, iou * 100

    def _compute_iou(self, outputs, targets, mask=None):
        if mask is not None:
            outputs = outputs[mask]
            targets = targets[mask]

        # Remove ignore label, i.e 255.
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        ious = []
        for i in range(self.num_classes + 1):
            seen = torch.sum(targets == i).item()
            correct = torch.sum((targets == i) & (outputs == i)).item()
            positive = torch.sum(outputs == i).item()

            if i == self.empty_label:
                # non empty
                seen = torch.sum(targets != self.empty_label).item()
                correct = torch.sum(
                    (targets != self.empty_label) & (outputs != self.empty_label)
                ).item()
                positive = torch.sum(outputs != self.empty_label).item()

            if seen + positive - correct == 0:
                ious.append(
                    1.0
                )  # If no pixels are seen or predicted, IoU is 1 (perfect match)
            else:
                iou = correct / (seen + positive - correct)
                ious.append(iou)

        return ious


class L1_Metric:
    def __init__(self):
        self.score = torch.tensor([0.0], device="cuda")
        self.num_elems = torch.tensor([0], device="cuda")

    def reset(self) -> None:
        self.score = torch.zeros((1,)).cuda()
        self.num_elems = torch.zeros((1,)).cuda()

    def _after_step(self, pred, target, mask=None):
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        metric = torch.nn.functional.l1_loss(pred, target, reduction="none")
        self.num_elems += pred.numel()
        self.score += torch.sum(metric)

    def _after_epoch(self, logger):
        l1_depth = self.score.item() / (self.num_elems.item() + 1)
        logger.info(f"Validation L1 loss: {l1_depth}")
        return l1_depth
