import torch.nn as nn

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class MultiLoss(nn.Module):
    def __init__(
        self,
        loss_cfgs,
        render_weight=1.0,
        occupancy_weight=1.0,
        w_occupancy=True,
        mask_elems=False,
        kl_loss=False,
    ):
        super().__init__()

        assert isinstance(loss_cfgs, list)
        self.num_losses = len(loss_cfgs)

        losses = []
        for loss_cfg in loss_cfgs:
            if loss_cfg["type"] == "OccupancyLoss":
                loss_cfg["weight"] = occupancy_weight
                if not w_occupancy:
                    continue

            if loss_cfg["type"] == "RenderLoss":
                loss_cfg["weight"] = render_weight
                loss_cfg["mask_elems"] = mask_elems
                loss_cfg["kl_loss"] = kl_loss
            losses.append(OPENOCC_LOSS.build(loss_cfg))
        assert len(losses) > 0, "No loss functions found."
        self.losses = nn.ModuleList(losses)
        self.iter_counter = 0
        self.writer = None

    def forward(self, inputs):
        writer = self.writer

        loss_dict = {}
        tot_loss = 0.0
        for loss_func in self.losses:
            loss = loss_func(inputs)
            if type(loss) != dict:
                tot_loss += loss
                loss_dict.update({loss_func.__class__.__name__: loss.detach().item()})
                if writer and self.iter_counter % 10 == 0:
                    writer.add_scalar(
                        f"loss/{loss_func.__class__.__name__}",
                        loss.detach().item(),
                        self.iter_counter,
                    )
            else:
                for k in loss.keys():
                    tot_loss += loss[k]
                    loss_dict.update(
                        {
                            loss_func.__class__.__name__
                            + "_"
                            + k: loss[k].detach().item()
                        }
                    )
                    if writer and self.iter_counter % 10 == 0:
                        writer.add_scalar(
                            f"loss/{loss_func.__class__.__name__}_{k}",
                            loss[k].detach().item(),
                            self.iter_counter,
                        )
        if writer and self.iter_counter % 100 == 0:
            writer.add_scalar("loss/total", tot_loss.detach().item(), self.iter_counter)
        self.iter_counter += 1

        return tot_loss, loss_dict
