_base_ = ["./loss.py"]
# =================== Loss ==========================
_base_.loss["w_occupancy"] = True
_base_.loss["render_weight"] = 10.0
_base_.loss["occupancy_weight"] = 1.0
_base_.loss["mask_elems"] = False
_base_.loss["kl_loss"] = False

_base_.loss.loss_cfgs.append(
    dict(
        type="RenderLoss",
        weight=None,
        mask_elems=None,
    )
)

_base_.loss_input_convertion["render"] = "render"
