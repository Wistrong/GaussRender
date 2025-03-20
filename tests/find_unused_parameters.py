import argparse

import torch
from mmengine import Config
from mmseg.models import build_segmentor

from dataset import get_dataloader
from src.model.utils.utils import set_matrix_to_render


def get_unused_parameters(model, inputs):
    """
    Run a forward pass and identify unused parameters.

    Args:
    - model: The neural network model.
    - inputs: The inputs for the forward pass.

    Returns:
    - unused_params: List of unused parameters.
    """
    model.eval()  # Set model to evaluation mode
    out = model(**inputs)  # Forward pass

    # Compute loss
    loss = sum(
        [
            tensor.sum()
            for tensor in out["pred_occ"]
            + [tensor.sum() for tensor in out["render"]["cam"]]
            + [tensor.sum() for tensor in out["render"]["bev"]]
            + [tensor.sum() for tensor in out["render"]["depth"]]
            + [tensor.sum() for tensor in out["render"]["bev_depth"]]
        ]
    )
    loss.backward()  # Backpropagate to compute gradients

    # Identify unused parameters
    unused_params = [
        (name, param.numel())
        for name, param in model.named_parameters()
        if param.grad is None
    ]
    return unused_params


def main(config_path):
    # Load configuration
    cfg = Config.fromfile(config_path)

    # Get dataloaders
    _, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=None,
        iter_resume=None,
    )

    # Get a single batch of data
    data = next(iter(val_dataset_loader))
    data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    # Initialize and configure model
    model = build_segmentor(cfg.model)
    model.init_weights()
    model.cuda()

    # Set matrix to render
    with torch.no_grad():
        set_matrix_to_render(False, model, data)

    # Get unused parameters
    unused_params = get_unused_parameters(model, {"imgs": data["img"], "metas": data})
    print([name for (name, _) in unused_params])
    print(f"Total unused parameters: {sum(param for _, param in unused_params)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model with a specified config file."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config/tpvformer/render.py",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    main(args.config_path)
