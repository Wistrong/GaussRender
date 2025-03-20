from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor


class GaussianPrediction(NamedTuple):
    means: Tensor
    scales: Tensor
    rotations: Tensor
    opacities: Tensor
    semantics: Tensor
