import torch
from torch import Tensor


def vscale(x: Tensor) -> Tensor:
    return torch.asinh(x)


def vscale_inv(y: Tensor) -> Tensor:
    return torch.sinh(y)
