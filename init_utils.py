from __future__ import annotations

import torch
from torch import Tensor


def overtone_spectral_init_(weight: Tensor, target_std: float, power: float = 0.5) -> None:
    # Power-law singular spectrum: s_k ~ k^-power, then rescale to the requested std.
    rows, cols = weight.shape
    rank = min(rows, cols)
    q_left = torch.linalg.qr(torch.randn(rows, rank, dtype=torch.float32), mode="reduced").Q
    q_right = torch.linalg.qr(torch.randn(cols, rank, dtype=torch.float32), mode="reduced").Q
    singular = torch.arange(1, rank + 1, dtype=torch.float32).pow(-power)
    init = (q_left * singular) @ q_right.T
    weight.copy_(init.mul_(target_std / init.std()).to(dtype=weight.dtype))
