"""Image metrics on tensors stored in [-1, 1]. None of these measure detection."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def to_unit(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) * 0.5


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    err = torch.mean((to_unit(pred) - to_unit(target)) ** 2).item()
    if err == 0:
        return 99.0
    return float(10 * torch.log10(torch.tensor(1.0 / err)))


def _window(channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(11, device=device, dtype=dtype) - 5
    kernel = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel[:, None] * kernel[None, :]
    kernel = kernel / kernel.sum()
    return kernel.expand(channels, 1, 11, 11).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    x = to_unit(pred)
    y = to_unit(target)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
    channels = x.shape[1]
    kernel = _window(channels, x.device, x.dtype)
    mu_x = F.conv2d(x, kernel, padding=5, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=5, groups=channels)
    sigma_x = F.conv2d(x * x, kernel, padding=5, groups=channels) - mu_x ** 2
    sigma_y = F.conv2d(y * y, kernel, padding=5, groups=channels) - mu_y ** 2
    sigma_xy = F.conv2d(x * y, kernel, padding=5, groups=channels) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean().item()


def lpips_or_none(pred: torch.Tensor, target: torch.Tensor):
    """LPIPS if the package is installed. A missing package is null, not zero."""
    try:
        import lpips
    except ImportError:
        return None
    if not hasattr(lpips_or_none, "_model"):
        lpips_or_none._model = lpips.LPIPS(net="alex")
    model = lpips_or_none._model.to(pred.device)
    x = to_unit(pred) * 2 - 1
    y = to_unit(target) * 2 - 1
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
    with torch.no_grad():
        return model(x, y).mean().item()
