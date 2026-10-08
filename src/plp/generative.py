"""Conditional diffusion: p(thermal | RGB) by noise prediction.

The thermal image is noised. RGB is concatenated to that noisy image.
The network predicts the noise. Sampling uses a short DDIM trajectory.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from plp.nets import UNet


class ConditionalDiffusion(nn.Module):
    def __init__(self, steps: int = 1000, base: int = 32):
        super().__init__()
        self.steps = steps
        self.net = UNet(4, 1, base=base, with_time=True)
        betas = torch.linspace(1e-4, 0.02, steps)
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def loss(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        batch = thermal.shape[0]
        t = torch.randint(0, self.steps, (batch,), device=thermal.device)
        noise = torch.randn_like(thermal)
        scale = self.alphas_cumprod[t][:, None, None, None]
        noisy = scale.sqrt() * thermal + (1.0 - scale).sqrt() * noise
        prediction = self.net(torch.cat([noisy, rgb], dim=1), t)
        return torch.mean((prediction - noise) ** 2)

    @torch.no_grad()
    def sample(self, rgb: torch.Tensor, ddim_steps: int = 50) -> torch.Tensor:
        batch, _, height, width = rgb.shape
        thermal = torch.randn(batch, 1, height, width, device=rgb.device)
        times = torch.linspace(self.steps - 1, 0, ddim_steps, device=rgb.device).long()
        for index, step in enumerate(times):
            step_batch = torch.full((batch,), int(step), device=rgb.device, dtype=torch.long)
            noise = self.net(torch.cat([thermal, rgb], dim=1), step_batch)
            alpha = self.alphas_cumprod[step]
            clean = ((thermal - (1.0 - alpha).sqrt() * noise) / alpha.sqrt()).clamp(-1, 1)
            if index + 1 == len(times):
                thermal = clean
            else:
                alpha_prev = self.alphas_cumprod[times[index + 1]]
                thermal = alpha_prev.sqrt() * clean + (1.0 - alpha_prev).sqrt() * noise
        return thermal
