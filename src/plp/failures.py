"""Rank test frames by residual. Does not invent a human observation."""

from __future__ import annotations

import torch

from plp.metrics import to_unit


def rank_by_mae(names: list[str], predictions: torch.Tensor, targets: torch.Tensor) -> list[dict]:
    error = torch.mean(torch.abs(to_unit(predictions) - to_unit(targets)), dim=(1, 2, 3))
    order = torch.argsort(error, descending=True)
    ranked = []
    for index in order.tolist():
        ranked.append(
            {
                "sample_id": names[index],
                "category": "unreviewed",
                "mae": float(error[index]),
                "observation": None,
                "possible_explanation": None,
            }
        )
    return ranked
