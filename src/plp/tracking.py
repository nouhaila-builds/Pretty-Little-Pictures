"""Optional Weights & Biases. Disabled unless WANDB_MODE=online and a key exists."""

from __future__ import annotations

import os


class Tracker:
    def __init__(self, enabled: bool, name: str, config: dict):
        self.enabled = enabled
        self.run = None
        if not enabled:
            return
        if not os.environ.get("WANDB_API_KEY"):
            self.enabled = False
            return
        import wandb

        self.run = wandb.init(project="pretty-little-pictures", name=name, config=config)

    def log(self, values: dict, step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
