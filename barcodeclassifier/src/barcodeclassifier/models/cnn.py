"""
tiny_cnn.py
-----------
A 4-layer CNN (~60 k parameters) that outputs P(label-present).

Input  : RGB tensor in range 0-1, shape [B, 3, 32, 32]
Output : sigmoidal probability in range [0, 1], shape [B]
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)      # 32×32 → 32×32
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)     # 16×16 after pool
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)     # 8×8  after pool
        self.head  = nn.Linear(32, 1)                    # GAP → logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)       # 32 → 16
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)       # 16 → 8
        x = F.relu(self.conv3(x))                        # 8 → 8
        x = x.mean(dim=[2, 3])                           # global avg pool → [B,32]
        logit = self.head(x).squeeze(1)                  # [B]
        return torch.sigmoid(logit)

    # ---------- convenience I/O  ----------
    @classmethod
    def load(cls, ckpt_path: str | Path, map_location: str | torch.device = "cpu") -> "TinyCNN":
        model = cls()
        model.load_state_dict(torch.load(ckpt_path, map_location=map_location))
        model.eval()
        return model

    def save(self, ckpt_path: str | Path) -> None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), ckpt_path)

