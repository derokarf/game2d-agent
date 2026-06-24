"""
VisionEncoder — CNN that maps a raw RGB game frame to a compact scene embedding.

Two classes:

  VisionEncoder   — the CNN backbone only (used during PPO training, frozen)
  VisionNet       — encoder + linear prediction head (used during pre-training)

Pre-training flow:
    frame (84×84×3) → VisionEncoder → 64-dim embedding → linear head → 18 predicted labels
    loss = MSE(predicted, true_labels_from_game_state)

PPO training flow:
    frame (84×84×3) → VisionEncoder (frozen) → 64-dim embedding → MLP policy → actions

Label layout (SCENE_DIM = 18):
    [0]     player_x   / screen_w
    [1]     player_y   / screen_h
    [2..4]  nearest target:  dx, dy, dist  (all normalized)
    [5..7]  2nd target:      dx, dy, dist
    [8..10] 3rd target:      dx, dy, dist
    [11..13] 4th target:     dx, dy, dist
    [14..16] 5th target:     dx, dy, dist
    [17]    nearest obstacle dist / max_dist
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SCENE_DIM = 18   # length of the label vector


def _init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class VisionEncoder(nn.Module):
    """
    CNN backbone: (B, 3, 84, 84) float32 [0,1]  →  (B, embed_dim).

    Same conv architecture as the PPO policy CNN so the learned features
    are compatible if we ever want to fine-tune end-to-end.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

        flat = self._flat_size(in_channels)

        self.net = nn.Sequential(
            _init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),          nn.ReLU(),
            _init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),          nn.ReLU(),
            nn.Flatten(),
            _init(nn.Linear(flat, embed_dim)),                          nn.ReLU(),
        )

    @staticmethod
    def _flat_size(in_channels: int) -> int:
        dummy = torch.zeros(1, in_channels, 84, 84)
        conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4),
            nn.Conv2d(32, 64, 4, 2),
            nn.Conv2d(64, 64, 3, 1),
            nn.Flatten(),
        )
        with torch.no_grad():
            return int(conv(dummy).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 84, 84) float32 [0,1]  →  (B, embed_dim)"""
        return self.net(x)

    def freeze(self) -> None:
        """Call before PPO training to stop gradients flowing into the encoder."""
        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def load(cls, path: str, device: torch.device | str = "cpu") -> "VisionEncoder":
        """Load a pre-trained encoder from a checkpoint file."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        encoder = cls(embed_dim=ckpt["embed_dim"])
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        return encoder


class VisionNet(nn.Module):
    """
    Encoder + linear prediction head — used ONLY during supervised pre-training.
    After training, only the encoder weights are saved and used downstream.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 64, out_dim: int = SCENE_DIM):
        super().__init__()
        self.encoder = VisionEncoder(in_channels, embed_dim)
        self.head = _init(nn.Linear(embed_dim, out_dim), gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 84, 84) float32  →  (B, SCENE_DIM) predicted labels"""
        return self.head(self.encoder(x))
