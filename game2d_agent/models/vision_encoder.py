"""
VisionEncoder — CNN that maps a raw RGB game frame to a compact scene embedding.

Two classes:

  VisionEncoder   — the CNN backbone only (used during PPO training)
  VisionNet       — encoder + typed prediction heads (used during pre-training)

Pre-training flow:
    frame (84×84×3) → VisionEncoder → 64-dim embedding (partitioned by type)
                                            ↓           ↓            ↓
                                      player_head  target_head  obstacle_head
                                        2 values    15 values     9 values

PPO training flow:
    frame (84×84×3) → VisionEncoder → 64-dim embedding → MLP policy → actions

Embedding partition (64 dims total):
    [0:8]   → player_head   → player_x, player_y
    [8:28]  → target_head   → 5 targets × (dx, dy, dist)
    [28:40] → obstacle_head → 3 obstacles × (dx, dy, dist)
    [40:64] → free context  → 24 dims for MLP policy use

Label layout (SCENE_DIM = 26):
    [0]      player_x / screen_w
    [1]      player_y / screen_h
    [2..4]   nearest target:  dx/W, dy/H, dist/max_dist
    [5..7]   2nd target:      dx, dy, dist
    ...
    [14..16] 5th target:      dx, dy, dist
    [17..19] nearest obstacle: dx_to_center/W, dy_to_center/H, edge_dist/max_dist
    [20..22] 2nd obstacle:    dx, dy, edge_dist
    [23..25] 3rd obstacle:    dx, dy, edge_dist
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SCENE_DIM = 26   # length of the label vector

# Embedding slices — which dims the CNN allocates to each object type
PLAYER_EMBED   = slice(0, 8)    # 8 dims  → player position
TARGET_EMBED   = slice(8, 28)   # 20 dims → 5 targets
OBSTACLE_EMBED = slice(28, 40)  # 12 dims → 3 obstacles
# slice(40, 64): 24 free dims used by MLP policy

# Label slices — which values in SCENE_DIM each head predicts
PLAYER_LABELS   = slice(0, 2)
TARGET_LABELS   = slice(2, 17)
OBSTACLE_LABELS = slice(17, 26)


def _init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class VisionEncoder(nn.Module):
    """
    CNN backbone: (B, 3, 84, 84) float32 [0,1]  →  (B, embed_dim).

    The 64-dim output is partitioned by type (see module docstring).
    The CNN learns to put player features in dims 0-7, target features
    in dims 8-27, obstacle features in dims 28-39, and uses dims 40-63
    freely for context the MLP policy can exploit.
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
        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def load(cls, path: str, device: torch.device | str = "cpu") -> "VisionEncoder":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        encoder = cls(embed_dim=ckpt["embed_dim"])
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        return encoder


class VisionNet(nn.Module):
    """
    Encoder + typed prediction heads — used ONLY during supervised pre-training.

    Three separate heads each read from their own slice of the embedding:
      player_head   : embed[0:8]   → 2 values  (player_x, player_y)
      target_head   : embed[8:28]  → 15 values (5 targets × dx,dy,dist)
      obstacle_head : embed[28:40] → 9 values  (3 obstacles × dx,dy,dist)

    This forces the CNN to place type-specific information in specific
    embedding dimensions, giving the MLP a structured input.

    After training, only encoder weights are saved and used downstream.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.encoder       = VisionEncoder(in_channels, embed_dim)
        self.player_head   = _init(nn.Linear(8,  2),  gain=1.0)
        self.target_head   = _init(nn.Linear(20, 15), gain=1.0)
        self.obstacle_head = _init(nn.Linear(12, 9),  gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 84, 84) float32  →  (B, SCENE_DIM) predicted labels"""
        emb = self.encoder(x)
        player_pred   = self.player_head(emb[:, PLAYER_EMBED])
        target_pred   = self.target_head(emb[:, TARGET_EMBED])
        obstacle_pred = self.obstacle_head(emb[:, OBSTACLE_EMBED])
        return torch.cat([player_pred, target_pred, obstacle_pred], dim=1)
