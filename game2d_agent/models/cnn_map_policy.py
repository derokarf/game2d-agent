"""
CNN Actor-Critic over an egocentric local occupancy map + global goal vector.

See docs/agent_v2_local_map_spec.md.  The CNN is the part that learns spatial
routing — "wall here, gap there, goal that way → move around" — which is the
transferable skill.  Interface mirrors MLPActorCritic / GRUActorCritic so the
PPO loop is unchanged apart from passing (map, global) instead of a flat obs.

Architecture:
    map (B, 2, 17, 17) ─ Conv 2→16  (3x3, p1)   ReLU
                       ─ Conv 16→32 (3x3, s2,p1) ReLU
                       ─ Conv 32→32 (3x3, s2,p1) ReLU
                       ─ flatten(800) → FC 128 ReLU
    global (B, 3) ────────────────────────────────┐
    concat[128, 3] → FC 128 ReLU → ┬ actor  (→ n_actions)
                                   └ critic (→ 1)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _layer_init(layer: nn.Module, gain: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class CNNMapActorCritic(nn.Module):
    def __init__(
        self,
        map_channels: int = 2,
        map_size: int = 17,
        global_dim: int = 3,
        n_actions: int = 5,
    ):
        super().__init__()
        self.map_channels = map_channels
        self.map_size     = map_size
        self.global_dim   = global_dim
        self.n_actions    = n_actions

        self.conv = nn.Sequential(
            _layer_init(nn.Conv2d(map_channels, 16, 3, stride=1, padding=1)), nn.ReLU(),
            _layer_init(nn.Conv2d(16, 32, 3, stride=2, padding=1)),           nn.ReLU(),
            _layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)),           nn.ReLU(),
        )
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, map_channels, map_size, map_size)).flatten(1).shape[1]

        self.map_fc = nn.Sequential(_layer_init(nn.Linear(conv_out, 128)), nn.ReLU())
        self.trunk  = nn.Sequential(_layer_init(nn.Linear(128 + global_dim, 128)), nn.ReLU())

        self.actor_head  = _layer_init(nn.Linear(128, n_actions), gain=0.01)
        self.critic_head = _layer_init(nn.Linear(128, 1),         gain=1.0)

    # ------------------------------------------------------------------

    def forward(self, map_obs: torch.Tensor, global_obs: torch.Tensor):
        """map_obs: (B, C, S, S), global_obs: (B, global_dim) -> logits, value."""
        f = self.conv(map_obs).flatten(1)
        f = self.map_fc(f)
        f = torch.cat([f, global_obs], dim=1)
        f = self.trunk(f)
        return self.actor_head(f), self.critic_head(f)

    def get_action_and_value(self, map_obs, global_obs, action=None):
        logits, value = self.forward(map_obs, global_obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, map_obs, global_obs):
        return self.forward(map_obs, global_obs)[1]

    @torch.no_grad()
    def act(self, map_obs, global_obs) -> int:
        logits, _ = self.forward(map_obs, global_obs)
        return int(logits.argmax(dim=-1).item())
