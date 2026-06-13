"""
MLP Actor-Critic for vector-observation environments (e.g. LunarLander-v2).

Mirrors the CNNActorCritic interface so the same PPO training loop works
with either network — swap the policy, keep everything else.

Architecture:
    Input: (B, obs_dim)
          ↓
    Linear(obs_dim → 64) + Tanh
    Linear(64 → 64) + Tanh
          ↓
    ┌─────┴──────────┐
    Actor head        Critic head
    Linear(64 → n)    Linear(64 → 1)
    logits π(a|s)     value  V(s)

Why Tanh instead of ReLU?
    MLP policies with orthogonal init + small gain on the actor head benefit
    from Tanh's bounded output.  ReLU can produce dead neurons early in
    training when weights are near-zero.  The PPO paper (Schulman 2017) and
    CleanRL both use Tanh for vector-obs environments.

Initialization follows the same orthogonal convention as CNNActorCritic:
    Hidden layers : gain = sqrt(2)
    Actor head    : gain = 0.01  → near-uniform initial policy
    Critic head   : gain = 1.0
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


class MLPActorCritic(nn.Module):
    """
    Shared-trunk MLP with separate actor and critic heads.

    Args:
        obs_dim:   Dimension of the flat observation vector (e.g. 8 for LunarLander).
        n_actions: Number of discrete actions.
        hidden:    Sizes of the shared hidden layers (default: (64, 64)).

    Example::

        policy = MLPActorCritic(obs_dim=8, n_actions=4)
        obs = torch.zeros(1, 8)
        action, log_prob, entropy, value = policy.get_action_and_value(obs)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = (64, 64),
    ):
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers += [_layer_init(nn.Linear(in_dim, h)), nn.Tanh()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        self.actor_head  = _layer_init(nn.Linear(in_dim, n_actions), gain=0.01)
        self.critic_head = _layer_init(nn.Linear(in_dim, 1),         gain=1.0)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, obs_dim) float32.

        Returns:
            logits: (B, n_actions)
            value:  (B, 1)
        """
        features = self.trunk(obs)
        return self.actor_head(features), self.critic_head(features)

    # ------------------------------------------------------------------
    # PPO training interface  (same signatures as CNNActorCritic)
    # ------------------------------------------------------------------

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or evaluate an action under the current policy.

        Args:
            obs:    (B, obs_dim) float32.
            action: (B,) int64, or None to sample.

        Returns:
            action:   (B,) int64
            log_prob: (B,) float32
            entropy:  (B,) float32
            value:    (B, 1) float32
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic-only pass for GAE bootstrapping. Returns (B, 1)."""
        return self.critic_head(self.trunk(obs))

    # ------------------------------------------------------------------
    # Evaluation interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> int:
        """Greedy action for deterministic evaluation. obs: (1, obs_dim)."""
        logits, _ = self.forward(obs)
        return int(logits.argmax(dim=-1).item())