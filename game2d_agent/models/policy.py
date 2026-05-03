"""
CNN Actor-Critic network for PPO.

Architecture (Mnih et al. 2015 — Nature DQN, adapted for PPO):

    Input: (B, C, H, W)  e.g. (B, 4, 84, 84)  float32 [0, 1]
              ↓
       ┌──────────────────────────────────────┐
       │  Shared CNN Backbone                 │
       │  Conv2d(C →32, 8×8, stride=4) + ReLU │  → (B, 32, 20, 20)
       │  Conv2d(32→64, 4×4, stride=2) + ReLU │  → (B, 64,  9,  9)
       │  Conv2d(64→64, 3×3, stride=1) + ReLU │  → (B, 64,  7,  7)
       │  Flatten → Linear(3136→512)  + ReLU  │  → (B, 512)
       └──────────────┬───────────────────────┘
                      │
             ┌────────┴────────┐
             ▼                 ▼
        Actor head         Critic head
        Linear(512→n)      Linear(512→1)
        logits π(a|s)      value  V(s)

Initialization (CleanRL convention):
    Conv + hidden linear : orthogonal, gain = sqrt(2)
    Actor head           : orthogonal, gain = 0.01  → near-uniform initial policy
    Critic head          : orthogonal, gain = 1.0

Why shared backbone?
    The critic needs to understand the same visual features as the actor to
    produce accurate value estimates.  Sharing encourages representations that
    are useful for both objectives, and halves the number of conv parameters.

Why orthogonal init?
    Default Kaiming init tends to produce large initial logit magnitudes,
    causing the initial policy to be near-deterministic and entropy to
    collapse early.  Orthogonal init with small gain (0.01) on the actor head
    keeps initial action probabilities close to uniform, preserving exploration.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layer_init(layer: nn.Module, gain: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    """Apply orthogonal weight init + constant bias init to a single layer."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


# ---------------------------------------------------------------------------
# CNNActorCritic
# ---------------------------------------------------------------------------

class CNNActorCritic(nn.Module):
    """
    Shared-backbone CNN with separate actor and critic heads.

    Args:
        obs_shape:  Observation shape as (C, H, W) — channel-first, matching
                    the (B, C, H, W) tensor convention used throughout the project.
                    After wrappers: (4, 84, 84).
        n_actions:  Number of discrete actions.

    Example::

        policy = CNNActorCritic(obs_shape=(4, 84, 84), n_actions=5)
        obs = torch.zeros(1, 4, 84, 84)
        action, log_prob, entropy, value = policy.get_action_and_value(obs)
    """

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int):
        super().__init__()

        in_channels = obs_shape[0]  # C from (C, H, W)

        # ------------------------------------------------------------------
        # Shared CNN backbone
        # ------------------------------------------------------------------
        self.backbone = nn.Sequential(
            _layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            _layer_init(nn.Linear(self._cnn_output_size(obs_shape), 512)),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Actor head  — outputs raw logits (NOT probabilities)
        # ------------------------------------------------------------------
        # gain=0.01: keeps initial logits small → near-uniform π(a|s)
        # This is critical: if initial logits are large, one action dominates
        # from step 0, collapsing exploration before training even starts.
        self.actor_head = _layer_init(nn.Linear(512, n_actions), gain=0.01)

        # ------------------------------------------------------------------
        # Critic head — outputs scalar V(s)
        # ------------------------------------------------------------------
        # gain=1.0: value estimates start at a reasonable scale
        self.critic_head = _layer_init(nn.Linear(512, 1), gain=1.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cnn_output_size(obs_shape: tuple[int, int, int]) -> int:
        """
        Compute the flattened CNN output size by doing a single dummy forward
        pass through the conv layers.  This avoids hardcoding 3136 and makes
        the class work with any (C, H, W) input.
        """
        c, h, w = obs_shape
        dummy = torch.zeros(1, c, h, w)
        conv = nn.Sequential(
            nn.Conv2d(c,  32, kernel_size=8, stride=4),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Flatten(),
        )
        with torch.no_grad():
            return int(conv(dummy).shape[1])

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass through backbone + both heads.

        Args:
            obs: (B, C, H, W) float32 tensor, values in [0, 1].

        Returns:
            logits: (B, n_actions) — unnormalised action scores.
            value:  (B, 1)         — state value estimate V(s).
        """
        features = self.backbone(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    # ------------------------------------------------------------------
    # PPO training interface
    # ------------------------------------------------------------------

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or evaluate an action under the current policy.

        Called in two contexts:
        1. Rollout collection (action=None):
               Sample a ~ π(·|s), return it along with log π(a|s), H[π], V(s).
        2. PPO update (action=stored action from rollout):
               Re-evaluate log π(a|s) and H[π] under the *updated* policy.
               This is what PPO's probability ratio r_t(θ) = π_new / π_old needs.

        Args:
            obs:    (B, C, H, W) float32.
            action: (B,) int64 tensor, or None to sample.

        Returns:
            action:   (B,) int64 — sampled or passed-through action.
            log_prob: (B,) float32 — log π(action | obs).
            entropy:  (B,) float32 — H[π(·|obs)] = -Σ π log π.
                      Used as entropy bonus in PPO loss to encourage exploration.
            value:    (B, 1) float32 — V(obs).
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Critic-only forward pass.  Used in GAE to bootstrap V(s_T) at the
        end of a rollout without running the actor head.

        Args:
            obs: (B, C, H, W) float32.

        Returns:
            value: (B, 1) float32.
        """
        features = self.backbone(obs)
        return self.critic_head(features)

    # ------------------------------------------------------------------
    # Evaluation / play interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> int:
        """
        Greedy action selection for evaluation (no gradient computation).

        Picks argmax of logits rather than sampling, which is appropriate
        for deterministic evaluation of a trained policy.

        Args:
            obs: (1, C, H, W) float32 — single observation, batch size 1.

        Returns:
            action: int — index of the chosen action.
        """
        logits, _ = self.forward(obs)
        return int(logits.argmax(dim=-1).item())
