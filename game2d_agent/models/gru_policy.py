"""
GRU Actor-Critic policy for PPO with recurrent memory.

The agent carries a hidden state h across time steps within an episode,
giving it temporal context the stateless MLP cannot have.

Architecture:
    obs_t (B, obs_dim)  +  h_{t-1} (1, B, hidden_dim)
                    ↓
               GRU cell
                    ↓
           out_t (B, hidden_dim),  h_t (1, B, hidden_dim)
            /                 \\
       actor head          critic head
    logits (B, n_actions)   value (B, 1)

Training contract:
    - Hidden state h is carried step-to-step during rollout collection.
    - h is zeroed for any env whose done flag is True (episode boundary).
    - During PPO update, sequences are replayed env-by-env from h_init
      with h zeroed at done boundaries (BPTT through T steps).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class GRUActorCritic(nn.Module):
    """
    Recurrent actor-critic: GRU trunk + actor head + critic head.

    Args:
        obs_dim:    Input observation dimension (e.g. 64 for CNN embedding).
        hidden_dim: GRU hidden state size (default 128).
        n_actions:  Number of discrete actions.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128, n_actions: int = 5):
        super().__init__()
        self.obs_dim    = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions  = n_actions

        self.gru   = nn.GRU(obs_dim, hidden_dim, batch_first=False)
        self.actor = _init(nn.Linear(hidden_dim, n_actions), gain=0.01)
        self.critic = _init(nn.Linear(hidden_dim, 1),        gain=1.0)

        # GRU weights — standard orthogonal init on each gate matrix
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    # ------------------------------------------------------------------
    # Single-step forward (used during rollout collection)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,   # (B, obs_dim)
        h: torch.Tensor,     # (1, B, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single GRU step.

        Returns:
            logits:  (B, n_actions)
            value:   (B, 1)
            h_new:   (1, B, hidden_dim)
        """
        out, h_new = self.gru(obs.unsqueeze(0), h)  # out: (1, B, H)
        out = out.squeeze(0)                          # (B, H)
        return self.actor(out), self.critic(out), h_new

    # ------------------------------------------------------------------
    # Sequence forward (used during PPO update — BPTT)
    # ------------------------------------------------------------------

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,    # (T, B, obs_dim)
        h0: torch.Tensor,         # (1, B, hidden_dim)  — initial hidden state
        dones: torch.Tensor,      # (T, B) float32  — 1.0 = episode ended at t-1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process T steps with h reset at episode boundaries.

        We step manually (not via packed sequences) so we can zero h
        for individual envs mid-sequence when done[t-1, env]=1.

        Returns:
            logits_seq:  (T, B, n_actions)
            value_seq:   (T, B, 1)
            h_final:     (1, B, hidden_dim)
        """
        T, B, _ = obs_seq.shape
        h = h0
        all_out: list[torch.Tensor] = []

        for t in range(T):
            if t > 0:
                # Zero hidden state for envs whose previous step ended an episode
                mask = (1.0 - dones[t - 1]).view(1, B, 1)
                h = h * mask

            out_t, h = self.gru(obs_seq[t : t + 1], h)   # (1, B, H)
            all_out.append(out_t.squeeze(0))               # (B, H)

        out_seq = torch.stack(all_out)  # (T, B, H)
        return self.actor(out_seq), self.critic(out_seq), h

    # ------------------------------------------------------------------
    # PPO training interface
    # ------------------------------------------------------------------

    def get_action_and_value(
        self,
        obs: torch.Tensor,              # (B, obs_dim)
        h: torch.Tensor,                # (1, B, hidden_dim)
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or evaluate action.

        Returns:
            action:   (B,) int64
            log_prob: (B,) float32
            entropy:  (B,) float32
            value:    (B, 1) float32
            h_new:    (1, B, hidden_dim)
        """
        logits, value, h_new = self.forward(obs, h)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, h_new

    def get_value(
        self,
        obs: torch.Tensor,   # (B, obs_dim)
        h: torch.Tensor,     # (1, B, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Critic-only pass for GAE bootstrapping."""
        _, value, h_new = self.forward(obs, h)
        return value, h_new

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def initial_state(self, n_envs: int, device: torch.device) -> torch.Tensor:
        """Return zeros initial hidden state: (1, n_envs, hidden_dim)."""
        return torch.zeros(1, n_envs, self.hidden_dim, device=device)
