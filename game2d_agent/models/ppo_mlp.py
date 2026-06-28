"""
PPO for vector-observation environments (MLP policy variant).

Mirrors models/ppo.py but fixes two pixel-specific assumptions:

    1. RolloutBuffer allocated obs as (T, M, C, H, W) with explicit C,H,W
       unpacking.  VectorRolloutBuffer uses *obs_shape so it works for any
       shape tuple, including (obs_dim,) from LunarLander.

    2. PPO._to_tensor() did a channel permute (M,H,W,C)→(M,C,H,W) for
       Gymnasium's pixel convention.  VectorPPO._to_tensor() skips this —
       LunarLander returns (M, obs_dim) float32 directly.

Everything else — GAE formula, PPO clip loss, value loss, entropy bonus,
advantage normalisation, LR annealing — is identical to ppo.py.

PPOConfig is re-exported from ppo.py; no need to duplicate it.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from models.mlp_policy import MLPActorCritic
from models.ppo import PPOConfig          # reuse the shared hyperparameter class


# ---------------------------------------------------------------------------
# VectorRolloutBuffer
# ---------------------------------------------------------------------------

class VectorRolloutBuffer:
    """
    Rollout buffer for flat (vector) observations.

    Identical to RolloutBuffer in ppo.py except obs is allocated as
    (T, M, *obs_shape) rather than the hardcoded (T, M, C, H, W).

    obs_shape examples:
        (8,)         ← LunarLander-v2
        (4,)         ← CartPole
        (17,)        ← HalfCheetah (continuous — not used here)
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: tuple,       # e.g. (8,)
        device: torch.device,
    ):
        self.n_steps   = n_steps
        self.n_envs    = n_envs
        self.obs_shape = obs_shape
        self.device    = device

        self._ptr  = 0
        self._full = False
        self._alloc()

    def _alloc(self) -> None:
        T, M = self.n_steps, self.n_envs
        d = self.device

        self.obs       = torch.zeros(T, M, *self.obs_shape, dtype=torch.float32, device=d)
        self.actions   = torch.zeros(T, M,                  dtype=torch.int64,   device=d)
        self.log_probs = torch.zeros(T, M,                  dtype=torch.float32, device=d)
        self.rewards   = torch.zeros(T, M,                  dtype=torch.float32, device=d)
        self.dones     = torch.zeros(T, M,                  dtype=torch.float32, device=d)
        self.values    = torch.zeros(T, M,                  dtype=torch.float32, device=d)
        self.advantages = torch.zeros(T, M,                 dtype=torch.float32, device=d)
        self.returns    = torch.zeros(T, M,                 dtype=torch.float32, device=d)

    def reset(self) -> None:
        self._ptr  = 0
        self._full = False

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        t = self._ptr
        self.obs[t]       = obs
        self.actions[t]   = action
        self.log_probs[t] = log_prob
        self.rewards[t]   = reward
        self.dones[t]     = done
        self.values[t]    = value.squeeze(-1)
        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True

    @property
    def is_full(self) -> bool:
        return self._full

    @torch.no_grad()
    def compute_gae(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """GAE(λ) — identical maths to RolloutBuffer.compute_gae()."""
        last_value = last_value.squeeze(-1)
        last_done  = last_done.float()
        gae = torch.zeros_like(last_value)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value        = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value        = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae   = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def minibatches(self, minibatch_size: int) -> Iterator[dict[str, torch.Tensor]]:
        total   = self.n_steps * self.n_envs
        indices = torch.randperm(total, device=self.device)

        obs_flat        = self.obs.view(total, *self.obs_shape)
        actions_flat    = self.actions.view(total)
        log_probs_flat  = self.log_probs.view(total)
        advantages_flat = self.advantages.view(total)
        returns_flat    = self.returns.view(total)
        values_flat     = self.values.view(total)

        for start in range(0, total, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield {
                "obs":        obs_flat[idx],
                "actions":    actions_flat[idx],
                "log_probs":  log_probs_flat[idx],
                "advantages": advantages_flat[idx],
                "returns":    returns_flat[idx],
                "values":     values_flat[idx],
            }


# ---------------------------------------------------------------------------
# VectorPPO
# ---------------------------------------------------------------------------

class VectorPPO:
    """
    PPO wrapper for vector-observation / MLP-policy environments.

    Same interface as PPO in ppo.py:
        collect(envs, obs, done) → next_obs, next_done
        update()                 → dict of loss metrics
        anneal_lr(step, total)

    The only behavioural difference is _to_tensor(): no channel permutation,
    just a numpy→torch cast.  LunarLander already returns float64 arrays
    shaped (M, obs_dim); we cast to float32 and send to device.
    """

    def __init__(
        self,
        policy: MLPActorCritic,
        config: PPOConfig,
        obs_shape: tuple,       # e.g. (8,)
        device: torch.device,
    ):
        self.policy = policy
        self.config = config
        self.device = device

        self.optimizer = Adam(
            policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )
        self.buffer = VectorRolloutBuffer(
            n_steps   = config.n_steps,
            n_envs    = config.n_envs,
            obs_shape = obs_shape,
            device    = device,
        )

    # ------------------------------------------------------------------
    # Phase 1: Collect rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect(
        self,
        envs,
        obs: torch.Tensor,
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.buffer.reset()

        for _ in range(self.config.n_steps):
            action, log_prob, _entropy, value = self.policy.get_action_and_value(obs)

            cpu_actions = action.cpu().numpy()
            next_obs_np, reward_np, terminated_np, truncated_np, _ = envs.step(cpu_actions)
            done_np = np.logical_or(terminated_np, truncated_np).astype(np.float32)

            next_obs  = self._to_tensor(next_obs_np)
            reward    = torch.tensor(reward_np, dtype=torch.float32, device=self.device)
            next_done = torch.tensor(done_np,   dtype=torch.float32, device=self.device)

            self.buffer.add(obs, action, log_prob, reward, done, value)
            obs  = next_obs
            done = next_done

        last_value = self.policy.get_value(obs)
        self.buffer.compute_gae(last_value, done, self.config.gamma, self.config.gae_lambda)
        return obs, done

    # ------------------------------------------------------------------
    # Phase 2: PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        cfg = self.config
        metrics: dict[str, list[float]] = {
            k: [] for k in [
                "policy_loss", "value_loss", "entropy_loss", "total_loss",
                "approx_kl", "clip_fraction",
            ]
        }

        adv = self.buffer.advantages.view(-1)
        self.buffer.advantages = (self.buffer.advantages - adv.mean()) / (adv.std() + 1e-8)

        for _epoch in range(cfg.n_epochs):
            for batch in self.buffer.minibatches(cfg.minibatch_size):
                _, new_log_prob, entropy, new_value = self.policy.get_action_and_value(
                    batch["obs"], batch["actions"]
                )
                new_value = new_value.squeeze(-1)

                log_ratio = new_log_prob - batch["log_probs"]
                ratio     = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()

                mb_adv = batch["advantages"]
                surr1  = ratio * mb_adv
                surr2  = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                if cfg.clip_value_loss:
                    v_clipped       = batch["values"] + torch.clamp(
                        new_value - batch["values"], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_unclipped = (new_value - batch["returns"]) ** 2
                    v_loss_clipped   = (v_clipped  - batch["returns"]) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((new_value - batch["returns"]) ** 2).mean()

                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + cfg.value_coef   * value_loss
                    + cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy_loss"].append(entropy_loss.item())
                metrics["total_loss"].append(loss.item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_fraction"].append(clip_frac)

        with torch.no_grad():
            returns_flat = self.buffer.returns.view(-1)
            values_flat  = self.buffer.values.view(-1)
            ev = (1.0 - (returns_flat - values_flat).var() / (returns_flat.var() + 1e-8)).item()

        return {k: float(np.mean(v)) for k, v in metrics.items()} | {"explained_variance": ev}

    # ------------------------------------------------------------------
    # LR annealing
    # ------------------------------------------------------------------

    def anneal_lr(self, current_step: int, total_steps: int) -> None:
        if not self.config.anneal_lr:
            return
        frac = 1.0 - current_step / total_steps
        for pg in self.optimizer.param_groups:
            base_lr = pg.get("initial_lr", self.config.learning_rate)
            pg["lr"] = frac * base_lr

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _to_tensor(self, obs_np: np.ndarray) -> torch.Tensor:
        """Cast (M, obs_dim) numpy array to float32 device tensor. No permute."""
        return torch.from_numpy(obs_np).float().to(self.device)
