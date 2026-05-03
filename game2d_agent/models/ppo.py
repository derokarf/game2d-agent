"""
PPO (Proximal Policy Optimization) — core algorithm.

This module contains two classes:

    RolloutBuffer
        Stores transitions collected from N parallel environments over T steps.
        Computes advantages via GAE(λ) and returns (discounted cumulative rewards).

    PPO
        Wraps the policy network + optimizer.
        Exposes collect() and update() — the two phases of the PPO training loop.

────────────────────────────────────────────────────────────────────────────────
PPO training loop (one iteration):

    1. COLLECT  — run policy in envs for T steps, store (obs, action, reward,
                  done, log_prob, value) in RolloutBuffer.

    2. GAE      — compute advantage A_t and return G_t for every timestep:

                  δ_t   = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
                  A_t   = δ_t + (γλ) · A_{t+1} · (1 - done_t)
                  G_t   = A_t + V(s_t)

    3. UPDATE   — for K epochs, shuffle the batch and compute:

                  ratio       = exp(log π_new(a|s) - log π_old(a|s))
                  L_clip      = -mean( min(ratio·A, clip(ratio, 1-ε, 1+ε)·A) )
                  L_value     = 0.5 · mean( (V_new(s) - G)² )
                  L_entropy   = -mean( H[π(·|s)] )
                  L_total     = L_clip + c_v · L_value + c_e · L_entropy

                  where c_v = value_coef, c_e = entropy_coef.

────────────────────────────────────────────────────────────────────────────────
Key concepts:

GAE (Generalized Advantage Estimation)
    λ=0 → pure TD residual (low variance, high bias)
    λ=1 → full Monte-Carlo return (high variance, zero bias)
    λ=0.95 (default) → practical bias-variance sweet spot.

PPO clipping
    The ratio r_t = π_new/π_old measures how much the policy changed.
    Without clipping, a large update could make r_t >> 1, causing the loss
    to blow up and the policy to collapse.  Clipping to [1-ε, 1+ε] ensures
    each gradient step is a small, conservative improvement.

Entropy bonus
    H[π] = -Σ π(a|s) log π(a|s).  Adding it to the loss encourages the
    policy to stay exploratory.  Without it, the policy often becomes
    deterministic too early (before finding the optimal strategy).

Value clipping (optional, enabled by default)
    Clips the value update similarly to the policy update, preventing the
    critic from making large jumps that destabilise the actor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from models.policy import CNNActorCritic


# ---------------------------------------------------------------------------
# Hyperparameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """All PPO hyperparameters in one place."""

    # Rollout
    n_steps: int = 128          # T — steps per env per rollout
    n_envs: int = 4             # M — parallel environments

    # Discount / GAE
    gamma: float = 0.99         # discount factor γ
    gae_lambda: float = 0.95    # GAE λ — bias-variance tradeoff

    # PPO update
    n_epochs: int = 4           # K — update epochs per rollout
    minibatch_size: int = 256   # samples per gradient step
    clip_coef: float = 0.1      # ε — PPO clip range
    value_coef: float = 0.5     # c_v — value loss weight
    entropy_coef: float = 0.01  # c_e — entropy bonus weight
    max_grad_norm: float = 0.5  # gradient clipping norm

    # Value clipping (mirrors policy clipping, stabilises critic)
    clip_value_loss: bool = True

    # Optimiser
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True       # linearly decay LR to 0 over training


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Fixed-size circular buffer that stores one rollout (T steps × M envs).

    Layout of stored tensors — all shape (T, M, ...):
        obs        (T, M, C, H, W)  float32
        actions    (T, M)           int64
        log_probs  (T, M)           float32  — log π_old(a|s)
        rewards    (T, M)           float32
        dones      (T, M)           float32  — 1.0 if episode ended
        values     (T, M)           float32  — V(s_t) from critic

    After compute_gae():
        advantages (T, M)           float32
        returns    (T, M)           float32
    """

    def __init__(self, n_steps: int, n_envs: int, obs_shape: tuple, device: torch.device):
        """
        Args:
            n_steps:   T — rollout length per environment.
            n_envs:    M — number of parallel environments.
            obs_shape: (C, H, W) — channel-first observation shape.
            device:    CPU or CUDA — all tensors stored here.
        """
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.device = device

        self._ptr = 0       # current write position
        self._full = False

        self._alloc()

    def _alloc(self) -> None:
        T, M = self.n_steps, self.n_envs
        C, H, W = self.obs_shape
        d = self.device

        self.obs       = torch.zeros(T, M, C, H, W, dtype=torch.float32, device=d)
        self.actions   = torch.zeros(T, M,           dtype=torch.int64,   device=d)
        self.log_probs = torch.zeros(T, M,           dtype=torch.float32, device=d)
        self.rewards   = torch.zeros(T, M,           dtype=torch.float32, device=d)
        self.dones     = torch.zeros(T, M,           dtype=torch.float32, device=d)
        self.values    = torch.zeros(T, M,           dtype=torch.float32, device=d)

        # Computed by compute_gae()
        self.advantages = torch.zeros(T, M, dtype=torch.float32, device=d)
        self.returns    = torch.zeros(T, M, dtype=torch.float32, device=d)

    def reset(self) -> None:
        """Clear the buffer for the next rollout."""
        self._ptr = 0
        self._full = False

    def add(
        self,
        obs: torch.Tensor,        # (M, C, H, W)
        action: torch.Tensor,     # (M,)
        log_prob: torch.Tensor,   # (M,)
        reward: torch.Tensor,     # (M,)
        done: torch.Tensor,       # (M,)  float32  1.0=terminal
        value: torch.Tensor,      # (M,)
    ) -> None:
        """Store one timestep of transitions from all M environments."""
        t = self._ptr
        self.obs[t]       = obs
        self.actions[t]   = action
        self.log_probs[t] = log_prob
        self.rewards[t]   = reward
        self.dones[t]     = done
        self.values[t]    = value.squeeze(-1)  # (M,1) → (M,)

        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True

    @property
    def is_full(self) -> bool:
        return self._full

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_gae(
        self,
        last_value: torch.Tensor,   # (M,) or (M,1) — V(s_{T+1})
        last_done: torch.Tensor,    # (M,)           — done flag at s_{T+1}
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """
        Compute advantages via GAE(λ) and store them in self.advantages.
        Also compute returns = advantages + values and store in self.returns.

        GAE formula (backward pass through time):

            δ_t   = r_t  +  γ · V(s_{t+1}) · (1 - done_t)  -  V(s_t)
            A_t   = δ_t  +  γλ · A_{t+1}   · (1 - done_t)

        Why backwards?
            A_t depends on A_{t+1}, so we iterate from t=T-1 down to t=0.

        Why multiply by (1 - done)?
            At episode boundaries V(s_{t+1}) = 0 (no future reward after
            terminal state), and the advantage chain must not bleed across
            episodes.

        Args:
            last_value: bootstrapped value V(s_{T+1}), (M,) or (M,1).
            last_done:  done flag for the step *after* the last stored step.
            gamma:      discount factor γ.
            gae_lambda: GAE λ.
        """
        last_value = last_value.squeeze(-1)   # (M,)
        last_done  = last_done.float()        # (M,)

        gae = torch.zeros_like(last_value)    # A_{t+1}, initialised to 0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            # TD residual
            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            # Accumulate GAE
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------
    # Mini-batch sampling
    # ------------------------------------------------------------------

    def minibatches(self, minibatch_size: int) -> Iterator[dict[str, torch.Tensor]]:
        """
        Flatten the (T, M) buffer into (T*M,) and yield random mini-batches.

        Yields dicts with keys:
            obs, actions, log_probs, advantages, returns, values
        All tensors are (minibatch_size, ...).
        """
        total = self.n_steps * self.n_envs
        indices = torch.randperm(total, device=self.device)

        # Flatten T×M → N
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
# PPO
# ---------------------------------------------------------------------------

class PPO:
    """
    PPO algorithm wrapper.

    Owns:
        - policy (CNNActorCritic)
        - optimizer (Adam)
        - rollout buffer (RolloutBuffer)

    Exposes:
        collect(envs, obs, done) → next_obs, next_done
        update()                 → dict of loss metrics
        anneal_lr(step, total)   → update LR schedule

    Usage::

        ppo = PPO(policy, config, obs_shape, device)
        obs, _ = envs.reset()
        done = torch.zeros(n_envs)

        for iteration in range(total_iterations):
            obs, done = ppo.collect(envs, obs, done)
            metrics = ppo.update()
            ppo.anneal_lr(iteration, total_iterations)
    """

    def __init__(
        self,
        policy: CNNActorCritic,
        config: PPOConfig,
        obs_shape: tuple,           # (C, H, W)
        device: torch.device,
    ):
        self.policy = policy
        self.config = config
        self.device = device

        self.optimizer = Adam(
            policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5,   # Adam ε — standard for RL, slightly larger than default
        )

        self.buffer = RolloutBuffer(
            n_steps=config.n_steps,
            n_envs=config.n_envs,
            obs_shape=obs_shape,
            device=device,
        )

    # ------------------------------------------------------------------
    # Phase 1: Collect rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect(
        self,
        envs,                        # gymnasium.vector.VectorEnv
        obs: torch.Tensor,           # (M, C, H, W) current observations
        done: torch.Tensor,          # (M,)         current done flags
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the current policy for n_steps steps across M environments.
        Stores transitions in the rollout buffer, then computes GAE.

        Args:
            envs: Vectorised Gymnasium environment.
            obs:  Current observations, float32 (M, C, H, W) on device.
            done: Current done flags, float32 (M,) on device.

        Returns:
            next_obs:  (M, C, H, W) — observations after the rollout.
            next_done: (M,)         — done flags after the rollout.
        """
        self.buffer.reset()

        for _ in range(self.config.n_steps):
            action, log_prob, _entropy, value = self.policy.get_action_and_value(obs)

            # Step all environments
            cpu_actions = action.cpu().numpy()
            next_obs_np, reward_np, terminated_np, truncated_np, _ = envs.step(cpu_actions)

            done_np = np.logical_or(terminated_np, truncated_np).astype(np.float32)

            next_obs  = self._to_tensor(next_obs_np)   # (M, H, W, C) → (M, C, H, W)
            reward    = torch.tensor(reward_np,  dtype=torch.float32, device=self.device)
            next_done = torch.tensor(done_np,    dtype=torch.float32, device=self.device)

            self.buffer.add(obs, action, log_prob, reward, done, value)

            obs  = next_obs
            done = next_done

        # Bootstrap value for GAE
        last_value = self.policy.get_value(obs)
        self.buffer.compute_gae(
            last_value=last_value,
            last_done=done,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        return obs, done

    # ------------------------------------------------------------------
    # Phase 2: PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """
        Run K epochs of PPO updates over the collected rollout.

        For each epoch:
            1. Shuffle the flattened buffer.
            2. Split into mini-batches.
            3. For each mini-batch:
               a. Re-evaluate log π(a|s), H[π], V(s) under the current policy.
               b. Compute PPO clip loss, value loss, entropy bonus.
               c. Backprop + clip gradients + Adam step.

        Returns:
            dict with scalar metrics for logging:
                policy_loss, value_loss, entropy_loss, total_loss,
                approx_kl, clip_fraction, explained_variance
        """
        cfg = self.config
        metrics: dict[str, list[float]] = {
            k: [] for k in [
                "policy_loss", "value_loss", "entropy_loss", "total_loss",
                "approx_kl", "clip_fraction",
            ]
        }

        # Normalise advantages over the full batch (reduces variance)
        adv = self.buffer.advantages.view(-1)
        adv_mean, adv_std = adv.mean(), adv.std()
        # Avoid division by zero for degenerate batches
        adv_norm = (self.buffer.advantages - adv_mean) / (adv_std + 1e-8)
        self.buffer.advantages = adv_norm

        for _epoch in range(cfg.n_epochs):
            for batch in self.buffer.minibatches(cfg.minibatch_size):
                # ── Re-evaluate under current policy ─────────────────
                _, new_log_prob, entropy, new_value = self.policy.get_action_and_value(
                    batch["obs"], batch["actions"]
                )
                new_value = new_value.squeeze(-1)   # (B,1) → (B,)

                # ── PPO policy loss ───────────────────────────────────
                # log ratio = log π_new - log π_old  (numerically stable)
                log_ratio = new_log_prob - batch["log_probs"]
                ratio = log_ratio.exp()

                # Approximate KL for early stopping / monitoring
                # Using the unbiased estimator: KL ≈ (ratio - 1) - log_ratio
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = (
                        (ratio - 1.0).abs() > cfg.clip_coef
                    ).float().mean().item()

                mb_adv = batch["advantages"]

                # Clipped surrogate objective (both terms are maximised,
                # so we negate to get a minimisation loss)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value loss ────────────────────────────────────────
                if cfg.clip_value_loss:
                    # Clip value update to prevent large critic jumps
                    v_clipped = batch["values"] + torch.clamp(
                        new_value - batch["values"],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_unclipped = (new_value   - batch["returns"]) ** 2
                    v_loss_clipped   = (v_clipped   - batch["returns"]) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((new_value - batch["returns"]) ** 2).mean()

                # ── Entropy bonus ─────────────────────────────────────
                # Negative because we want to maximise entropy (minimise -H)
                entropy_loss = -entropy.mean()

                # ── Total loss ────────────────────────────────────────
                loss = (
                    policy_loss
                    + cfg.value_coef  * value_loss
                    + cfg.entropy_coef * entropy_loss
                )

                # ── Gradient step ─────────────────────────────────────
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                # ── Record metrics ────────────────────────────────────
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy_loss"].append(entropy_loss.item())
                metrics["total_loss"].append(loss.item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_fraction"].append(clip_frac)

        # Explained variance — how well V(s) predicts G_t
        # ev = 1 - Var(G - V) / Var(G).  Close to 1 → good critic.
        with torch.no_grad():
            returns_flat = self.buffer.returns.view(-1)
            values_flat  = self.buffer.values.view(-1)
            var_returns  = returns_flat.var()
            ev = (
                1.0 - (returns_flat - values_flat).var() / (var_returns + 1e-8)
            ).item()

        return {k: float(np.mean(v)) for k, v in metrics.items()} | {
            "explained_variance": ev
        }

    # ------------------------------------------------------------------
    # LR annealing
    # ------------------------------------------------------------------

    def anneal_lr(self, current_step: int, total_steps: int) -> None:
        """
        Linearly decay the learning rate from its initial value to 0.

        Why anneal?
            Early in training large LR steps make fast progress.
            Late in training small LR steps make fine-grained adjustments
            without overshooting the optimal policy.

        Args:
            current_step: Current training iteration (0-indexed).
            total_steps:  Total number of training iterations.
        """
        if not self.config.anneal_lr:
            return
        frac = 1.0 - current_step / total_steps
        new_lr = frac * self.config.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, obs_np: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy observation from vectorised envs to a device tensor.

        gymnasium.vector returns (M, H, W, C) uint8 or float32.
        The network expects (M, C, H, W) float32 — already normalised by
        the NormalizeObsWrapper, so no /255 needed here.
        """
        obs = torch.from_numpy(obs_np).to(self.device)
        if obs.ndim == 4:
            obs = obs.permute(0, 3, 1, 2)   # (M, H, W, C) → (M, C, H, W)
        return obs.float()
