"""
PPO for the local-map CNN policy (dict observation: {map, global}).

Mirrors models/ppo_mlp.py exactly, except the rollout stores two observation
tensors (map + global) and passes both to CNNMapActorCritic.get_action_and_value.
GAE, PPO clip loss, value clipping, entropy bonus, advantage normalisation and
LR annealing are identical.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from models.ppo import PPOConfig


class MapRolloutBuffer:
    def __init__(self, n_steps, n_envs, map_shape, global_dim, device):
        self.n_steps, self.n_envs = n_steps, n_envs
        self.map_shape, self.global_dim = map_shape, global_dim
        self.device = device
        T, M, d = n_steps, n_envs, device
        self.map_obs    = torch.zeros(T, M, *map_shape,   dtype=torch.float32, device=d)
        self.global_obs = torch.zeros(T, M, global_dim,   dtype=torch.float32, device=d)
        self.actions    = torch.zeros(T, M, dtype=torch.int64,   device=d)
        self.log_probs  = torch.zeros(T, M, dtype=torch.float32, device=d)
        self.rewards    = torch.zeros(T, M, dtype=torch.float32, device=d)
        self.dones      = torch.zeros(T, M, dtype=torch.float32, device=d)
        self.values     = torch.zeros(T, M, dtype=torch.float32, device=d)
        self.advantages = torch.zeros(T, M, dtype=torch.float32, device=d)
        self.returns    = torch.zeros(T, M, dtype=torch.float32, device=d)
        self._ptr = 0

    def reset(self):
        self._ptr = 0

    def add(self, map_obs, global_obs, action, log_prob, reward, done, value):
        t = self._ptr
        self.map_obs[t]    = map_obs
        self.global_obs[t] = global_obs
        self.actions[t]    = action
        self.log_probs[t]  = log_prob
        self.rewards[t]    = reward
        self.dones[t]      = done
        self.values[t]     = value.squeeze(-1)
        self._ptr += 1

    @torch.no_grad()
    def compute_gae(self, last_value, last_done, gamma, gae_lambda):
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

    def minibatches(self, minibatch_size):
        total = self.n_steps * self.n_envs
        idx_all = torch.randperm(total, device=self.device)
        maps    = self.map_obs.view(total, *self.map_shape)
        globs   = self.global_obs.view(total, self.global_dim)
        acts    = self.actions.view(total)
        logps   = self.log_probs.view(total)
        advs    = self.advantages.view(total)
        rets    = self.returns.view(total)
        vals    = self.values.view(total)
        for start in range(0, total, minibatch_size):
            idx = idx_all[start:start + minibatch_size]
            yield {
                "map":        maps[idx],
                "global":     globs[idx],
                "actions":    acts[idx],
                "log_probs":  logps[idx],
                "advantages": advs[idx],
                "returns":    rets[idx],
                "values":     vals[idx],
            }


class VectorMapPPO:
    def __init__(self, policy, config: PPOConfig, map_shape, global_dim, device):
        self.policy, self.config, self.device = policy, config, device
        self.optimizer = Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
        self.buffer = MapRolloutBuffer(config.n_steps, config.n_envs,
                                       map_shape, global_dim, device)

    def update(self) -> dict[str, float]:
        cfg = self.config
        metrics = {k: [] for k in ["policy_loss", "value_loss", "entropy_loss",
                                   "approx_kl", "clip_fraction"]}
        adv = self.buffer.advantages.view(-1)
        self.buffer.advantages = (self.buffer.advantages - adv.mean()) / (adv.std() + 1e-8)

        for _epoch in range(cfg.n_epochs):
            for b in self.buffer.minibatches(cfg.minibatch_size):
                _, new_log_prob, entropy, new_value = self.policy.get_action_and_value(
                    b["map"], b["global"], b["actions"])
                new_value = new_value.squeeze(-1)

                log_ratio = new_log_prob - b["log_probs"]
                ratio     = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()

                mb_adv = b["advantages"]
                surr1  = ratio * mb_adv
                surr2  = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                if cfg.clip_value_loss:
                    v_clipped = b["values"] + torch.clamp(
                        new_value - b["values"], -cfg.clip_coef, cfg.clip_coef)
                    value_loss = 0.5 * torch.max((new_value - b["returns"]) ** 2,
                                                 (v_clipped - b["returns"]) ** 2).mean()
                else:
                    value_loss = 0.5 * ((new_value - b["returns"]) ** 2).mean()

                entropy_loss = -entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy_loss"].append(entropy_loss.item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_fraction"].append(clip_frac)

        with torch.no_grad():
            r = self.buffer.returns.view(-1)
            v = self.buffer.values.view(-1)
            ev = (1.0 - (r - v).var() / (r.var() + 1e-8)).item()
        return {k: float(np.mean(x)) for k, x in metrics.items()} | {"explained_variance": ev}

    def anneal_lr(self, current_step: int, total_steps: int) -> None:
        if not self.config.anneal_lr:
            return
        frac = 1.0 - current_step / total_steps
        for pg in self.optimizer.param_groups:
            pg["lr"] = frac * self.config.learning_rate
