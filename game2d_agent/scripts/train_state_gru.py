"""
PPO + GRU training on true game state (no CNN).

The GRU receives the exact 26-dim state vector directly from the game engine
(player pos, 5 targets × dx/dy/dist, 3 obstacles × dx/dy/dist) and carries
a hidden state h across steps so it remembers recent movement history.

This combines the two insights from previous experiments:
    - Direct coordinates → no perception noise (confirmed better than CNN)
    - GRU memory        → reduces oscillation vs stateless MLP

If this agent navigates decisively, we know:
    1. The coordinate representation is correct
    2. GRU memory adds value on top of coordinates
    3. The remaining CNN work is purely a perception problem

Usage:
    python scripts/train_state_gru.py
    python scripts/train_state_gru.py --total-timesteps 1_000_000 --log-file logs/train_state_gru_run1.log
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
from gymnasium.vector import SyncVectorEnv

from envs.state_env import StateGameEnv, OBS_DIM
from models.gru_policy import GRUActorCritic
from models.ppo import PPOConfig
from models.ppo_mlp import VectorRolloutBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _log(msg: str, log_file: str | None) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def save_checkpoint(policy, optimizer, hidden_dim, n_actions, iteration,
                    global_step, best_reward, save_dir, tag=""):
    os.makedirs(save_dir, exist_ok=True)
    name = f"checkpoint_{tag}.pt" if tag else "checkpoint_latest.pt"
    path = os.path.join(save_dir, name)
    torch.save({
        "policy_type":          "gru",
        "policy_state_dict":    policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_dim":              OBS_DIM,
        "hidden_dim":           hidden_dim,
        "n_actions":            n_actions,
        "iteration":            iteration,
        "global_step":          global_step,
        "best_reward":          best_reward,
    }, path)
    return path


# ---------------------------------------------------------------------------
# GRU PPO update (sequence-based BPTT)
# ---------------------------------------------------------------------------

def gru_update(
    policy: GRUActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: VectorRolloutBuffer,
    h_init: torch.Tensor,      # (M, hidden_dim)
    config: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    T, M = buffer.n_steps, buffer.n_envs

    adv_flat = buffer.advantages.view(-1)
    adv_norm = (buffer.advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    envs_per_mb = max(1, config.minibatch_size // T)

    metrics: dict[str, list[float]] = {
        k: [] for k in ["policy_loss", "value_loss", "entropy_loss", "approx_kl"]
    }

    for _epoch in range(config.n_epochs):
        env_order = torch.randperm(M, device=device)

        for start in range(0, M, envs_per_mb):
            env_idx = env_order[start : start + envs_per_mb]
            batch_M = len(env_idx)

            h = h_init[env_idx].unsqueeze(0)   # (1, batch_M, hidden_dim)

            obs_seq = buffer.obs[:, env_idx]     # (T, batch_M, obs_dim)
            don_seq = buffer.dones[:, env_idx]   # (T, batch_M)
            act_seq = buffer.actions[:, env_idx] # (T, batch_M)

            logits_seq, value_seq, _ = policy.forward_sequence(obs_seq, h, don_seq)

            dist = Categorical(logits=logits_seq)
            new_log_probs = dist.log_prob(act_seq)
            entropy       = dist.entropy()
            new_values    = value_seq.squeeze(-1)

            old_log_probs = buffer.log_probs[:, env_idx]
            old_values    = buffer.values[:, env_idx]
            returns_seq   = buffer.returns[:, env_idx]
            adv_seq       = adv_norm[:, env_idx]

            log_ratio = (new_log_probs - old_log_probs).view(-1)
            ratio     = log_ratio.exp()
            mb_adv    = adv_seq.view(-1)

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            nv_flat  = new_values.view(-1)
            ov_flat  = old_values.view(-1)
            ret_flat = returns_seq.view(-1)
            v_clip   = ov_flat + torch.clamp(nv_flat - ov_flat, -config.clip_coef, config.clip_coef)
            value_loss = 0.5 * torch.max(
                (nv_flat - ret_flat) ** 2,
                (v_clip  - ret_flat) ** 2,
            ).mean()

            entropy_loss = -entropy.mean()
            loss = (
                policy_loss
                + config.value_coef   * value_loss
                + config.entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy_loss"].append(entropy_loss.item())
            metrics["approx_kl"].append(approx_kl)

    with torch.no_grad():
        ret_flat = buffer.returns.view(-1)
        val_flat = buffer.values.view(-1)
        ev = (1.0 - (ret_flat - val_flat).var() / (ret_flat.var() + 1e-8)).item()

    return {k: float(np.mean(v)) for k, v in metrics.items()} | {"explained_variance": ev}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(policy, eval_envs, eval_seeds, device, max_steps=1000):
    n = len(eval_seeds)
    obs_np, _ = eval_envs.reset(seed=eval_seeds)
    obs = torch.from_numpy(obs_np).float().to(device)
    h   = policy.initial_state(n, device)

    ep_rewards = np.zeros(n)
    ep_done    = np.zeros(n, dtype=bool)
    final_rew  = np.full(n, np.nan)

    for _ in range(max_steps):
        done_t = torch.tensor(ep_done, dtype=torch.float32, device=device)
        h = h * (1.0 - done_t).view(1, -1, 1)

        action, _, _, _, h = policy.get_action_and_value(obs, h)
        obs_np, rewards, terminated, truncated, _ = eval_envs.step(action.cpu().numpy())

        for i in range(n):
            if not ep_done[i]:
                ep_rewards[i] += rewards[i]
                if terminated[i] or truncated[i]:
                    final_rew[i] = ep_rewards[i]
                    ep_done[i]   = True

        if ep_done.all():
            break
        obs = torch.from_numpy(obs_np).float().to(device)

    final_rew = np.where(np.isnan(final_rew), ep_rewards, final_rew)
    return final_rew


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        with open(args.log_file, "w") as f:
            f.write(f"  {'iter':>5} | {'steps':>10} | {'sps':>6} | {'ep_rew':>7} | "
                    f"{'ep_len':>6} | {'n_eps':>5} | {'pol':>8} | {'val':>8} | "
                    f"{'ent':>8} | {'kl':>10} | {'ev':>6} | {'lr':>10}\n")

    # ── Environments ─────────────────────────────────────────────────────────
    env_kwargs = dict(max_steps=1000, n_obstacles=args.n_obstacles, n_targets=args.n_targets)
    envs = SyncVectorEnv([
        (lambda kw: lambda: StateGameEnv(**kw))(env_kwargs)
        for _ in range(args.num_envs)
    ])
    eval_seeds = list(range(1000, 1000 + args.n_eval))
    eval_envs  = SyncVectorEnv([
        (lambda kw: lambda: StateGameEnv(**kw))(env_kwargs)
        for _ in range(args.n_eval)
    ])
    n_actions = envs.single_action_space.n

    # ── GRU Policy ───────────────────────────────────────────────────────────
    policy = GRUActorCritic(
        obs_dim    = OBS_DIM,
        hidden_dim = args.hidden_dim,
        n_actions  = n_actions,
    ).to(device)

    config = PPOConfig(
        n_steps        = args.n_steps,
        n_envs         = args.num_envs,
        n_epochs       = 4,
        minibatch_size = 256,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_coef      = 0.2,
        value_coef     = 0.5,
        entropy_coef   = args.entropy_coef,
        max_grad_norm  = 0.5,
        learning_rate  = args.lr,
        anneal_lr      = True,
        clip_value_loss= True,
    )

    optimizer = Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    buffer = VectorRolloutBuffer(
        n_steps   = args.n_steps,
        n_envs    = args.num_envs,
        obs_shape = (OBS_DIM,),
        device    = device,
    )

    steps_per_iter = args.n_steps * args.num_envs
    total_iters    = args.total_timesteps // steps_per_iter

    _log(f"Device        : {device}", args.log_file)
    _log(f"Obs dim       : {OBS_DIM}  (true game state, no CNN)", args.log_file)
    _log(f"GRU hidden    : {args.hidden_dim}", args.log_file)
    _log(f"Environments  : {args.num_envs}", args.log_file)
    _log(f"Steps/iter    : {steps_per_iter:,}", args.log_file)
    _log(f"Total iters   : {total_iters:,}", args.log_file)
    _log(f"Total steps   : {total_iters * steps_per_iter:,}", args.log_file)
    _log(f"Eval envs     : {args.n_eval}  (seeds {eval_seeds[0]}–{eval_seeds[-1]}, every {args.eval_interval} iters)", args.log_file)

    # ── Initial state ─────────────────────────────────────────────────────────
    obs_np, _ = envs.reset(seed=42)
    obs  = torch.from_numpy(obs_np).float().to(device)
    done = torch.zeros(args.num_envs, device=device)
    h    = policy.initial_state(args.num_envs, device)

    ep_rewards = np.zeros(args.num_envs)
    ep_lengths = np.zeros(args.num_envs, dtype=int)

    best_mean_reward = -np.inf
    best_eval_reward = -np.inf
    global_step      = 0
    t_start          = time.time()

    def anneal_lr(iteration: int) -> None:
        frac = 1.0 - (iteration - 1) / total_iters
        for pg in optimizer.param_groups:
            pg["lr"] = frac * args.lr

    # ── Main loop ────────────────────────────────────────────────────────────
    for iteration in range(1, total_iters + 1):
        anneal_lr(iteration)
        buffer.reset()

        completed_rewards: list[float] = []
        completed_lengths: list[int]   = []

        h_init = h.squeeze(0).detach().clone()   # (M, hidden_dim)

        with torch.no_grad():
            for _ in range(args.n_steps):
                h = h * (1.0 - done).view(1, -1, 1)
                action, log_prob, _ent, value, h = policy.get_action_and_value(obs, h)

                obs_np, rewards, terminated, truncated, _ = envs.step(action.cpu().numpy())
                done_np = np.logical_or(terminated, truncated).astype(np.float32)

                ep_rewards += rewards
                ep_lengths += 1
                for i, d in enumerate(done_np):
                    if d:
                        completed_rewards.append(float(ep_rewards[i]))
                        completed_lengths.append(int(ep_lengths[i]))
                        ep_rewards[i] = 0
                        ep_lengths[i] = 0

                next_obs  = torch.from_numpy(obs_np).float().to(device)
                reward_t  = torch.tensor(rewards, dtype=torch.float32, device=device)
                done_t    = torch.tensor(done_np, dtype=torch.float32, device=device)

                buffer.add(obs, action, log_prob, reward_t, done_t, value)
                obs  = next_obs
                done = done_t

        global_step += steps_per_iter

        with torch.no_grad():
            h_boot = h * (1.0 - done).view(1, -1, 1)
            last_value, _ = policy.get_value(obs, h_boot)
        buffer.compute_gae(last_value, done, config.gamma, config.gae_lambda)

        metrics = gru_update(policy, optimizer, buffer, h_init, config, device)

        # ── Logging ──────────────────────────────────────────────────────────
        elapsed    = time.time() - t_start
        sps        = int(global_step / elapsed)
        lr_now     = optimizer.param_groups[0]["lr"]
        ep_rew_str = f"{np.mean(completed_rewards):>7.2f}" if completed_rewards else "    nan"
        ep_len_str = f"{int(np.mean(completed_lengths)):>6}" if completed_lengths else "   nan"
        n_eps      = len(completed_rewards)

        _log(
            f"iter {iteration:>5} | steps {global_step:>10,} | sps {sps:>6} | "
            f"ep_rew {ep_rew_str} | ep_len {ep_len_str} | n_eps {n_eps:>5} | "
            f"pol {metrics['policy_loss']:>8.4f} | val {metrics['value_loss']:>8.4f} | "
            f"ent {metrics['entropy_loss']:>8.4f} | kl {metrics['approx_kl']:>10.6f} | "
            f"ev {metrics['explained_variance']:>6.3f} | lr {lr_now:.2e}",
            args.log_file,
        )

        if iteration % 50 == 0:
            remaining = (total_iters - iteration) * (elapsed / iteration)
            _log(f"  eta {datetime.timedelta(seconds=int(remaining))}", args.log_file)

        if completed_rewards:
            mean_rew = float(np.mean(completed_rewards))
            if mean_rew > best_mean_reward:
                best_mean_reward = mean_rew
                path = save_checkpoint(policy, optimizer, args.hidden_dim, n_actions,
                                       iteration, global_step, best_mean_reward,
                                       args.save_dir, tag="best")
                _log(f"  [best] new best ep_rew {best_mean_reward:.3f} → {path}", args.log_file)

        if iteration % args.eval_interval == 0:
            policy.eval()
            eval_rewards = run_eval(policy, eval_envs, eval_seeds, device)
            policy.train()
            eval_mean = float(np.mean(eval_rewards))
            eval_std  = float(np.std(eval_rewards))
            _log(
                f"  [eval] mean {eval_mean:>8.2f} | std {eval_std:>7.2f} | "
                f"min {float(np.min(eval_rewards)):>8.2f} | max {float(np.max(eval_rewards)):>8.2f}",
                args.log_file,
            )
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                path = save_checkpoint(policy, optimizer, args.hidden_dim, n_actions,
                                       iteration, global_step, best_eval_reward,
                                       args.save_dir, tag="best_eval")
                _log(f"  [eval] new best eval {best_eval_reward:.3f} → {path}", args.log_file)

    # ── Done ─────────────────────────────────────────────────────────────────
    save_checkpoint(policy, optimizer, args.hidden_dim, n_actions,
                    total_iters, global_step, best_mean_reward,
                    args.save_dir, tag="final")
    _log(f"\nTraining complete — {global_step:,} steps in {time.time()-t_start:.1f}s", args.log_file)
    _log(f"Best ep_rew   : {best_mean_reward:.3f}", args.log_file)
    _log(f"Best eval_rew : {best_eval_reward:.3f}", args.log_file)

    envs.close()
    eval_envs.close()
    import os as _os; _os._exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU on true game state (no CNN)")
    parser.add_argument("--total-timesteps", type=int,   default=1_000_000)
    parser.add_argument("--num-envs",        type=int,   default=8)
    parser.add_argument("--n-steps",         type=int,   default=128)
    parser.add_argument("--lr",              type=float, default=2.5e-4)
    parser.add_argument("--entropy-coef",    type=float, default=0.01)
    parser.add_argument("--hidden-dim",      type=int,   default=128)
    parser.add_argument("--save-dir",        default="models/state_gru")
    parser.add_argument("--log-file",        default="logs/train_state_gru_run1.log")
    parser.add_argument("--eval-interval",   type=int,   default=20)
    parser.add_argument("--n-eval",          type=int,   default=16)
    parser.add_argument("--n-obstacles",     type=int,   default=3)
    parser.add_argument("--n-targets",       type=int,   default=5)
    args = parser.parse_args()
    train(args)
