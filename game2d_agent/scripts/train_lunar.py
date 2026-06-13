"""
PPO training script for LunarLander-v3 (discrete).

Usage:
    python scripts/train_lunar.py
    python scripts/train_lunar.py --total-timesteps 2000000 --num-envs 8
    python scripts/train_lunar.py --track          # enable TensorBoard

LunarLander-v3 basics:
    Observation : 8 continuous values (position, velocity, angle, leg contacts)
    Actions     : 4 discrete (nothing, left engine, main engine, right engine)
    Solved      : mean episode reward >= 200 over 100 consecutive episodes
    Typical     : converges in ~1-2 M steps with 4 parallel envs on CPU

    Note: gymnasium >= 1.3.0 uses v3; v2 is removed.

This script is self-contained for LunarLander. For pixel-based game training
see scripts/train.py (uses CNNActorCritic + pixel wrappers from envs/wrappers.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

from models.mlp_policy import MLPActorCritic
from models.ppo import PPOConfig
from models.ppo_mlp import VectorPPO


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_lunar_env(rank: int, seed: int):
    """Return a callable that builds one LunarLander-v3 instance."""
    def _init():
        env = gym.make("LunarLander-v3")
        # NormalizeObservation keeps each feature near zero-mean/unit-variance.
        # Rewards are scaled by REWARD_SCALE in the training loop instead of using
        # NormalizeReward, so raw episode rewards remain accessible for logging
        # and best-checkpoint tracking.
        env = gym.wrappers.NormalizeObservation(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def make_writer(log_dir: str, run_name: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
        path = os.path.join(log_dir, run_name)
        os.makedirs(path, exist_ok=True)
        return SummaryWriter(path)
    except ImportError:
        print("[WARNING] tensorboard not installed — skipping TensorBoard logging.")
        return None


def log_metrics(writer, metrics, iteration, global_step, episode_stats, lr, sps):
    ep_rew = episode_stats.get("mean_reward", float("nan"))
    ep_len = episode_stats.get("mean_length", float("nan"))
    n_eps  = episode_stats.get("n_episodes", 0)
    print(
        f"iter {iteration:>5} | "
        f"steps {global_step:>8,} | "
        f"sps {sps:>6.0f} | "
        f"ep_rew {ep_rew:>8.2f} | "
        f"ep_len {ep_len:>6.0f} | "
        f"n_eps {n_eps:>4} | "
        f"pol {metrics['policy_loss']:>7.4f} | "
        f"val {metrics['value_loss']:>7.4f} | "
        f"ent {metrics['entropy_loss']:>7.4f} | "
        f"kl {metrics['approx_kl']:>8.6f} | "
        f"ev {metrics['explained_variance']:>6.3f} | "
        f"lr {lr:.2e}"
    )

    if writer is None:
        return

    writer.add_scalar("charts/learning_rate",       lr,                             global_step)
    writer.add_scalar("charts/SPS",                 sps,                            global_step)
    writer.add_scalar("charts/mean_episode_reward", ep_rew,                         global_step)
    writer.add_scalar("charts/mean_episode_length", ep_len,                         global_step)
    writer.add_scalar("losses/policy_loss",         metrics["policy_loss"],         global_step)
    writer.add_scalar("losses/value_loss",          metrics["value_loss"],          global_step)
    writer.add_scalar("losses/entropy_loss",        metrics["entropy_loss"],        global_step)
    writer.add_scalar("losses/total_loss",          metrics["total_loss"],          global_step)
    writer.add_scalar("losses/approx_kl",           metrics["approx_kl"],           global_step)
    writer.add_scalar("losses/clip_fraction",       metrics["clip_fraction"],       global_step)
    writer.add_scalar("losses/explained_variance",  metrics["explained_variance"],  global_step)


# ---------------------------------------------------------------------------
# Episode stats tracker
# ---------------------------------------------------------------------------

class EpisodeTracker:
    """Accumulate episode returns and lengths across vectorised environments."""

    def __init__(self, n_envs: int):
        self.n_envs = n_envs
        self._ep_returns = np.zeros(n_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(n_envs, dtype=np.int32)
        self.completed_returns: list[float] = []
        self.completed_lengths: list[int]   = []

    def update(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        self._ep_returns += rewards
        self._ep_lengths += 1
        for i, done in enumerate(dones):
            if done:
                self.completed_returns.append(float(self._ep_returns[i]))
                self.completed_lengths.append(int(self._ep_lengths[i]))
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0

    def stats(self) -> dict:
        if not self.completed_returns:
            return {"mean_reward": float("nan"), "mean_length": float("nan"), "n_episodes": 0}
        stats = {
            "mean_reward": float(np.mean(self.completed_returns)),
            "mean_length": float(np.mean(self.completed_lengths)),
            "n_episodes":  len(self.completed_returns),
        }
        self.completed_returns.clear()
        self.completed_lengths.clear()
        return stats


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    policy: MLPActorCritic,
    obs_dim: int,
    n_actions: int,
    iteration: int,
    global_step: int,
    save_dir: str,
    tag: str = "",
    obs_rms=None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    name = f"lunar_checkpoint_{tag or global_step}.pt"
    path = os.path.join(save_dir, name)
    data: dict = {
        "policy_state_dict": policy.state_dict(),
        "obs_dim":           obs_dim,
        "n_actions":         n_actions,
        "iteration":         iteration,
        "total_timesteps":   global_step,
    }
    if obs_rms is not None:
        # Save running mean/var so NormalizeObservation can be restored at eval
        # time. Without this, a fresh wrapper starts from mean=0/var=1 and the
        # policy receives incorrectly-scaled observations.
        data["obs_rms"] = {
            "mean":  obs_rms.mean.copy(),
            "var":   obs_rms.var.copy(),
            "count": float(obs_rms.count),
        }
    torch.save(data, path)
    return path


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device : {device}")
    print(f"Seed   : {args.seed}")

    # ── Config ───────────────────────────────────────────────────────────────
    config = PPOConfig(
        n_steps         = args.n_steps,
        n_envs          = args.num_envs,
        gamma           = args.gamma,
        gae_lambda      = args.gae_lambda,
        n_epochs        = args.n_epochs,
        minibatch_size  = args.minibatch_size,
        clip_coef       = args.clip_coef,
        value_coef      = args.value_coef,
        entropy_coef    = args.entropy_coef,
        max_grad_norm   = args.max_grad_norm,
        clip_value_loss = True,
        learning_rate   = args.learning_rate,
        anneal_lr       = args.anneal_lr,
    )

    steps_per_iter = config.n_steps * config.n_envs
    total_iters    = args.total_timesteps // steps_per_iter
    print(f"Total timesteps : {args.total_timesteps:,}")
    print(f"Steps/iter      : {steps_per_iter}")
    print(f"Total iters     : {total_iters}")
    print(f"Parallel envs   : {config.n_envs}")

    # ── Environments ─────────────────────────────────────────────────────────
    print("\nCreating LunarLander-v3 environments...")
    envs = SyncVectorEnv([
        make_lunar_env(rank=i, seed=args.seed)
        for i in range(config.n_envs)
    ])

    obs_dim   = envs.single_observation_space.shape[0]   # 8
    n_actions = envs.single_action_space.n               # 4
    obs_shape = (obs_dim,)
    print(f"obs_dim   : {obs_dim}")
    print(f"n_actions : {n_actions}")

    # ── Policy + PPO ─────────────────────────────────────────────────────────
    policy = MLPActorCritic(obs_dim=obs_dim, n_actions=n_actions).to(device)
    ppo    = VectorPPO(policy, config, obs_shape, device)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy params : {total_params:,}")

    # ── Logging ──────────────────────────────────────────────────────────────
    run_name = f"lunar_ppo_{int(time.time())}"
    writer   = make_writer(os.path.join("logs", "lunar_ppo"), run_name) if args.track else None
    save_dir = os.path.join("models", "lunar_ppo")
    tracker  = EpisodeTracker(config.n_envs)

    # ── Initial observations ──────────────────────────────────────────────────
    obs_np, _ = envs.reset(seed=args.seed)
    # LunarLander returns (M, 8) float64 — cast to float32 on device
    obs  = torch.from_numpy(obs_np).float().to(device)
    done = torch.zeros(config.n_envs, device=device)

    print(f"\nStarting training — run: {run_name}\n")
    print(
        f"{'iter':>6} | {'steps':>9} | {'sps':>6} | "
        f"{'ep_rew':>8} | {'ep_len':>6} | {'n_eps':>5} | "
        f"{'pol':>7} | {'val':>7} | {'ent':>7} | "
        f"{'kl':>8} | {'ev':>6} | lr"
    )
    print("-" * 115)

    # Raw LunarLander rewards are in [-300, 300] per episode. Scaling by 0.01
    # keeps per-step values in a range where value MSE is comparable to policy
    # loss, without obscuring raw reward signals in logs or checkpoint selection.
    REWARD_SCALE = 0.01

    global_step      = 0
    t_start          = time.time()
    best_mean_reward = -np.inf      # raw reward — comparable across runs

    for iteration in range(1, total_iters + 1):
        t_iter = time.time()

        ppo.anneal_lr(iteration - 1, total_iters)
        current_lr = ppo.optimizer.param_groups[0]["lr"]

        # ── Collect rollout (inline for episode tracking) ─────────────────────
        ppo.buffer.reset()

        with torch.no_grad():
            for _ in range(config.n_steps):
                action, log_prob, _, value = policy.get_action_and_value(obs)

                cpu_actions = action.cpu().numpy()
                next_obs_np, reward_np, terminated_np, truncated_np, _ = envs.step(cpu_actions)
                done_np = np.logical_or(terminated_np, truncated_np).astype(np.float32)

                tracker.update(reward_np, done_np)   # raw rewards for logging

                next_obs  = torch.from_numpy(next_obs_np).float().to(device)
                reward    = torch.tensor(reward_np * REWARD_SCALE, dtype=torch.float32, device=device)
                next_done = torch.tensor(done_np, dtype=torch.float32, device=device)

                ppo.buffer.add(obs, action, log_prob, reward, done, value)
                obs  = next_obs
                done = next_done

            last_value = policy.get_value(obs)
            ppo.buffer.compute_gae(last_value, done, config.gamma, config.gae_lambda)

        global_step += steps_per_iter

        # ── PPO update ────────────────────────────────────────────────────────
        metrics = ppo.update()

        # ── Log ───────────────────────────────────────────────────────────────
        sps      = steps_per_iter / (time.time() - t_iter)
        ep_stats = tracker.stats()
        log_metrics(writer, metrics, iteration, global_step, ep_stats, current_lr, sps)

        # ── Best checkpoint (raw reward) ──────────────────────────────────────
        obs_rms  = envs.envs[0].obs_rms
        mean_rew = ep_stats.get("mean_reward", float("nan"))
        if ep_stats["n_episodes"] > 0 and mean_rew > best_mean_reward:
            best_mean_reward = mean_rew
            path = save_checkpoint(
                policy, obs_dim, n_actions, iteration, global_step,
                save_dir, tag="best", obs_rms=obs_rms,
            )
            print(f"  [best] new best {best_mean_reward:.2f} → {path}")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if iteration % args.save_interval == 0:
            path = save_checkpoint(
                policy, obs_dim, n_actions, iteration, global_step,
                save_dir, obs_rms=obs_rms,
            )
            print(f"  [ckpt] saved → {path}")

    # ── Final checkpoint ──────────────────────────────────────────────────────
    path = save_checkpoint(
        policy, obs_dim, n_actions, total_iters, global_step,
        save_dir, tag="final", obs_rms=envs.envs[0].obs_rms,
    )
    print(f"\nTraining complete — {global_step:,} steps in {time.time() - t_start:.1f}s")
    print(f"Final checkpoint : {path}")

    if writer:
        writer.close()
    envs.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v3 (discrete)")

    # Scale
    parser.add_argument("--total-timesteps", type=int,   default=1_000_000)
    parser.add_argument("--num-envs",        type=int,   default=4)
    parser.add_argument("--n-steps",         type=int,   default=128)

    # PPO
    parser.add_argument("--n-epochs",        type=int,   default=4)
    parser.add_argument("--minibatch-size",  type=int,   default=256)
    parser.add_argument("--clip-coef",       type=float, default=0.2)
    parser.add_argument("--value-coef",      type=float, default=0.25)
    parser.add_argument("--entropy-coef",    type=float, default=0.05)
    parser.add_argument("--max-grad-norm",   type=float, default=0.5)

    # Discount / GAE
    parser.add_argument("--gamma",           type=float, default=0.99)
    parser.add_argument("--gae-lambda",      type=float, default=0.95)

    # Optimiser
    parser.add_argument("--learning-rate",   type=float, default=3e-4)
    parser.add_argument("--anneal-lr",       action=argparse.BooleanOptionalAction, default=True)

    # Infrastructure
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--cpu",             action="store_true")
    parser.add_argument("--track",           action="store_true", help="Log to TensorBoard")
    parser.add_argument("--save-interval",   type=int,   default=100)

    args = parser.parse_args()
    train(args)
