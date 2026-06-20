"""
PPO training script — from scratch, no Stable Baselines3.

Usage:
    python scripts/train.py
    python scripts/train.py --total-timesteps 1000000 --num-envs 8
    python scripts/train.py --total-timesteps 500000 --track  # TensorBoard logging

Structure of one training iteration:
    1. collect()  — run policy for n_steps across n_envs, compute GAE
    2. update()   — K epochs of PPO gradient steps over the rollout
    3. log        — print metrics + write to TensorBoard (if --track)
    4. checkpoint — save policy every --save-interval iterations

Checkpoint format (plain PyTorch, loadable by play.py):
    {
        "policy_state_dict": OrderedDict,
        "n_actions":         int,
        "obs_shape":         tuple,   # (C, H, W)
        "iteration":         int,
        "total_timesteps":   int,
    }
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use SDL dummy video driver so pygame doesn't need an X11 display.
# Must be set before any pygame import (which happens inside envs/).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

from envs.pixel_env import PixelGameEnv
from envs.wrappers import make_env
from models.policy import CNNActorCritic
from models.ppo import PPO, PPOConfig


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_single_env(rank: int, seed: int):
    """
    Return a callable that builds one wrapped environment instance.

    Each env gets a different seed (seed + rank) so parallel envs explore
    different trajectories — critical for diverse experience collection.
    """
    def _init():
        env = make_env(
            env_factory=lambda: PixelGameEnv(frame_size=(84, 84), max_steps=1000),
            frame_size=(84, 84),
            num_stack=4,
            reward_clip=(-1.0, 1.0),
            episodic_life=True,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def make_writer(log_dir: str, run_name: str):
    """Create a TensorBoard SummaryWriter. Returns None if tensorboard missing."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        path = os.path.join(log_dir, run_name)
        os.makedirs(path, exist_ok=True)
        return SummaryWriter(path)
    except ImportError:
        print("[WARNING] tensorboard not installed — skipping TensorBoard logging.")
        print("          pip install tensorboard")
        return None


def log_metrics(
    writer,
    metrics: dict,
    iteration: int,
    total_iters: int,
    global_step: int,
    episode_stats: dict,
    lr: float,
    sps: float,
    t_start: float,
) -> None:
    """Write all metrics to TensorBoard and print a summary line."""
    ep_rew = episode_stats.get("mean_reward", float("nan"))
    ep_len = episode_stats.get("mean_length", float("nan"))
    n_eps  = episode_stats.get("n_episodes", 0)

    now     = datetime.datetime.now().strftime("%H:%M:%S")
    elapsed = time.time() - t_start

    print(
        f"[{now}] "
        f"iter {iteration:>5} | "
        f"steps {global_step:>8,} | "
        f"sps {sps:>6.0f} | "
        f"ep_rew {ep_rew:>7.2f} | "
        f"ep_len {ep_len:>6.0f} | "
        f"n_eps {n_eps:>4} | "
        f"pol {metrics['policy_loss']:>7.4f} | "
        f"val {metrics['value_loss']:>7.4f} | "
        f"ent {metrics['entropy_loss']:>7.4f} | "
        f"kl {metrics['approx_kl']:>8.6f} | "
        f"ev {metrics['explained_variance']:>6.3f} | "
        f"lr {lr:.2e}"
    )

    if iteration % 50 == 0 and iteration > 0:
        remaining_secs = elapsed / iteration * (total_iters - iteration)
        eta = datetime.timedelta(seconds=int(remaining_secs))
        done_pct = 100.0 * iteration / total_iters
        print(f"  --> {done_pct:.1f}% done | elapsed {datetime.timedelta(seconds=int(elapsed))} | eta {eta}")

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
    """
    Tracks episode returns and lengths across vectorised environments.

    gymnasium.vector doesn't accumulate episode stats across resets
    automatically, so we track them manually here.
    """

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
    policy: CNNActorCritic,
    obs_shape: tuple,
    n_actions: int,
    iteration: int,
    global_step: int,
    save_dir: str,
    tag: str = "",
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    name = f"checkpoint_{tag or global_step}.pt"
    path = os.path.join(save_dir, name)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "obs_shape":         obs_shape,
            "n_actions":         n_actions,
            "iteration":         iteration,
            "total_timesteps":   global_step,
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device : {device}")
    print(f"Seed   : {args.seed}")

    # ── Config ───────────────────────────────────────────────────────────────
    config = PPOConfig(
        n_steps        = args.n_steps,
        n_envs         = args.num_envs,
        gamma          = args.gamma,
        gae_lambda     = args.gae_lambda,
        n_epochs       = args.n_epochs,
        minibatch_size = args.minibatch_size,
        clip_coef      = args.clip_coef,
        value_coef     = args.value_coef,
        entropy_coef   = args.entropy_coef,
        max_grad_norm  = args.max_grad_norm,
        clip_value_loss= True,
        learning_rate  = args.learning_rate,
        anneal_lr      = args.anneal_lr,
    )

    steps_per_iter = config.n_steps * config.n_envs
    total_iters    = args.total_timesteps // steps_per_iter
    print(f"Total timesteps : {args.total_timesteps:,}")
    print(f"Steps/iter      : {steps_per_iter}")
    print(f"Total iters     : {total_iters}")
    print(f"Parallel envs   : {config.n_envs}")

    # ── Environments ─────────────────────────────────────────────────────────
    print("\nCreating environments...")
    envs = SyncVectorEnv([
        make_single_env(rank=i, seed=args.seed)
        for i in range(config.n_envs)
    ])

    # Observation shape for the network: (C, H, W)
    # envs.single_observation_space.shape is (H, W, C) → transpose
    h, w, c = envs.single_observation_space.shape
    obs_shape = (c, h, w)   # channel-first for CNN
    n_actions = envs.single_action_space.n
    print(f"Obs shape (C,H,W): {obs_shape}")
    print(f"Action space     : {n_actions} discrete actions")

    # ── Policy + PPO ─────────────────────────────────────────────────────────
    policy = CNNActorCritic(obs_shape=obs_shape, n_actions=n_actions).to(device)
    ppo    = PPO(policy, config, obs_shape, device)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy params    : {total_params:,}")

    # ── Logging ──────────────────────────────────────────────────────────────
    run_name = f"ppo_{int(time.time())}"
    writer   = make_writer(os.path.join("logs", "ppo"), run_name) if args.track else None
    save_dir = args.save_dir
    tracker  = EpisodeTracker(config.n_envs)

    # ── Initial observations ──────────────────────────────────────────────────
    obs_np, _ = envs.reset(seed=args.seed)
    # (M, H, W, C) → (M, C, H, W) float32
    obs  = torch.from_numpy(obs_np).permute(0, 3, 1, 2).float().to(device)
    done = torch.zeros(config.n_envs, device=device)

    print(f"\nStarting training — run: {run_name}\n")
    print(
        f"{'iter':>6} | {'steps':>9} | {'sps':>6} | "
        f"{'ep_rew':>7} | {'ep_len':>6} | {'n_eps':>5} | "
        f"{'pol':>7} | {'val':>7} | {'ent':>7} | "
        f"{'kl':>8} | {'ev':>6} | lr"
    )
    print("-" * 110)

    global_step      = 0
    t_start          = time.time()
    best_mean_reward = -np.inf

    for iteration in range(1, total_iters + 1):
        t_iter = time.time()

        # ── Anneal LR ────────────────────────────────────────────────────────
        ppo.anneal_lr(iteration - 1, total_iters)
        current_lr = ppo.optimizer.param_groups[0]["lr"]

        # ── Phase 1: Collect rollout ──────────────────────────────────────────
        # Run inline (rather than ppo.collect) so we can intercept step outputs
        # for episode tracking before they are overwritten by the next reset.
        ppo.buffer.reset()

        with torch.no_grad():
            for _ in range(config.n_steps):
                action, log_prob, _, value = policy.get_action_and_value(obs)

                cpu_actions = action.cpu().numpy()
                next_obs_np, reward_np, terminated_np, truncated_np, _ = envs.step(cpu_actions)
                done_np = np.logical_or(terminated_np, truncated_np).astype(np.float32)

                tracker.update(reward_np, done_np)

                next_obs  = torch.from_numpy(next_obs_np).permute(0, 3, 1, 2).float().to(device)
                reward    = torch.tensor(reward_np, dtype=torch.float32, device=device)
                next_done = torch.tensor(done_np,   dtype=torch.float32, device=device)

                ppo.buffer.add(obs, action, log_prob, reward, done, value)

                obs  = next_obs
                done = next_done

            last_value = policy.get_value(obs)
            ppo.buffer.compute_gae(last_value, done, config.gamma, config.gae_lambda)

        global_step += steps_per_iter

        # ── Phase 2: PPO update ───────────────────────────────────────────────
        metrics = ppo.update()

        # ── Logging ──────────────────────────────────────────────────────────
        sps      = steps_per_iter / (time.time() - t_iter)
        ep_stats = tracker.stats()
        log_metrics(writer, metrics, iteration, total_iters, global_step, ep_stats, current_lr, sps, t_start)

        # ── Best checkpoint ───────────────────────────────────────────────────
        mean_rew = ep_stats.get("mean_reward", float("nan"))
        if ep_stats["n_episodes"] > 0 and mean_rew > best_mean_reward:
            best_mean_reward = mean_rew
            path = save_checkpoint(
                policy, obs_shape, n_actions, iteration, global_step, save_dir, tag="best"
            )
            print(f"  [best] new best {best_mean_reward:.3f} → {path}")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if iteration % args.save_interval == 0:
            path = save_checkpoint(
                policy, obs_shape, n_actions, iteration, global_step, save_dir
            )
            print(f"  [ckpt] saved → {path}")

    # ── Final checkpoint ──────────────────────────────────────────────────────
    path = save_checkpoint(
        policy, obs_shape, n_actions, total_iters, global_step, save_dir, tag="final"
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
    parser = argparse.ArgumentParser(description="Train a PPO agent on pixel-based 2D game")

    # Scale
    parser.add_argument("--total-timesteps", type=int,   default=500_000)
    parser.add_argument("--num-envs",        type=int,   default=4)
    parser.add_argument("--n-steps",         type=int,   default=128,
                        help="Rollout steps per env per iteration")

    # PPO
    parser.add_argument("--n-epochs",        type=int,   default=4)
    parser.add_argument("--minibatch-size",  type=int,   default=256)
    parser.add_argument("--clip-coef",       type=float, default=0.1)
    parser.add_argument("--value-coef",      type=float, default=0.5)
    parser.add_argument("--entropy-coef",    type=float, default=0.01)
    parser.add_argument("--max-grad-norm",   type=float, default=0.5)

    # Discount / GAE
    parser.add_argument("--gamma",           type=float, default=0.99)
    parser.add_argument("--gae-lambda",      type=float, default=0.95)

    # Optimiser
    parser.add_argument("--learning-rate",   type=float, default=2.5e-4)
    parser.add_argument("--anneal-lr",       action=argparse.BooleanOptionalAction, default=True)

    # Infrastructure
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--cpu",             action="store_true",
                        help="Force CPU even if CUDA is available")
    parser.add_argument("--track",           action="store_true",
                        help="Log to TensorBoard")
    parser.add_argument("--save-interval",   type=int,   default=50,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--save-dir",        type=str,   default=os.path.join("models", "ppo"),
                        help="Directory to save checkpoints")

    args = parser.parse_args()
    train(args)
