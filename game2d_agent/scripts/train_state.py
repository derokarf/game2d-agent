"""
PPO training on true game state (diagnostic run).

Bypasses the CNN entirely — the MLP receives a perfect 26-dim state vector
(player pos, 5 targets × dx/dy/dist, 3 obstacles × dx/dy/dist) directly
from the game engine.

If the MLP learns well here, the bottleneck in previous runs was the CNN
perception, not the MLP or the reward.  If it still fails, the problem is
in the reward / MLP capacity.

Usage:
    python scripts/train_state.py
    python scripts/train_state.py --total-timesteps 1_000_000 --log-file logs/train_state_run1.log
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
from gymnasium.vector import SyncVectorEnv

from envs.state_env import StateGameEnv, OBS_DIM
from models.mlp_policy import MLPActorCritic
from models.ppo_mlp import VectorPPO
from models.ppo import PPOConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Welford online estimator — tracks variance of returns across all iterations."""
    def __init__(self):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        batch_mean  = float(x.mean())
        batch_var   = float(x.var())
        batch_count = len(x)
        total       = self.count + batch_count
        delta       = batch_mean - self.mean
        self.mean  += delta * batch_count / total
        self.var    = (
            self.var * self.count
            + batch_var * batch_count
            + delta ** 2 * self.count * batch_count / total
        ) / total
        self.count  = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + 1e-8))


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _log(msg: str, log_file: str | None) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def save_checkpoint(policy, optimizer, iteration, global_step, best_reward, save_dir, tag=""):
    os.makedirs(save_dir, exist_ok=True)
    name = f"checkpoint_{tag}.pt" if tag else "checkpoint_latest.pt"
    path = os.path.join(save_dir, name)
    torch.save({
        "policy_state_dict":    policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_dim":              OBS_DIM,
        "n_actions":            policy.actor_head.out_features,
        "iteration":            iteration,
        "global_step":          global_step,
        "best_reward":          best_reward,
    }, path)
    return path


@torch.no_grad()
def run_eval(policy, eval_envs, eval_seeds, device, max_steps=1000):
    """Run one episode per fixed seed, return array of episode rewards."""
    n = len(eval_seeds)
    obs_np, _ = eval_envs.reset(seed=eval_seeds)
    obs = torch.from_numpy(obs_np).float().to(device)

    ep_rewards = np.zeros(n)
    ep_done    = np.zeros(n, dtype=bool)
    final_rew  = np.full(n, np.nan)

    policy.eval()
    for _ in range(max_steps):
        action, _, _, _ = policy.get_action_and_value(obs)
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

    policy.train()
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

    # ── Policy + PPO ─────────────────────────────────────────────────────────
    n_actions = envs.single_action_space.n
    policy = MLPActorCritic(
        obs_dim   = OBS_DIM,
        n_actions = n_actions,
        hidden    = (256, 128),
    ).to(device)

    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy_state_dict"])
        _log(f"Loaded policy  : {args.load_checkpoint}", args.log_file)

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
        anneal_lr      = not args.no_anneal_lr,
        clip_value_loss= True,
    )

    ppo = VectorPPO(policy, config, obs_shape=(OBS_DIM,), device=device)

    steps_per_iter = args.n_steps * args.num_envs
    total_iters    = args.total_timesteps // steps_per_iter

    _log(f"Device        : {device}", args.log_file)
    _log(f"Obs dim       : {OBS_DIM}  (true game state, no CNN)", args.log_file)
    _log(f"Curriculum    : {args.n_obstacles} obstacle(s), {args.n_targets} target(s)", args.log_file)
    _log(f"Environments  : {args.num_envs}", args.log_file)
    _log(f"Steps/iter    : {steps_per_iter:,}", args.log_file)
    _log(f"Total iters   : {total_iters:,}", args.log_file)
    _log(f"Total steps   : {total_iters * steps_per_iter:,}", args.log_file)
    _log(f"Eval envs     : {args.n_eval}  (seeds {eval_seeds[0]}–{eval_seeds[-1]}, every {args.eval_interval} iters)", args.log_file)
    _log(f"Reward norm   : {'off' if args.no_reward_norm else 'on (RunningMeanStd over returns)'}", args.log_file)

    # ── Initial obs ──────────────────────────────────────────────────────────
    obs_np, _ = envs.reset(seed=42)
    obs  = torch.from_numpy(obs_np).float().to(device)
    done = torch.zeros(args.num_envs, device=device)

    ep_rewards = np.zeros(args.num_envs)
    ep_lengths = np.zeros(args.num_envs, dtype=int)

    best_mean_reward = -np.inf
    best_eval_reward = -np.inf
    global_step      = 0
    t_start          = time.time()
    ret_rms          = RunningMeanStd()  # tracks return variance for normalization

    # ── Main loop ────────────────────────────────────────────────────────────
    for iteration in range(1, total_iters + 1):
        ppo.anneal_lr(iteration - 1, total_iters)
        ppo.buffer.reset()

        completed_rewards: list[float] = []
        completed_lengths: list[int]   = []

        with torch.no_grad():
            for _ in range(args.n_steps):
                action, log_prob, _ent, value = policy.get_action_and_value(obs)

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

                ppo.buffer.add(obs, action, log_prob, reward_t, done_t, value)
                obs  = next_obs
                done = done_t

        global_step += steps_per_iter

        last_value = policy.get_value(obs)
        ppo.buffer.compute_gae(last_value, done, config.gamma, config.gae_lambda)

        if not args.no_reward_norm:
            ret_np = ppo.buffer.returns.cpu().numpy().flatten()
            ret_rms.update(ret_np)
            scale = ret_rms.std
            ppo.buffer.returns    /= scale
            ppo.buffer.advantages /= scale

        metrics = ppo.update()

        # ── Logging ──────────────────────────────────────────────────────────
        elapsed    = time.time() - t_start
        sps        = int(global_step / elapsed)
        lr_now     = ppo.optimizer.param_groups[0]["lr"]
        ep_rew_str = f"{np.mean(completed_rewards):>7.2f}" if completed_rewards else "    nan"
        ep_len_str = f"{int(np.mean(completed_lengths)):>6}"  if completed_lengths else "   nan"
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

        # Save best by training ep_rew
        if completed_rewards:
            mean_rew = float(np.mean(completed_rewards))
            if mean_rew > best_mean_reward:
                best_mean_reward = mean_rew
                path = save_checkpoint(policy, ppo.optimizer, iteration, global_step,
                                       best_mean_reward, args.save_dir, tag="best")
                _log(f"  [best] new best ep_rew {best_mean_reward:.3f} → {path}", args.log_file)

        # Fixed-seed evaluation
        if iteration % args.eval_interval == 0:
            eval_rewards = run_eval(policy, eval_envs, eval_seeds, device)
            eval_mean    = float(np.mean(eval_rewards))
            eval_std     = float(np.std(eval_rewards))
            _log(
                f"  [eval] mean {eval_mean:>8.2f} | std {eval_std:>7.2f} | "
                f"min {float(np.min(eval_rewards)):>8.2f} | max {float(np.max(eval_rewards)):>8.2f}",
                args.log_file,
            )
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                path = save_checkpoint(policy, ppo.optimizer, iteration, global_step,
                                       best_eval_reward, args.save_dir, tag="best_eval")
                _log(f"  [eval] new best eval {best_eval_reward:.3f} → {path}", args.log_file)

    # ── Done ─────────────────────────────────────────────────────────────────
    save_checkpoint(policy, ppo.optimizer, total_iters, global_step,
                    best_mean_reward, args.save_dir, tag="final")
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
    parser = argparse.ArgumentParser(description="Train MLP on true game state (diagnostic)")
    parser.add_argument("--total-timesteps", type=int,   default=1_000_000)
    parser.add_argument("--num-envs",        type=int,   default=8)
    parser.add_argument("--n-steps",         type=int,   default=128)
    parser.add_argument("--lr",              type=float, default=2.5e-4)
    parser.add_argument("--entropy-coef",    type=float, default=0.03)
    parser.add_argument("--no-anneal-lr",    action="store_true")
    parser.add_argument("--save-dir",        default="models/state_mlp")
    parser.add_argument("--log-file",        default="logs/train_state_run1.log")
    parser.add_argument("--eval-interval",   type=int,   default=20)
    parser.add_argument("--n-eval",          type=int,   default=16)
    parser.add_argument("--n-obstacles",     type=int,   default=3,
                        help="Number of obstacles in the game (curriculum: start low)")
    parser.add_argument("--n-targets",       type=int,   default=5,
                        help="Number of targets in the game (curriculum: start low)")
    parser.add_argument("--load-checkpoint", type=str,   default=None,
                        help="Load policy weights from this checkpoint before training")
    parser.add_argument("--no-reward-norm", action="store_true",
                        help="Disable return normalization (on by default)")
    args = parser.parse_args()
    train(args)
