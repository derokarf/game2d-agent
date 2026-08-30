"""
PPO training for the local-map CNN agent (agent v2).

The agent observes an egocentric local occupancy map + a global goal vector
(see docs/agent_v2_local_map_spec.md).  The interpolated geodesic field is the
reward teacher (training-only); the CNN learns the routing.

Usage:
    python scripts/train_map.py
    python scripts/train_map.py --total-timesteps 5_000_000 --random-counts \
        --n-obstacles 5 --n-targets 5 --log-file logs/train_map_run1.log
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

from envs.local_map_env import LocalMapGameEnv, MAP_SIZE, GLOBAL_DIM
from envs.games import game_kwargs
from models.cnn_map_policy import CNNMapActorCritic
from models.ppo_map import VectorMapPPO
from models.ppo import PPOConfig

MAP_SHAPE = (2, MAP_SIZE, MAP_SIZE)


class RunningMeanStd:
    """Welford estimator for return normalisation."""
    def __init__(self):
        self.mean = 0.0; self.var = 1.0; self.count = 0
    def update(self, x):
        bm, bv, bc = float(x.mean()), float(x.var()), len(x)
        tot = self.count + bc
        delta = bm - self.mean
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta**2 * self.count * bc / tot) / tot
        self.count = tot
    @property
    def std(self):
        return float(np.sqrt(self.var + 1e-8))


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _log(msg, log_file):
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def _split(obs_dict, device):
    """SyncVectorEnv Dict obs -> (map tensor, global tensor)."""
    m = torch.from_numpy(np.asarray(obs_dict["map"])).float().to(device)
    g = torch.from_numpy(np.asarray(obs_dict["global"])).float().to(device)
    return m, g


def save_checkpoint(policy, optimizer, n_actions, iteration, global_step, best, save_dir, tag=""):
    os.makedirs(save_dir, exist_ok=True)
    name = f"checkpoint_{tag}.pt" if tag else "checkpoint_latest.pt"
    path = os.path.join(save_dir, name)
    torch.save({
        "policy_type":       "cnn_map",
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "map_size":  MAP_SIZE,
        "global_dim": GLOBAL_DIM,
        "n_actions": n_actions,
        "iteration": iteration,
        "global_step": global_step,
        "best_reward": best,
    }, path)
    return path


@torch.no_grad()
def run_eval(policy, eval_envs, eval_seeds, device, max_steps=1000):
    n = len(eval_seeds)
    obs, _ = eval_envs.reset(seed=eval_seeds)
    m, g = _split(obs, device)
    ep_rewards = np.zeros(n); ep_done = np.zeros(n, dtype=bool); final = np.full(n, np.nan)
    policy.eval()
    for _ in range(max_steps):
        action, _, _, _ = policy.get_action_and_value(m, g)
        obs, rewards, term, trunc, _ = eval_envs.step(action.cpu().numpy())
        for i in range(n):
            if not ep_done[i]:
                ep_rewards[i] += rewards[i]
                if term[i] or trunc[i]:
                    final[i] = ep_rewards[i]; ep_done[i] = True
        if ep_done.all():
            break
        m, g = _split(obs, device)
    policy.train()
    return np.where(np.isnan(final), ep_rewards, final)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        open(args.log_file, "w").close()

    env_kwargs = dict(max_steps=1000, n_obstacles=args.n_obstacles, n_targets=args.n_targets,
                      random_counts=args.random_counts, **game_kwargs(args.game))
    envs = SyncVectorEnv([(lambda kw: lambda: LocalMapGameEnv(**kw))(env_kwargs)
                          for _ in range(args.num_envs)])
    eval_seeds = list(range(1000, 1000 + args.n_eval))
    eval_envs  = SyncVectorEnv([(lambda kw: lambda: LocalMapGameEnv(**kw))(env_kwargs)
                                for _ in range(args.n_eval)])
    n_actions = envs.single_action_space.n

    policy = CNNMapActorCritic(map_channels=2, map_size=MAP_SIZE,
                               global_dim=GLOBAL_DIM, n_actions=n_actions).to(device)
    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy_state_dict"])
        _log(f"Loaded policy  : {args.load_checkpoint}", args.log_file)

    config = PPOConfig(
        n_steps=args.n_steps, n_envs=args.num_envs, n_epochs=4, minibatch_size=256,
        gamma=0.99, gae_lambda=0.95, clip_coef=0.2, value_coef=0.5,
        entropy_coef=args.entropy_coef, max_grad_norm=0.5, learning_rate=args.lr,
        anneal_lr=not args.no_anneal_lr, clip_value_loss=True,
    )
    ppo = VectorMapPPO(policy, config, MAP_SHAPE, GLOBAL_DIM, device)

    steps_per_iter = args.n_steps * args.num_envs
    total_iters = args.total_timesteps // steps_per_iter

    _log(f"Device        : {device}", args.log_file)
    _log(f"Obs           : map {MAP_SHAPE} + global ({GLOBAL_DIM},)  (egocentric, CNN)", args.log_file)
    _log(f"Params        : {sum(p.numel() for p in policy.parameters()):,}", args.log_file)
    _log(f"Environments  : {args.num_envs}", args.log_file)
    _log(f"Total iters   : {total_iters:,}  ({total_iters*steps_per_iter:,} steps)", args.log_file)
    _log(f"Random counts : {args.random_counts}  (obs 1-{args.n_obstacles}, tgt 1-{args.n_targets})", args.log_file)
    _log(f"Reward norm   : {'off' if args.no_reward_norm else 'on'}", args.log_file)

    obs, _ = envs.reset(seed=42)
    m, g = _split(obs, device)
    done = torch.zeros(args.num_envs, device=device)

    ep_rewards = np.zeros(args.num_envs); ep_lengths = np.zeros(args.num_envs, dtype=int)
    best_mean = -np.inf; best_eval = -np.inf; global_step = 0
    t_start = time.time(); ret_rms = RunningMeanStd()

    for iteration in range(1, total_iters + 1):
        ppo.anneal_lr(iteration - 1, total_iters)
        ppo.buffer.reset()
        completed_rewards, completed_lengths = [], []

        with torch.no_grad():
            for _ in range(args.n_steps):
                action, log_prob, _ent, value = policy.get_action_and_value(m, g)
                obs, rewards, term, trunc, _ = envs.step(action.cpu().numpy())
                done_np = np.logical_or(term, trunc).astype(np.float32)

                ep_rewards += rewards; ep_lengths += 1
                for i, d in enumerate(done_np):
                    if d:
                        completed_rewards.append(float(ep_rewards[i]))
                        completed_lengths.append(int(ep_lengths[i]))
                        ep_rewards[i] = 0; ep_lengths[i] = 0

                reward_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                done_t   = torch.tensor(done_np, dtype=torch.float32, device=device)
                ppo.buffer.add(m, g, action, log_prob, reward_t, done, value)
                m, g = _split(obs, device)
                done = done_t

        global_step += steps_per_iter
        last_value = policy.get_value(m, g)
        ppo.buffer.compute_gae(last_value, done, config.gamma, config.gae_lambda)

        if not args.no_reward_norm:
            ret_np = ppo.buffer.returns.cpu().numpy().flatten()
            ret_rms.update(ret_np)
            scale = ret_rms.std
            ppo.buffer.returns    /= scale
            ppo.buffer.advantages /= scale

        metrics = ppo.update()

        elapsed = time.time() - t_start
        sps = int(global_step / elapsed)
        lr_now = ppo.optimizer.param_groups[0]["lr"]
        ep_rew_str = f"{np.mean(completed_rewards):>7.2f}" if completed_rewards else "    nan"
        ep_len_str = f"{int(np.mean(completed_lengths)):>6}" if completed_lengths else "   nan"
        _log(
            f"iter {iteration:>5} | steps {global_step:>10,} | sps {sps:>5} | "
            f"ep_rew {ep_rew_str} | ep_len {ep_len_str} | n_eps {len(completed_rewards):>4} | "
            f"pol {metrics['policy_loss']:>8.4f} | val {metrics['value_loss']:>8.4f} | "
            f"ent {metrics['entropy_loss']:>8.4f} | kl {metrics['approx_kl']:>9.6f} | "
            f"ev {metrics['explained_variance']:>6.3f} | lr {lr_now:.2e}",
            args.log_file,
        )
        if iteration % 50 == 0:
            remaining = (total_iters - iteration) * (elapsed / iteration)
            _log(f"  eta {datetime.timedelta(seconds=int(remaining))}", args.log_file)

        if completed_rewards:
            mr = float(np.mean(completed_rewards))
            if mr > best_mean:
                best_mean = mr
                save_checkpoint(policy, ppo.optimizer, n_actions, iteration, global_step,
                                best_mean, args.save_dir, tag="best")

        if iteration % args.eval_interval == 0:
            er = run_eval(policy, eval_envs, eval_seeds, device)
            em, es = float(np.mean(er)), float(np.std(er))
            _log(f"  [eval] mean {em:>8.2f} | std {es:>7.2f} | min {float(er.min()):>8.2f} | "
                 f"max {float(er.max()):>8.2f}", args.log_file)
            if em > best_eval:
                best_eval = em
                path = save_checkpoint(policy, ppo.optimizer, n_actions, iteration, global_step,
                                       best_eval, args.save_dir, tag="best_eval")
                _log(f"  [eval] new best eval {best_eval:.3f} -> {path}", args.log_file)

    save_checkpoint(policy, ppo.optimizer, n_actions, total_iters, global_step,
                    best_mean, args.save_dir, tag="final")
    _log(f"\nTraining complete — {global_step:,} steps in {time.time()-t_start:.1f}s", args.log_file)
    _log(f"Best ep_rew   : {best_mean:.3f}", args.log_file)
    _log(f"Best eval_rew : {best_eval:.3f}", args.log_file)
    envs.close(); eval_envs.close()
    import os as _os; _os._exit(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train local-map CNN agent (v2)")
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--entropy-coef", type=float, default=0.03)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--game", default="collect2d", help="game preset (see envs/games.py)")
    p.add_argument("--run",  default="run1", help="run id; sets default output paths")
    p.add_argument("--save-dir", default=None, help="default: runs/<game>/local_map/<run>/ckpt")
    p.add_argument("--log-file", default=None, help="default: runs/<game>/local_map/<run>/train.log")
    p.add_argument("--eval-interval", type=int, default=20)
    p.add_argument("--n-eval", type=int, default=32)
    p.add_argument("--n-obstacles", type=int, default=5)
    p.add_argument("--n-targets", type=int, default=5)
    p.add_argument("--random-counts", action="store_true")
    p.add_argument("--no-reward-norm", action="store_true")
    p.add_argument("--load-checkpoint", type=str, default=None)
    args = p.parse_args()
    run_dir = f"runs/{args.game}/local_map/{args.run}"
    if args.save_dir is None:
        args.save_dir = f"{run_dir}/ckpt"
    if args.log_file is None:
        args.log_file = f"{run_dir}/train.log"
    train(args)
