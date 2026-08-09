"""
Play and record episodes using the state-based MLP policy (no CNN).

The agent receives the exact 26-dim game state vector directly from the
engine — player position, 5 targets × (dx,dy,dist), 3 obstacles × (dx,dy,dist).

Usage:
    python scripts/play_state.py --checkpoint models/state_mlp_diag/checkpoint_best_eval.pt
    python scripts/play_state.py --checkpoint models/state_mlp_diag/checkpoint_best_eval.pt --episodes 5 --out recordings/state_best.gif
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch

from envs.state_env import StateGameEnv
from models.mlp_policy import MLPActorCritic


def load_policy(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy = MLPActorCritic(
        obs_dim   = ckpt["obs_dim"],
        n_actions = ckpt["n_actions"],
        hidden    = (256, 128),
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    policy.to(device)
    print(f"Policy loaded  : {checkpoint_path}")
    print(f"Obs dim        : {ckpt['obs_dim']}  (direct game state, no CNN)")
    return policy


def run_episode(policy, env, device, temperature=1.0):
    obs_np, _ = env.reset()
    total_reward = 0.0
    steps = 0
    frames = []

    while True:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            if temperature != 1.0:
                logits, _ = policy.forward(obs)
                action = torch.distributions.Categorical(logits=logits / temperature).sample()
            else:
                action, _, _, _ = policy.get_action_and_value(obs)

        obs_np, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    score = info.get("game_state", {}).get("score", 0)
    return total_reward, steps, score, frames


def save_gif(frames, path, fps=15):
    import imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"GIF saved → {path}  ({len(frames)} frames)")


def play(args):
    device = torch.device("cpu")
    policy = load_policy(args.checkpoint, device)

    if args.temperature != 1.0:
        mode = f"temperature={args.temperature}"
    else:
        mode = "stochastic"
    print(f"Rendering {args.episodes} episode(s) ...  [{mode}]\n")

    all_frames = []
    results = []

    for ep in range(1, args.episodes + 1):
        env = StateGameEnv(max_steps=args.max_steps, render_mode="rgb_array",
                           n_obstacles=args.n_obstacles, n_targets=args.n_targets,
                           random_counts=args.random_counts)
        reward, steps, score, frames = run_episode(policy, env, device, args.temperature)
        env.close()

        all_frames.extend(frames)
        results.append((reward, steps, score))
        print(f"  ep {ep:>2} | reward {reward:>7.2f} | steps {steps:>5} | score {score}")

    mean_rew   = np.mean([r for r, _, _ in results])
    mean_score = np.mean([s for _, _, s in results])
    print(f"\n  Mean reward : {mean_rew:.2f}")
    print(f"  Mean score  : {mean_score:.1f}")

    if all_frames:
        save_gif(all_frames, args.out, fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="models/state_mlp_diag/checkpoint_best_eval.pt")
    parser.add_argument("--episodes",    type=int,   default=5)
    parser.add_argument("--max-steps",   type=int,   default=1000)
    parser.add_argument("--out",         default="recordings/state_best.gif")
    parser.add_argument("--fps",         type=int,   default=15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-obstacles",  type=int, default=3)
    parser.add_argument("--n-targets",    type=int, default=5)
    parser.add_argument("--random-counts", action="store_true",
                        help="Randomize obstacle & target counts each level (1..n)")
    args = parser.parse_args()
    play(args)
