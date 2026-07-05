"""
Play and record episodes using the state-based GRU policy (no CNN).

The agent receives the exact 26-dim game state vector directly from the
engine — player position, 5 targets × (dx,dy,dist), 3 obstacles × (dx,dy,dist).
Hidden state h is carried across timesteps within each episode (reset at start).

Usage:
    python scripts/play_with_pure_gru_no_cnn.py --checkpoint models/state_gru/checkpoint_best_eval.pt
    python scripts/play_with_pure_gru_no_cnn.py --checkpoint models/state_gru/checkpoint_best_eval.pt \
        --episodes 5 --out recordings/state_gru_best.gif --temperature 0.5
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
from torch.distributions import Categorical

from envs.state_env import StateGameEnv
from models.gru_policy import GRUActorCritic


def load_policy(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    assert ckpt.get("policy_type") == "gru", \
        f"Expected policy_type='gru', got '{ckpt.get('policy_type')}'. " \
        f"Use play_with_pure_mlp_no_cnn.py for MLP checkpoints."

    policy = GRUActorCritic(
        obs_dim    = ckpt["obs_dim"],
        hidden_dim = ckpt["hidden_dim"],
        n_actions  = ckpt["n_actions"],
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    policy.to(device)
    print(f"Policy loaded  : {checkpoint_path}")
    print(f"Obs dim        : {ckpt['obs_dim']}  (direct game state, no CNN)")
    print(f"GRU hidden dim : {ckpt['hidden_dim']}")
    print(f"Actions        : {ckpt['n_actions']}")
    return policy


def run_episode(policy, env, device, temperature=1.0):
    obs_np, _ = env.reset()
    h = policy.initial_state(1, device)
    total_reward = 0.0
    steps = 0
    frames = []

    while True:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _, h = policy.forward(obs, h)
            if temperature != 1.0:
                action = Categorical(logits=logits / temperature).sample()
            else:
                action = Categorical(logits=logits).sample()

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

    mode = f"temperature={args.temperature}" if args.temperature != 1.0 else "stochastic"
    print(f"Rendering {args.episodes} episode(s) ...  [{mode}]\n")

    all_frames = []
    results = []

    for ep in range(1, args.episodes + 1):
        env = StateGameEnv(max_steps=args.max_steps, render_mode="rgb_array")
        reward, steps, score, frames = run_episode(policy, env, device, args.temperature)
        env.close()

        all_frames.extend(frames)
        results.append((reward, steps, score))
        print(f"  ep {ep:>2} | reward {reward:>7.2f} | steps {steps:>5} | score {score}")

    print(f"\n  Mean reward : {np.mean([r for r, _, _ in results]):.2f}")
    print(f"  Mean score  : {np.mean([s for _, _, s in results]):.1f}")

    if all_frames:
        save_gif(all_frames, args.out, fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="models/state_gru/checkpoint_best_eval.pt")
    parser.add_argument("--episodes",    type=int,   default=5)
    parser.add_argument("--max-steps",   type=int,   default=1000)
    parser.add_argument("--out",         default="recordings/state_gru_best.gif")
    parser.add_argument("--fps",         type=int,   default=15)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    play(args)
