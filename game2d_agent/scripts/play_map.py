"""
Play and record episodes for the local-map CNN agent (agent v2).

Usage:
    python scripts/play_map.py --checkpoint models/map_run1/checkpoint_best_eval.pt \
        --episodes 6 --n-obstacles 5 --n-targets 5 --random-counts --temperature 0.15 \
        --out recordings/map_run1.gif
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

from envs.local_map_env import LocalMapGameEnv, MAP_SIZE, GLOBAL_DIM
from models.cnn_map_policy import CNNMapActorCritic


def load_policy(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    assert ckpt.get("policy_type") == "cnn_map", \
        f"Expected policy_type='cnn_map', got '{ckpt.get('policy_type')}'."
    policy = CNNMapActorCritic(map_channels=2, map_size=ckpt["map_size"],
                               global_dim=ckpt["global_dim"], n_actions=ckpt["n_actions"])
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval().to(device)
    print(f"Policy loaded  : {checkpoint_path}")
    print(f"Obs            : map (2,{ckpt['map_size']},{ckpt['map_size']}) + global ({ckpt['global_dim']},)")
    return policy


def run_episode(policy, env, device, temperature=1.0):
    obs, _ = env.reset()
    total_reward = 0.0; steps = 0; frames = []
    while True:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        m = torch.from_numpy(obs["map"]).float().unsqueeze(0).to(device)
        g = torch.from_numpy(obs["global"]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = policy.forward(m, g)
            if temperature != 1.0:
                action = Categorical(logits=logits / temperature).sample()
            else:
                action = Categorical(logits=logits).sample()
        obs, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward; steps += 1
        if terminated or truncated:
            break
    score = info.get("game_state", {}).get("score", 0)
    return total_reward, steps, score, frames


def save_gif(frames, path, fps=15):
    import imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"GIF saved -> {path}  ({len(frames)} frames)")


def play(args):
    device = torch.device("cpu")
    policy = load_policy(args.checkpoint, device)
    mode = f"temperature={args.temperature}" if args.temperature != 1.0 else "stochastic"
    print(f"Rendering {args.episodes} episode(s) ...  [{mode}]\n")

    all_frames, results = [], []
    for ep in range(1, args.episodes + 1):
        env = LocalMapGameEnv(max_steps=args.max_steps, render_mode="rgb_array",
                              n_obstacles=args.n_obstacles, n_targets=args.n_targets,
                              random_counts=args.random_counts)
        reward, steps, score, frames = run_episode(policy, env, device, args.temperature)
        env.close()
        all_frames.extend(frames); results.append((reward, steps, score))
        print(f"  ep {ep:>2} | reward {reward:>7.2f} | steps {steps:>5} | score {score}")

    print(f"\n  Mean reward : {np.mean([r for r,_,_ in results]):.2f}")
    print(f"  Mean score  : {np.mean([s for _,_,s in results]):.1f}")
    if all_frames:
        save_gif(all_frames, args.out, fps=args.fps)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/map_run1/checkpoint_best_eval.pt")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--out", default="recordings/map_run1.gif")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--n-obstacles", type=int, default=5)
    p.add_argument("--n-targets", type=int, default=5)
    p.add_argument("--random-counts", action="store_true")
    args = p.parse_args()
    play(args)
