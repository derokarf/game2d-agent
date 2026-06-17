"""
Render trained pixel game agent episodes and save as GIF.

Usage:
    python scripts/play_pixel.py
    python scripts/play_pixel.py --checkpoint models/ppo/checkpoint_best.pt
    python scripts/play_pixel.py --episodes 5 --out recordings/pixel.gif
    python scripts/play_pixel.py --live          # pygame window (needs DISPLAY)
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

from envs.pixel_env import PixelGameEnv
from envs.wrappers import make_env
from models.policy import CNNActorCritic


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_policy(path: str) -> CNNActorCritic:
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    policy = CNNActorCritic(obs_shape=ckpt["obs_shape"], n_actions=ckpt["n_actions"])
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


def make_pixel_env(render_mode: str, max_steps: int = 1000):
    env = make_env(
        env_factory=lambda: PixelGameEnv(frame_size=(84, 84), render_mode=render_mode, max_steps=max_steps),
        frame_size=(84, 84),
        num_stack=4,
        reward_clip=(-1.0, 1.0),
        episodic_life=False,  # full episodes for evaluation
    )
    return env


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_episode(policy: CNNActorCritic, env, device: torch.device) -> tuple[float, int, list]:
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    frames: list[np.ndarray] = []

    while True:
        # (H, W, C) → (1, C, H, W)
        obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(device)
        action = policy.act(obs_t)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            break

    return total_reward, steps, frames, info


# ---------------------------------------------------------------------------
# GIF export
# ---------------------------------------------------------------------------

def save_gif(frames: list[np.ndarray], path: str, fps: int = 15) -> None:
    import imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"GIF saved → {path}  ({len(frames)} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def play(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    print(f"Checkpoint : {args.checkpoint}")
    policy = load_policy(args.checkpoint)
    policy.to(device)

    render_mode = "human" if args.live else "rgb_array"
    all_frames: list[np.ndarray] = []
    results: list[tuple[float, int, int]] = []

    print(f"Rendering {args.episodes} episode(s) ...\n")

    for ep in range(1, args.episodes + 1):
        env = make_pixel_env(render_mode, max_steps=args.max_steps)
        reward, steps, frames, info = run_episode(policy, env, device)
        env.close()

        score = info.get("game_state", {}).get("score", 0)
        all_frames.extend(frames)
        results.append((reward, steps, score))
        print(f"  ep {ep:>2} | reward {reward:>7.2f} | steps {steps:>5} | score {score}")

    mean_rew = np.mean([r for r, _, _ in results])
    mean_score = np.mean([s for _, _, s in results])
    print(f"\n  Mean reward : {mean_rew:.2f}")
    print(f"  Mean score  : {mean_score:.1f}  (full target waves cleared)")

    if not args.live and all_frames:
        save_gif(all_frames, args.out, fps=args.fps)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render pixel game PPO agent")
    parser.add_argument("--checkpoint", default="models/ppo/checkpoint_best.pt")
    parser.add_argument("--episodes",   type=int,  default=5)
    parser.add_argument("--max-steps",  type=int,  default=1000)
    parser.add_argument("--out",        default="recordings/pixel.gif")
    parser.add_argument("--fps",        type=int,  default=15)
    parser.add_argument("--live",       action="store_true",
                        help="Open pygame window (requires DISPLAY env var)")
    args = parser.parse_args()
    play(args)