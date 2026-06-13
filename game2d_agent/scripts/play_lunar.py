"""
Render trained LunarLander agent episodes and save as GIF.

Usage:
    python scripts/play_lunar.py
    python scripts/play_lunar.py --checkpoint models/lunar_ppo/lunar_checkpoint_best.pt
    python scripts/play_lunar.py --episodes 5 --out recordings/lunar.gif
    python scripts/play_lunar.py --live          # open pygame window (needs DISPLAY)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch

from models.mlp_policy import MLPActorCritic


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_policy_and_rms(path: str):
    ckpt = torch.load(path, weights_only=False)
    policy = MLPActorCritic(obs_dim=ckpt["obs_dim"], n_actions=ckpt["n_actions"])
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    obs_rms = ckpt.get("obs_rms")   # None for old checkpoints without rms
    return policy, obs_rms


def make_env(render_mode: str, obs_rms: dict | None) -> gym.Env:
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    env = gym.wrappers.NormalizeObservation(env)
    if obs_rms is not None:
        env.obs_rms.mean  = obs_rms["mean"]
        env.obs_rms.var   = obs_rms["var"]
        env.obs_rms.count = obs_rms["count"]
    return env


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_episode(policy: MLPActorCritic, env: gym.Env) -> tuple[float, int, list]:
    """
    Run one episode. Returns (total_reward, n_steps, rgb_frames).
    rgb_frames is populated only when env has render_mode="rgb_array".
    """
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    frames: list[np.ndarray] = []

    while True:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        action = policy.act(obs_t)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            break

    return total_reward, steps, frames


# ---------------------------------------------------------------------------
# GIF export
# ---------------------------------------------------------------------------

def save_gif(all_frames: list[np.ndarray], path: str, fps: int = 30) -> None:
    import imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, all_frames, fps=fps, loop=0)
    print(f"GIF saved → {path}  ({len(all_frames)} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def play(args: argparse.Namespace) -> None:
    print(f"Checkpoint : {args.checkpoint}")
    policy, obs_rms = load_policy_and_rms(args.checkpoint)

    if obs_rms is None:
        print("[WARNING] Checkpoint has no obs_rms — observations may be mis-scaled.")

    render_mode = "human" if args.live else "rgb_array"
    all_frames: list[np.ndarray] = []
    results: list[tuple[float, int]] = []

    print(f"Rendering {args.episodes} episode(s) …\n")

    for ep in range(1, args.episodes + 1):
        env = make_env(render_mode, obs_rms)
        reward, steps, frames = run_episode(policy, env)
        env.close()

        all_frames.extend(frames)
        results.append((reward, steps))
        status = "✓ LANDED" if reward >= 200 else "✗ crashed"
        print(f"  ep {ep:>2} | {status} | reward {reward:>8.1f} | steps {steps}")

    print(f"\n  Mean reward : {np.mean([r for r, _ in results]):.1f}")
    print(f"  Solved ≥200 : {sum(r >= 200 for r, _ in results)}/{args.episodes}")

    if not args.live and all_frames:
        save_gif(all_frames, args.out, fps=args.fps)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render LunarLander PPO agent")
    parser.add_argument("--checkpoint", default="models/lunar_ppo/lunar_checkpoint_final.pt")
    parser.add_argument("--episodes",   type=int, default=3)
    parser.add_argument("--out",        default="recordings/lunar.gif")
    parser.add_argument("--fps",        type=int, default=30)
    parser.add_argument("--live",       action="store_true",
                        help="Open live pygame window (requires DISPLAY env var)")
    args = parser.parse_args()
    play(args)