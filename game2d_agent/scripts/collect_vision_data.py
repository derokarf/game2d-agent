"""
Collect (frame, label) pairs by running a random agent.

Each sample:
  frame : (84, 84, 3) uint8   — RGB render of the current game state
  label : (18,) float32       — normalized scene description

Label layout (18 values):
  [0]     player_x   / screen_w
  [1]     player_y   / screen_h
  [2..4]  nearest target:  dx/screen_w, dy/screen_h, dist/max_dist
  [5..7]  2nd target:      dx, dy, dist
  [8..10] 3rd target:      dx, dy, dist
  [11..13] 4th target:     dx, dy, dist
  [14..16] 5th target:     dx, dy, dist
  [17]    nearest obstacle distance / max_dist

Usage:
    python scripts/collect_vision_data.py
    python scripts/collect_vision_data.py --episodes 500 --out data/vision/dataset.npz
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from envs.pixel_env import PixelGameEnv

SCREEN_W = 640
SCREEN_H = 480
MAX_DIST = np.sqrt(SCREEN_W ** 2 + SCREEN_H ** 2)


def extract_label(state: dict) -> np.ndarray:
    """Convert game_state → flat float32 label vector of shape (18,)."""
    px, py = state["player_x"], state["player_y"]

    label = [px / SCREEN_W, py / SCREEN_H]

    # Sort targets by distance to player
    def _dist(t):
        return np.sqrt((t["x"] - px) ** 2 + (t["y"] - py) ** 2)

    targets = sorted(state["targets"], key=_dist)

    for i in range(5):
        if i < len(targets):
            dx = (targets[i]["x"] - px) / SCREEN_W
            dy = (targets[i]["y"] - py) / SCREEN_H
            d  = _dist(targets[i]) / MAX_DIST
        else:
            dx, dy, d = 0.0, 0.0, 1.0
        label.extend([dx, dy, d])

    # Nearest obstacle distance
    min_obs = float("inf")
    for obs in state["obstacles"]:
        cx = np.clip(px, obs["x"], obs["x"] + obs["w"])
        cy = np.clip(py, obs["y"], obs["y"] + obs["h"])
        min_obs = min(min_obs, np.sqrt((px - cx) ** 2 + (py - cy) ** 2))

    label.append(min_obs / MAX_DIST if min_obs != float("inf") else 1.0)

    return np.array(label, dtype=np.float32)


def collect(args: argparse.Namespace) -> None:
    frames_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    env = PixelGameEnv(frame_size=(84, 84), render_mode="rgb_array", max_steps=args.max_steps)

    for ep in range(1, args.episodes + 1):
        env.reset()

        while True:
            frame = env.render()                        # (84, 84, 3) uint8
            label = extract_label(env.game_state)       # (18,) float32
            frames_list.append(frame)
            labels_list.append(label)

            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        if ep % 50 == 0 or ep == args.episodes:
            print(f"  ep {ep:>4}/{args.episodes}  |  frames collected: {len(frames_list):,}")

    env.close()

    frames_arr = np.stack(frames_list)   # (N, 84, 84, 3)
    labels_arr = np.stack(labels_list)   # (N, 18)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, frames=frames_arr, labels=labels_arr)

    print(f"\nDataset saved → {args.out}")
    print(f"  frames : {frames_arr.shape}  dtype={frames_arr.dtype}")
    print(f"  labels : {labels_arr.shape}  dtype={labels_arr.dtype}")
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"  file size: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect vision pre-training data")
    parser.add_argument("--episodes",  type=int, default=200,
                        help="Number of random episodes to run (default: 200)")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per episode (default: 1000)")
    parser.add_argument("--out",       default="data/vision/dataset.npz",
                        help="Output path for the .npz dataset")
    args = parser.parse_args()

    print(f"Collecting {args.episodes} episodes × up to {args.max_steps} steps ...")
    collect(args)
