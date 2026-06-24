"""
Collect (frame, label) pairs by running random agents in parallel.

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
    python scripts/collect_vision_data.py --episodes 500 --workers 8
    python scripts/collect_vision_data.py --episodes 200 --out data/vision/dataset.npz
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from envs.pixel_env import PixelGameEnv

SCREEN_W = 640
SCREEN_H = 480
MAX_DIST  = float(np.sqrt(SCREEN_W ** 2 + SCREEN_H ** 2))


def extract_label(state: dict) -> np.ndarray:
    """Convert game_state → flat float32 label vector of shape (18,)."""
    px, py = state["player_x"], state["player_y"]

    def _d(t):
        return np.sqrt((t["x"] - px) ** 2 + (t["y"] - py) ** 2)

    targets = sorted(state["targets"], key=_d)
    label   = [px / SCREEN_W, py / SCREEN_H]

    for i in range(5):
        if i < len(targets):
            label.extend([
                (targets[i]["x"] - px) / SCREEN_W,
                (targets[i]["y"] - py) / SCREEN_H,
                _d(targets[i]) / MAX_DIST,
            ])
        else:
            label.extend([0.0, 0.0, 1.0])

    min_obs = float("inf")
    for obs in state["obstacles"]:
        cx = np.clip(px, obs["x"], obs["x"] + obs["w"])
        cy = np.clip(py, obs["y"], obs["y"] + obs["h"])
        min_obs = min(min_obs, np.sqrt((px - cx) ** 2 + (py - cy) ** 2))

    label.append(min_obs / MAX_DIST if min_obs != float("inf") else 1.0)
    return np.array(label, dtype=np.float32)


# ---------------------------------------------------------------------------
# Worker — runs in a subprocess
# ---------------------------------------------------------------------------

def _worker(task: tuple) -> tuple[np.ndarray, np.ndarray]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    n_eps, max_steps, seed, wid = task
    np.random.seed(seed)

    frames: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    env = PixelGameEnv(frame_size=(84, 84), render_mode="rgb_array", max_steps=max_steps)

    for ep in range(1, n_eps + 1):
        env.reset()
        while True:
            frames.append(env.render())
            labels.append(extract_label(env.game_state))
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break

        if ep % 10 == 0:
            print(f"  [worker {wid}] ep {ep:>4}/{n_eps}  |  frames: {len(frames):>6,}", flush=True)

    env.close()
    print(f"  [worker {wid}] done  —  {len(frames):,} frames", flush=True)
    return np.stack(frames), np.stack(labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect(args: argparse.Namespace) -> None:
    n_workers = min(args.workers, mp.cpu_count(), args.episodes)
    base, rem  = divmod(args.episodes, n_workers)
    eps_per    = [base + (1 if i < rem else 0) for i in range(n_workers)]

    print(f"Episodes    : {args.episodes}")
    print(f"Workers     : {n_workers}  (episodes per worker: {eps_per})")
    print(f"Max steps   : {args.max_steps}")
    print()

    tasks = [(n, args.max_steps, i * 7919, i) for i, n in enumerate(eps_per)]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = list(pool.imap_unordered(_worker, tasks))

    all_frames = np.concatenate([r[0] for r in results])
    all_labels = np.concatenate([r[1] for r in results])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, frames=all_frames, labels=all_labels)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.out) / 1024 / 1024

    print(f"\nTotal frames  : {len(all_frames):,}")
    print(f"Dataset saved → {args.out}  ({size_mb:.1f} MB)")
    print(f"Time elapsed  : {elapsed:.1f}s  ({len(all_frames)/elapsed:.0f} frames/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect vision pre-training data")
    parser.add_argument("--episodes",  type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--workers",   type=int, default=min(4, mp.cpu_count()),
                        help=f"Parallel workers (default: min(4, cpu_count)={min(4, mp.cpu_count())})")
    parser.add_argument("--out",       default="data/vision/dataset.npz")
    args = parser.parse_args()
    collect(args)
