"""
Watch a trained PPO agent play the 2D game.

Usage:
    python scripts/play.py --model-path models/ppo/checkpoint_500000.pt
    python scripts/play.py --model-path models/ppo/checkpoint_500000.pt --save-video
    python scripts/play.py --model-path models/ppo/checkpoint_500000.pt --no-render

The checkpoint is a plain PyTorch file saved by the PPO training loop:
    torch.save({"policy_state_dict": policy.state_dict(), ...}, path)

No Stable Baselines3 dependency.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from envs.pixel_env import PixelGameEnv
from envs.wrappers import make_env
from torch import device

# ---------------------------------------------------------------------------
# Policy stub
#
# play.py needs to load whatever nn.Module the PPO training script saves.
# We keep a minimal interface here: the module must expose a method
#   act(obs_tensor) -> action (int)
# The actual CNN + actor/critic heads will be defined in models/policy.py
# (Week 2/3).  Until that module exists, we provide a RandomPolicy fallback
# so this script is runnable end-to-end right now.
# ---------------------------------------------------------------------------


class RandomPolicy:
    """Fallback: samples uniformly from the action space. Useful for smoke-testing."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def act(self, obs_tensor: torch.Tensor) -> int:  # noqa: ARG002
        return int(np.random.randint(0, self.n_actions))

    def to(self, device):  # noqa: ANN001
        return self


def load_policy(model_path: str, n_actions: int, device: torch.device):
    """
    Load a PPO policy from a checkpoint produced by the training loop.

    Expected checkpoint format (saved with torch.save):
        {
            "policy_state_dict": OrderedDict,   # nn.Module weights
            "n_actions": int,                   # action-space size
            "obs_shape": tuple,                 # (C, H, W) — channel-first
        }

    If the file does not exist, falls back to RandomPolicy with a warning.
    """
    from models.policy import CNNActorCritic

    if not os.path.isfile(model_path):
        print(f"[WARNING] Checkpoint not found: {model_path}")
        print("[WARNING] Falling back to RandomPolicy (for testing only).")
        return RandomPolicy(n_actions)

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    obs_shape = tuple(ckpt["obs_shape"])   # e.g. (4, 84, 84) — (C, H, W)
    policy = CNNActorCritic(obs_shape, ckpt["n_actions"])  # type: ignore[arg-type]
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Core play loop
# ---------------------------------------------------------------------------


def play(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    # Build the wrapped environment (identical preprocessing as training)
    render_mode = None if args.no_render else "rgb_array"
    env = make_env(
        env_factory=lambda: PixelGameEnv(
            frame_size=(84, 84),
            render_mode=render_mode,
            max_steps=args.max_steps,
        ),
        frame_size=(84, 84),
        num_stack=4,
        reward_clip=(-1.0, 1.0),
        episodic_life=False,  # single eval episode — don't truncate on life loss
    )

    n_actions = env.action_space.n
    policy = load_policy(args.model_path, n_actions, device)
    policy.to(device)

    # --- episode loop -------------------------------------------------------
    obs, info = env.reset()
    total_reward = 0.0
    raw_total_reward = 0.0
    step = 0
    frames: list[np.ndarray] = []

    print(f"\nModel : {args.model_path}")
    print(f"Device: {device}")
    print("-" * 50)

    try:
        while True:
            # obs is (H, W, C) float32 → policy expects (1, C, H, W)
            obs_tensor = (
                torch.from_numpy(obs)  # (H, W, C)
                .permute(2, 0, 1)  # (C, H, W)
                .unsqueeze(0)  # (1, C, H, W)
                .to(device)
            )
            action = policy.act(obs_tensor)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            raw_total_reward += info.get("raw_reward", reward)
            step += 1

            # Collect frames for video (render_frame returns raw RGB before wrappers)
            if not args.no_render:
                # Unwrap to the base env to grab the full-resolution RGB frame
                raw_frame = env.unwrapped._render_frame()
                frames.append(raw_frame)

            if step % 50 == 0:
                score = info.get("game_state", {}).get("score", "?")
                print(
                    f"Step {step:>5} | "
                    f"Clipped reward: {total_reward:>8.2f} | "
                    f"Raw reward: {raw_total_reward:>8.2f} | "
                    f"Score: {score}"
                )

            if terminated or truncated:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print("-" * 50)
    print(f"Episode complete")
    print(f"  Steps        : {step}")
    print(f"  Clipped reward: {total_reward:.2f}")
    print(f"  Raw reward    : {raw_total_reward:.2f}")
    print(f"  Final score   : {info.get('game_state', {}).get('score', '?')}")

    # --- optional video save ------------------------------------------------
    if args.save_video and frames:
        try:
            import imageio
        except ImportError:
            print("[WARNING] imageio not installed; skipping video save.")
            print("          pip install imageio[ffmpeg]")
        else:
            os.makedirs("recordings", exist_ok=True)
            video_path = os.path.join("recordings", "gameplay.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"\nVideo saved: {video_path}")

    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Watch a trained PPO agent play the 2D game"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo/checkpoint_final.pt",
        help="Path to a PyTorch checkpoint (.pt) produced by train.py",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum steps per episode (default: 5000)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable on-screen rendering (faster, for headless servers)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save episode as recordings/gameplay.mp4",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )
    args = parser.parse_args()

    play(args)
