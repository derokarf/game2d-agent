"""
Play and record episodes using the frozen VisionEncoder + trained MLP policy.

Usage:
    python scripts/play_vision.py
    python scripts/play_vision.py --checkpoint models/ppo_vision/checkpoint_best.pt
    python scripts/play_vision.py --episodes 5 --out recordings/vision.gif
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
from models.vision_encoder import VisionEncoder
from models.mlp_policy import MLPActorCritic


def load_policy(checkpoint_path: str, encoder_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder = VisionEncoder.load(encoder_path, device=device)
    if "encoder_state_dict" in ckpt:
        # Fine-tuned encoder saved inside the checkpoint — use those weights
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        print("Encoder: loaded fine-tuned weights from checkpoint")
    else:
        print(f"Encoder: loaded base weights from {encoder_path}")
    encoder.eval()
    encoder.freeze()

    policy = MLPActorCritic(
        obs_dim   = ckpt["embed_dim"],
        n_actions = ckpt["n_actions"],
        hidden    = (256, 128),
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    policy.to(device)

    return encoder, policy


def run_episode(encoder, policy, env, device, deterministic=False, temperature=1.0):
    obs_raw, _ = env.reset()
    total_reward = 0.0
    steps = 0
    frames = []

    while True:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        t = torch.from_numpy(obs_raw).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
        with torch.no_grad():
            obs_enc = encoder(t)                              # (1, 64)
            if deterministic:
                action = torch.tensor([policy.act(obs_enc)])  # argmax — no sampling
            elif temperature != 1.0:
                logits, _ = policy.forward(obs_enc)
                action = torch.distributions.Categorical(logits=logits / temperature).sample()
            else:
                action, _, _, _ = policy.get_action_and_value(obs_enc)

        obs_raw, reward, terminated, truncated, info = env.step(action.item())
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
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Encoder    : {args.encoder}")

    encoder, policy = load_policy(args.checkpoint, args.encoder, device)

    all_frames = []
    results = []

    if args.deterministic:
        mode = "deterministic"
    elif args.temperature != 1.0:
        mode = f"temperature={args.temperature}"
    else:
        mode = "stochastic"
    print(f"Rendering {args.episodes} episode(s) ...  [{mode}]\n")
    for ep in range(1, args.episodes + 1):
        env = PixelGameEnv(frame_size=(84, 84), render_mode="rgb_array", max_steps=args.max_steps)
        reward, steps, score, frames = run_episode(
            encoder, policy, env, device, args.deterministic, args.temperature
        )
        env.close()

        all_frames.extend(frames)
        results.append((reward, steps, score))
        print(f"  ep {ep:>2} | reward {reward:>7.2f} | steps {steps:>5} | score {score}")

    mean_rew   = np.mean([r for r, _, _ in results])
    mean_score = np.mean([s for _, _, s in results])
    print(f"\n  Mean reward : {mean_rew:.2f}")
    print(f"  Mean score  : {mean_score:.1f}  (full target waves cleared)")

    if all_frames and not args.live:
        save_gif(all_frames, args.out, fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/ppo_vision/checkpoint_best.pt")
    parser.add_argument("--encoder",    default="models/vision_encoder.pt")
    parser.add_argument("--episodes",   type=int,  default=5)
    parser.add_argument("--max-steps",  type=int,  default=1000)
    parser.add_argument("--out",        default="recordings/vision.gif")
    parser.add_argument("--fps",        type=int,  default=15)
    parser.add_argument("--live",          action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                        help="Pick highest-confidence action every step (no sampling)")
    parser.add_argument("--temperature",   type=float, default=1.0,
                        help="Sampling temperature: <1 more decisive, >1 more random (default 1.0)")
    args = parser.parse_args()
    play(args)
