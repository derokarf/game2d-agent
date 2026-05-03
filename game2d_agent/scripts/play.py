import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import cv2
import imageio
from envs.pixel_env import PixelGameEnv
from stable_baselines3 import DQN


def play(args):
    model_path = args.model_path
    model = DQN.load(model_path)

    env = PixelGameEnv(
        frame_size=(84, 84),
        render_mode="human" if not args.no_render else None,
        max_steps=args.max_steps,
    )

    frames = []
    obs, info = env.reset()
    total_reward = 0
    step = 0

    print("Starting evaluation...")
    print(f"Model: {model_path}")
    print("-" * 50)

    try:
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if not args.no_render:
                frame = env._render_frame()
                frames.append(frame)

            if step % 50 == 0:
                print(f"Step {step} | Reward: {total_reward:.1f} | Score: {info['game_state']['score']}")

            if terminated or truncated:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print("-" * 50)
    print(f"Episode complete!")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final score: {info['game_state']['score']}")

    if args.save_video and len(frames) > 0:
        os.makedirs("recordings", exist_ok=True)
        video_path = os.path.join("recordings", "gameplay.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved to {video_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained DQN agent play the 2D game")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved DQN model")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    play(args)
