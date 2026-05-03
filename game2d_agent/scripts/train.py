import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    EpisodicLifeEnv,
)
import argparse


class FrameStackWrapper(gymnasium.Wrapper):
    """Stacks N consecutive frames to capture motion information."""

    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = []
        h, w, c = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, c * num_stack),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.num_stack
        return self._stack_frames(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        if len(self.frames) > self.num_stack:
            self.frames.pop(0)
        return self._stack_frames(), reward, terminated, truncated, info

    def _stack_frames(self):
        import numpy as np
        return np.concatenate(self.frames, axis=-1)


def make_env(render_mode=None, frame_size=(84, 84)):
    from envs.pixel_env import PixelGameEnv

    def _init():
        env = PixelGameEnv(frame_size=frame_size, render_mode=render_mode)
        return env

    return _init


def train(args):
    save_dir = os.path.join("logs", "dqn")
    model_dir = os.path.join("models", "dqn")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Creating training environments...")
    vec_env = make_vec_env(
        make_env(frame_size=(84, 84)),
        n_envs=args.num_envs,
        seed=args.seed,
    )

    print("Initializing DQN agent...")
    policy_kwargs = dict(
        features_extractor_class=None,
        net_arch=[256, 256, 128],
    )

    model = DQN(
        "CnnPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=(4, "step"),
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=save_dir,
        verbose=1,
        seed=args.seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="dqn_model",
    )

    print("Starting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    final_model_path = os.path.join(model_dir, "dqn_final")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent on pixel-based 2D game")
    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--exploration-fraction", type=float, default=0.1)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
