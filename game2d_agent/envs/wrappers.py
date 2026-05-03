"""
Environment wrappers for pixel-based RL agents.

All wrappers follow the Gymnasium wrapper API and are designed to be
composable — stack them in the make_env() factory function.

Wrapper stack (applied bottom-up):
    PixelGameEnv          ← raw RGB, shape (H, W, 3), dtype uint8
        └─ GrayscaleWrapper    → (H, W, 1), uint8
              └─ ResizeWrapper        → (84, 84, 1), uint8
                    └─ NormalizeWrapper      → (84, 84, 1), float32, [0, 1]
                          └─ FrameStackWrapper     → (84, 84, 4), float32
                                └─ RewardClipWrapper    → reward in [-1, 1]
                                      └─ EpisodicLifeWrapper  → terminal on life loss
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import cv2
import gymnasium
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# 1. Grayscale
# ---------------------------------------------------------------------------

class GrayscaleWrapper(gymnasium.ObservationWrapper):
    """
    Convert RGB observations to single-channel grayscale.

    Why grayscale?
    - Reduces input size 3x (3 channels → 1) without losing structural info.
    - Color is rarely meaningful for policy decisions in simple 2D games.
    - Standard practice in Atari benchmarks (Mnih et al. 2015).

    Input observation:  (H, W, 3)  uint8  [0, 255]
    Output observation: (H, W, 1)  uint8  [0, 255]
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        h, w, _c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 1),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # cv2.cvtColor expects HWC RGB; returns HW
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return gray[:, :, np.newaxis]  # keep channel dim → (H, W, 1)


# ---------------------------------------------------------------------------
# 2. Resize
# ---------------------------------------------------------------------------

class ResizeWrapper(gymnasium.ObservationWrapper):
    """
    Resize spatial dimensions to a fixed target size.

    Why resize?
    - Standardises input regardless of game window resolution.
    - Reduces compute: 84×84 is the canonical Atari size, roughly 4× less
      than a 168×168 frame, while retaining enough visual detail.

    Input observation:  (H, W, C)
    Output observation: (target_h, target_w, C)
    """

    def __init__(self, env: gymnasium.Env, size: Tuple[int, int] = (84, 84)):
        super().__init__(env)
        self.size = size  # (height, width)
        _h, _w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(size[0], size[1], c),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # cv2.resize takes (width, height); our size tuple is (h, w)
        resized = cv2.resize(
            obs,
            (self.size[1], self.size[0]),  # (width, height)
            interpolation=cv2.INTER_AREA,
        )
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        return resized


# ---------------------------------------------------------------------------
# 3. Normalize
# ---------------------------------------------------------------------------

class NormalizeObsWrapper(gymnasium.ObservationWrapper):
    """
    Scale pixel values from [0, 255] uint8 → [0.0, 1.0] float32.

    Why normalize?
    - Neural network weights are initialised near zero; unnormalised [0,255]
      inputs would require orders-of-magnitude smaller initial weights,
      creating instability.
    - Keeps activations in a stable range, improving gradient flow.

    Input observation:  (H, W, C)  uint8   [0, 255]
    Output observation: (H, W, C)  float32 [0.0, 1.0]
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(h, w, c),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# 4. Frame Stack
# ---------------------------------------------------------------------------

class FrameStackWrapper(gymnasium.Wrapper):
    """
    Stack the last N observations along the channel axis.

    Why frame stacking?
    - A single frame violates the Markov property: the agent cannot infer
      velocity or direction from a static snapshot.
    - Stacking 4 frames gives the CNN enough temporal context to estimate
      motion without an explicit recurrent architecture.
    - The stacked observation is (H, W, N) where N = num_stack.

    On reset: the buffer is filled with N copies of the initial frame.
    On step:  the oldest frame is dropped, the newest is appended (FIFO).

    Input observation:  (H, W, C)
    Output observation: (H, W, C * num_stack)
    """

    def __init__(self, env: gymnasium.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self._frames: deque[np.ndarray] = deque(maxlen=num_stack)

        h, w, c = env.observation_space.shape
        low = np.repeat(env.observation_space.low, num_stack, axis=-1)
        high = np.repeat(env.observation_space.high, num_stack, axis=-1)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(h, w, c * num_stack),
            dtype=env.observation_space.dtype,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_buffer(self, frame: np.ndarray) -> None:
        """Fill deque with `num_stack` copies of `frame` (used on reset)."""
        for _ in range(self.num_stack):
            self._frames.append(frame)

    def _stacked_obs(self) -> np.ndarray:
        """Concatenate buffered frames along the last (channel) axis."""
        return np.concatenate(list(self._frames), axis=-1)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._fill_buffer(obs)
        return self._stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)  # deque auto-evicts oldest
        return self._stacked_obs(), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# 5. Reward Clipping
# ---------------------------------------------------------------------------

class RewardClipWrapper(gymnasium.RewardWrapper):
    """
    Clip rewards to the range [low, high] (default [-1, 1]).

    Why clip rewards?
    - Raw game scores can vary wildly (+50 for clearing a wave, -0.01 per
      step, etc.). This causes huge gradient variance in the value loss.
    - Clipping collapses the magnitude problem: the policy still learns the
      correct ordering of outcomes, but gradients stay bounded.
    - Trade-off: you lose precise reward magnitude information. For most
      curriculum / game scenarios the sign (positive/negative) is what
      matters most for credit assignment.

    The original (unclipped) reward is preserved in info["raw_reward"].
    """

    def __init__(self, env: gymnasium.Env, low: float = -1.0, high: float = 1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, self.low, self.high))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["raw_reward"] = reward          # preserve original for logging
        clipped = self.reward(reward)
        return obs, clipped, terminated, truncated, info


# ---------------------------------------------------------------------------
# 6. Episodic Life
# ---------------------------------------------------------------------------

class EpisodicLifeWrapper(gymnasium.Wrapper):
    """
    Treat each life loss as a terminal signal for the agent, while only
    actually resetting the game when all lives are gone.

    Why episodic lives?
    - In games with multiple lives the agent may learn to play recklessly
      after losing one life, since the episode doesn't end.
    - Marking `terminated=True` on life loss tells the value function that
      there is no future reward to bootstrap, encouraging safer play.
    - The *environment* reset only happens at true game-over.

    Assumption: the environment's `info` dict contains a "lives" key
    (added by PixelGameEnv). Override `_lives_from_info` if your env
    stores life count differently.
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self._lives: int = 0
        self._real_done: bool = True  # True → next reset is a true reset

    # ------------------------------------------------------------------
    # Override in subclasses if needed
    # ------------------------------------------------------------------

    def _lives_from_info(self, info: dict) -> int:
        """Extract remaining lives from step info."""
        game_state = info.get("game_state", {})
        return int(game_state.get("lives", 0))

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        if self._real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Life was lost but game isn't over: take a no-op step to get
            # a fresh observation without resetting game state.
            obs, _reward, _terminated, _truncated, info = self.env.step(0)
        self._lives = self._lives_from_info(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_lives = self._lives_from_info(info)

        life_lost = (current_lives < self._lives) and not terminated
        self._lives = current_lives
        self._real_done = terminated  # track whether game is truly over

        # Signal terminal to the agent on life loss even though the game
        # will continue (real reset happens only at true game-over).
        agent_terminated = terminated or life_lost

        return obs, reward, agent_terminated, truncated, info


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_env(
    env_factory,
    frame_size: Tuple[int, int] = (84, 84),
    num_stack: int = 4,
    reward_clip: Optional[Tuple[float, float]] = (-1.0, 1.0),
    episodic_life: bool = True,
) -> gymnasium.Env:
    """
    Compose a fully wrapped environment from a raw env factory callable.

    Args:
        env_factory:   Callable[[], gymnasium.Env] — creates the base env.
        frame_size:    (height, width) to resize frames to.
        num_stack:     Number of consecutive frames to stack.
        reward_clip:   (low, high) clipping bounds, or None to skip.
        episodic_life: Whether to apply EpisodicLifeWrapper.

    Returns:
        A fully wrapped Gymnasium environment ready for PPO.

    Example::

        from envs.pixel_env import PixelGameEnv
        from envs.wrappers import make_env

        env = make_env(lambda: PixelGameEnv())
        obs, info = env.reset()
        # obs.shape → (84, 84, 4)  float32  [0, 1]
    """
    env = env_factory()

    # Frame preprocessing
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, size=frame_size)
    env = NormalizeObsWrapper(env)

    # Temporal context
    env = FrameStackWrapper(env, num_stack=num_stack)

    # Reward shaping
    if reward_clip is not None:
        env = RewardClipWrapper(env, low=reward_clip[0], high=reward_clip[1])

    # Life management
    if episodic_life:
        env = EpisodicLifeWrapper(env)

    return env
