"""
StateGameEnv — same game as PixelGameEnv but returns a flat 18-dim state
vector instead of pixels.  No rendering, no pygame overhead.

Observation vector (18 values, all normalized to roughly [0, 1]):
    [0]     player_x / W
    [1]     player_y / H
    [2..4]  target 0: (dx/W, dy/H, dist/max_dist)
    [5..7]  target 1: ...
    [8..10] target 2: ...
    [11..13] target 3: ...
    [14..16] target 4: ...
    [17]    nearest obstacle dist / max_dist

Absent targets (when fewer than 5 remain) are padded with zeros.
"""

from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces

from envs.pixel_env import PixelGameEnv

OBS_DIM = 18


class StateGameEnv(PixelGameEnv):
    """
    Drop-in replacement for PixelGameEnv that observes game state directly.

    Skips all rendering so training is faster (no SDL calls).
    Everything else — reward, actions, done conditions — is identical.
    """

    def __init__(self, max_steps: int = 1000):
        super().__init__(frame_size=(84, 84), render_mode=None, max_steps=max_steps)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Override reset / step to skip pygame entirely
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        gymnasium.Env.reset(self, seed=seed)   # seeds numpy RNG, skips pygame
        self._init_game()
        self.current_step = 0
        self._hit_wall = False
        return self._extract_state(), {"game_state": self.game_state}

    def step(self, action):
        self._apply_action(action)
        self.current_step += 1
        reward     = self._compute_reward()
        terminated = self.game_state["lives"] <= 0
        truncated  = self.current_step >= self.max_steps
        return self._extract_state(), reward, terminated, truncated, {"game_state": self.game_state}

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def _extract_state(self) -> np.ndarray:
        state    = self.game_state
        W, H     = self.screen_width, self.screen_height
        max_dist = np.sqrt(W * W + H * H)
        px, py   = float(state["player_x"]), float(state["player_y"])

        obs = [px / W, py / H]

        targets = state["targets"]
        for i in range(5):
            if i < len(targets):
                t    = targets[i]
                dx   = (t["x"] - px) / W
                dy   = (t["y"] - py) / H
                dist = np.sqrt((t["x"] - px) ** 2 + (t["y"] - py) ** 2) / max_dist
            else:
                dx = dy = dist = 0.0
            obs.extend([dx, dy, dist])

        # Compute nearest obstacle distance directly (don't rely on prev_obstacle_dist)
        nearest = max_dist
        for ob in state["obstacles"]:
            cx = np.clip(px, ob["x"], ob["x"] + ob["w"])
            cy = np.clip(py, ob["y"], ob["y"] + ob["h"])
            nearest = min(nearest, np.sqrt((px - cx) ** 2 + (py - cy) ** 2))
        obs.append(nearest / max_dist)

        return np.array(obs, dtype=np.float32)
