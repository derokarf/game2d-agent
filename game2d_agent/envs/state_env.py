"""
StateGameEnv — same game as PixelGameEnv but returns a flat 26-dim state
vector instead of pixels.  No rendering, no pygame overhead.

Observation vector (26 values, all normalized to roughly [0, 1]):
    [0]     player_x / W
    [1]     player_y / H
    [2..4]  target 0: (dx/W, dy/H, dist/max_dist)
    [5..7]  target 1: ...
    [8..10] target 2: ...
    [11..13] target 3: ...
    [14..16] target 4: ...
    [17..19] obstacle 0: (dx/W, dy/H, edge_dist/max_dist)
    [20..22] obstacle 1: ...
    [23..25] obstacle 2: ...

dx/dy point from player to obstacle center.
edge_dist is distance to the nearest point on the obstacle rectangle edge.
Absent targets (when fewer than 5 remain) are padded with zeros.
"""

from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces

from envs.pixel_env import PixelGameEnv

OBS_DIM = 26


class StateGameEnv(PixelGameEnv):
    """
    Drop-in replacement for PixelGameEnv that observes game state directly.

    Skips all rendering so training is faster (no SDL calls).
    Everything else — reward, actions, done conditions — is identical.
    """

    def __init__(self, max_steps: int = 1000, n_obstacles: int = 3, n_targets: int = 5,
                 render_mode: str | None = None):
        super().__init__(frame_size=(84, 84), render_mode=render_mode, max_steps=max_steps,
                         n_obstacles=n_obstacles, n_targets=n_targets)
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

        # Targets: direction + distance (same as before)
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

        # Obstacles: direction to center + distance to nearest edge (always 3 slots)
        obstacles = state["obstacles"]
        for i in range(3):
            if i < len(obstacles):
                ob = obstacles[i]
                cx_center = ob["x"] + ob["w"] / 2.0
                cy_center = ob["y"] + ob["h"] / 2.0
                dx   = (cx_center - px) / W
                dy   = (cy_center - py) / H
                cx_edge = np.clip(px, ob["x"], ob["x"] + ob["w"])
                cy_edge = np.clip(py, ob["y"], ob["y"] + ob["h"])
                dist = np.sqrt((px - cx_edge) ** 2 + (py - cy_edge) ** 2) / max_dist
            else:
                dx = dy = dist = 0.0
            obs.extend([dx, dy, dist])

        return np.array(obs, dtype=np.float32)
