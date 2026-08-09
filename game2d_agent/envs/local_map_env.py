"""
LocalMapGameEnv — same game as PixelGameEnv, but the observation is an
egocentric local occupancy map plus a global goal vector.  Designed as a
transferable perception contract (see docs/agent_v2_local_map_spec.md):

    obs = {
        "map":    float32 (2, 17, 17)   # egocentric, 12px cells, ±102px window
                    channel 0 = blocked (obstacle OR outside screen)
                    channel 1 = target present
        "global": float32 (3,)          # (bearing_x, bearing_y, dist) to nearest target
    }

Everything is relative/egocentric — no absolute coordinates, no fixed entity
slots — so the same contract survives a different board size, obstacle count,
or objective.  The geodesic BFS stays only in the reward (training teacher);
it is never part of the observation.
"""

from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces

from envs.pixel_env import PixelGameEnv

MAP_CELL   = 12          # px per cell
MAP_SIZE   = 17          # cells per side (odd -> agent at the center cell)
GLOBAL_DIM = 3


class LocalMapGameEnv(PixelGameEnv):
    """Drop-in game with an egocentric occupancy-map + goal-vector observation."""

    def __init__(self, max_steps: int = 1000, n_obstacles: int = 3, n_targets: int = 5,
                 render_mode: str | None = None, random_counts: bool = False,
                 min_obstacles: int = 1, min_targets: int = 1):
        super().__init__(frame_size=(84, 84), render_mode=render_mode, max_steps=max_steps,
                         n_obstacles=n_obstacles, n_targets=n_targets,
                         random_counts=random_counts, min_obstacles=min_obstacles,
                         min_targets=min_targets)
        self.observation_space = spaces.Dict({
            "map":    spaces.Box(0.0, 1.0, shape=(2, MAP_SIZE, MAP_SIZE), dtype=np.float32),
            "global": spaces.Box(-2.0, 2.0, shape=(GLOBAL_DIM,), dtype=np.float32),
        })

    # ------------------------------------------------------------------
    # Skip pygame for training (obs is computed from game state directly)
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        gymnasium.Env.reset(self, seed=seed)
        self._init_game()
        self.current_step = 0
        self._hit_wall = False
        return self._extract_obs(), {"game_state": self.game_state}

    def step(self, action):
        self._apply_action(action)
        self.current_step += 1
        reward     = self._compute_reward()
        terminated = self.game_state["lives"] <= 0
        truncated  = self.current_step >= self.max_steps
        return self._extract_obs(), reward, terminated, truncated, {"game_state": self.game_state}

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _extract_obs(self):
        state = self.game_state
        W, H  = self.screen_width, self.screen_height
        px, py = float(state["player_x"]), float(state["player_y"])
        cs, R, half = MAP_CELL, MAP_SIZE, MAP_SIZE // 2

        # Cell-center world coordinates for the egocentric window.
        xs = px + (np.arange(R) - half) * cs      # (R,) column world-x
        ys = py + (np.arange(R) - half) * cs      # (R,) row world-y
        CX = np.broadcast_to(xs, (R, R))
        CY = np.broadcast_to(ys[:, None], (R, R))

        # Channel 0 — blocked: outside the screen, or inside any obstacle.
        blocked = (CX < 0) | (CX > W) | (CY < 0) | (CY > H)
        for o in state["obstacles"]:
            blocked |= ((CX >= o["x"]) & (CX <= o["x"] + o["w"])
                        & (CY >= o["y"]) & (CY <= o["y"] + o["h"]))

        # Channel 1 — target present (nearest cell to each target within the window).
        target_ch = np.zeros((R, R), dtype=np.float32)
        for t in state["targets"]:
            j = int(round((t["x"] - px) / cs)) + half
            i = int(round((t["y"] - py) / cs)) + half
            if 0 <= i < R and 0 <= j < R:
                target_ch[i, j] = 1.0

        map_obs = np.stack([blocked.astype(np.float32), target_ch], axis=0)  # (2, R, R)

        # Global goal vector: straight-line bearing + distance to nearest target.
        if state["targets"]:
            best = min(state["targets"],
                       key=lambda t: (t["x"] - px) ** 2 + (t["y"] - py) ** 2)
            dx, dy = best["x"] - px, best["y"] - py
            d = np.sqrt(dx * dx + dy * dy)
            if d > 1e-6:
                bx, by = dx / d, dy / d
            else:
                bx, by = 0.0, 0.0
            gdist = d / np.sqrt(W * W + H * H)
        else:
            bx = by = gdist = 0.0
        global_obs = np.array([bx, by, gdist], dtype=np.float32)

        return {"map": map_obs, "global": global_obs}
