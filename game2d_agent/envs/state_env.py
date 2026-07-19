"""
StateGameEnv — same game as PixelGameEnv but returns a flat state vector
instead of pixels.  No rendering, no pygame overhead.

Observation vector (49 values, all normalized to roughly [0, 1]):
    [0]      player_x / W
    [1]      player_y / H
    [2..16]  5 targets × (dx/W, dy/H, dist/max_dist)   # 15 values
    [17..48] 32 ray clearances (normalized distance to first barrier)

Targets are still encoded egocentrically (direction + distance); absent
targets (fewer than 5 remain) are zero-padded.

Obstacles are perceived by RAY-CASTING instead of per-object slots.  We cast
N_RAYS evenly-spaced rays outward from the player; each value is the distance
to the first barrier hit along that ray — an obstacle edge OR the screen wall
— normalized by the screen diagonal.  This gives the agent a direct
"how far can I go in each direction" reading, is independent of obstacle
count/ordering, and includes the screen walls for free.
"""

from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces

from envs.pixel_env import PixelGameEnv

N_RAYS  = 32
OBS_DIM = 2 + 5 * 3 + N_RAYS   # player(2) + targets(15) + rays(32) = 49


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

        # Precompute ray directions (constant across steps).
        angles = np.linspace(0.0, 2.0 * np.pi, N_RAYS, endpoint=False)
        self._ray_dx = np.cos(angles)
        self._ray_dy = np.sin(angles)

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
    # Ray casting
    # ------------------------------------------------------------------

    @staticmethod
    def _slab(o: float, d: np.ndarray, lo: float, hi: float):
        """Per-axis ray/slab entry & exit distances (vectorized over rays).

        Handles rays parallel to the axis: if the origin is outside [lo, hi]
        the ray never intersects (tmin=+inf, tmax=-inf); if inside, the axis
        does not constrain the intersection (tmin=-inf, tmax=+inf).
        """
        eps = 1e-9
        parallel = np.abs(d) < eps
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - o) / d
            t2 = (hi - o) / d
        tmin = np.minimum(t1, t2)
        tmax = np.maximum(t1, t2)
        inside = (o >= lo) and (o <= hi)
        tmin = np.where(parallel, (-np.inf if inside else np.inf), tmin)
        tmax = np.where(parallel, ( np.inf if inside else -np.inf), tmax)
        return tmin, tmax

    def _cast_rays(self, px: float, py: float) -> np.ndarray:
        """Distance to the first barrier (obstacle edge or screen wall) per ray."""
        dx, dy = self._ray_dx, self._ray_dy
        W, H   = self.screen_width, self.screen_height

        # Screen-wall exit distance (player is always inside the screen box).
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 0, (W - px) / dx, np.where(dx < 0, -px / dx, np.inf))
            ty = np.where(dy > 0, (H - py) / dy, np.where(dy < 0, -py / dy, np.inf))
        dist = np.minimum(tx, ty)

        # Nearest obstacle entry along each ray.
        for ob in self.game_state["obstacles"]:
            xmin, ymin = ob["x"], ob["y"]
            xmax, ymax = ob["x"] + ob["w"], ob["y"] + ob["h"]
            tminx, tmaxx = self._slab(px, dx, xmin, xmax)
            tminy, tmaxy = self._slab(py, dy, ymin, ymax)
            tmin = np.maximum(tminx, tminy)
            tmax = np.minimum(tmaxx, tmaxy)
            hit  = (tmax >= np.maximum(tmin, 0.0)) & (tmin <= tmax) & (tmin >= 0.0)
            t_hit = np.where(hit, tmin, np.inf)
            dist = np.minimum(dist, t_hit)

        max_dist = np.sqrt(W * W + H * H)
        return (dist / max_dist).astype(np.float32)

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def _extract_state(self) -> np.ndarray:
        state    = self.game_state
        W, H     = self.screen_width, self.screen_height
        max_dist = np.sqrt(W * W + H * H)
        px, py   = float(state["player_x"]), float(state["player_y"])

        obs = [px / W, py / H]

        # Targets: direction + distance (egocentric, zero-padded).
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

        # Obstacles + screen walls: ray-cast clearances.
        rays = self._cast_rays(px, py)
        obs.extend(rays.tolist())

        return np.array(obs, dtype=np.float32)
