import gymnasium
import numpy as np
import cv2
import pygame
from collections import deque
from gymnasium import spaces
from typing import Optional


class PixelGameEnv(gymnasium.Env):
    """
    Custom Gymnasium environment for playing a 2D game from pixels only.
    
    Replace the game-specific rendering and logic in this class with your actual game.
    The environment:
    - Captures frames as RGB numpy arrays
    - Defines discrete action space
    - Computes reward based on game state
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        frame_size: tuple = (84, 84),
        render_mode: Optional[str] = None,
        max_steps: int = 5000,
        n_obstacles: int = 3,
        n_targets: int = 5,
        random_counts: bool = False,
        min_obstacles: int = 1,
        min_targets: int = 1,
    ):
        super().__init__()

        self.frame_size = frame_size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        # n_obstacles / n_targets are the caps; with random_counts each level
        # samples uniformly in [min_*, cap].  Targets cap must stay <= 5 (obs
        # has 5 target slots).  Obstacle count is free (ray-based perception).
        self.n_obstacles   = n_obstacles
        self.n_targets     = n_targets
        self.random_counts = random_counts
        self.min_obstacles = min_obstacles
        self.min_targets   = min_targets

        self.screen_width = 640
        self.screen_height = 480
        self._player_radius = 20
        self.screen = None
        self.clock = None

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(frame_size[1], frame_size[0], 3),
            dtype=np.uint8,
        )

        self.game_state = None
        self._hit_wall = False

        # Geodesic (obstacle-aware) distance field for reward shaping, cached
        # and recomputed only when the target/obstacle set changes.
        self._dist_cell_size = 10
        self._dist_field = None
        self._dist_sig   = None

    def _init_pygame(self):
        if pygame.get_init() is False:
            pygame.init()
        if self.screen is None:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self._font = pygame.font.Font(None, 36)   # create once, reuse every step

    def _sample_counts(self):
        """(n_obstacles, n_targets) for a level — random per level if enabled."""
        if not self.random_counts:
            return self.n_obstacles, self.n_targets
        n_obs = int(np.random.randint(self.min_obstacles, self.n_obstacles + 1))
        n_tgt = int(np.random.randint(self.min_targets,   self.n_targets   + 1))
        return n_obs, n_tgt

    def _init_game(self):
        """Initialize your game state here."""
        n_obs, n_tgt = self._sample_counts()
        self.game_state = {
            "player_x": self.screen_width // 2,
            "player_y": self.screen_height // 2,
            "player_speed": 8,
            "targets": [],
            "obstacles": self._spawn_obstacles(n_obs),
            "score": 0,
            "lives": 3,
            "prev_target_dist": None,
        }
        self.game_state["targets"] = self._spawn_targets(n_tgt)
        x, y = self._safe_spawn()
        self.game_state["player_x"] = x
        self.game_state["player_y"] = y

    def _spawn_targets(self, count):
        targets = []
        for _ in range(count):
            for _ in range(100):
                x = np.random.randint(30, self.screen_width - 30)
                y = np.random.randint(30, self.screen_height - 30)
                safe = True
                for obs in self.game_state["obstacles"]:
                    cx = np.clip(x, obs["x"], obs["x"] + obs["w"])
                    cy = np.clip(y, obs["y"], obs["y"] + obs["h"])
                    if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < 35:
                        safe = False
                        break
                if safe:
                    targets.append({"x": x, "y": y, "radius": 25})
                    break
            else:
                targets.append({"x": x, "y": y, "radius": 25})
        return targets

    def _spawn_obstacles(self, count, avoid=None):
        # Every gap between obstacles must be at least the agent's passable
        # width. The agent is blocked within player_radius of an edge, so a
        # corridor needs > 2*player_radius of clearance to traverse. A thinner
        # slot is impassable, yet the (infinitely thin) rays and line-of-sight
        # check would still "see" through it — luring the agent into a trap.
        min_gap = 2 * self._player_radius + 12  # comfortably passable corridor

        def edge_gap(a, b):
            dx = max(b["x"] - (a["x"] + a["w"]), a["x"] - (b["x"] + b["w"]), 0.0)
            dy = max(b["y"] - (a["y"] + a["h"]), a["y"] - (b["y"] + b["h"]), 0.0)
            return np.sqrt(dx * dx + dy * dy)

        def hits_player(a):
            # Reject if the obstacle would spawn on/near the player (used when
            # obstacles respawn mid-episode so the agent can't be trapped).
            if avoid is None:
                return False
            ax, ay = avoid
            cx = np.clip(ax, a["x"], a["x"] + a["w"])
            cy = np.clip(ay, a["y"], a["y"] + a["h"])
            return np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2) < self._player_radius + 30

        obstacles = []
        for _ in range(count * 100):
            w = np.random.randint(60, 120)
            h = np.random.randint(60, 120)
            x = np.random.randint(50, self.screen_width  - 50 - w)
            y = np.random.randint(50, self.screen_height - 50 - h)
            cand = {"x": x, "y": y, "w": w, "h": h}
            # Reject if it would form a too-narrow (impassable) gap with any
            # existing obstacle (gap == 0 means overlapping, also rejected),
            # or if it would spawn on the player.
            if not hits_player(cand) and all(edge_gap(cand, o) >= min_gap for o in obstacles):
                obstacles.append(cand)
            if len(obstacles) == count:
                break
        return obstacles

    def _safe_spawn(self):
        """Return a (x, y) position that doesn't overlap any obstacle."""
        for _ in range(100):
            x = np.random.randint(40, self.screen_width - 40)
            y = np.random.randint(40, self.screen_height - 40)
            safe = True
            for obs in self.game_state["obstacles"]:
                cx = np.clip(x, obs["x"], obs["x"] + obs["w"])
                cy = np.clip(y, obs["y"], obs["y"] + obs["h"])
                if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < 30:
                    safe = False
                    break
            if safe:
                return x, y
        return self.screen_width // 2, self.screen_height // 2

    def _blocked(self, x, y):
        """True if point (x, y) is within the player radius of any obstacle edge."""
        for obs in self.game_state["obstacles"]:
            cx = np.clip(x, obs["x"], obs["x"] + obs["w"])
            cy = np.clip(y, obs["y"], obs["y"] + obs["h"])
            if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < self._player_radius:
                return True
        return False

    def _segment_blocked(self, x0, y0, x1, y1):
        """True if the segment (x0,y0)->(x1,y1) passes through any obstacle.

        Liang-Barsky clip of the parametric segment against each obstacle's
        axis-aligned box; if any box clips a sub-interval of t in [0,1], the
        line of sight is obstructed.
        """
        dx, dy = x1 - x0, y1 - y0
        for obs in self.game_state["obstacles"]:
            xmin, ymin = obs["x"], obs["y"]
            xmax, ymax = obs["x"] + obs["w"], obs["y"] + obs["h"]
            t0, t1 = 0.0, 1.0
            hit = True
            for p, q in ((-dx, x0 - xmin), (dx, xmax - x0),
                         (-dy, y0 - ymin), (dy, ymax - y0)):
                if p == 0:
                    if q < 0:           # parallel and outside this slab
                        hit = False
                        break
                else:
                    r = q / p
                    if p < 0:
                        if r > t1:
                            hit = False
                            break
                        if r > t0:
                            t0 = r
                    else:
                        if r < t0:
                            hit = False
                            break
                        if r < t1:
                            t1 = r
            if hit and t0 <= t1:
                return True
        return False

    def _apply_action(self, action):
        state = self.game_state
        prev_x, prev_y = state["player_x"], state["player_y"]
        new_x, new_y = prev_x, prev_y
        if action == 1:
            new_x -= state["player_speed"]
        elif action == 2:
            new_x += state["player_speed"]
        elif action == 3:
            new_y -= state["player_speed"]
        elif action == 4:
            new_y += state["player_speed"]

        clipped_x = np.clip(new_x, 0, self.screen_width)
        clipped_y = np.clip(new_y, 0, self.screen_height)
        self._hit_wall = (clipped_x != new_x) or (clipped_y != new_y)

        # Obstacles are solid: block the move (per-axis) if it would enter one.
        # Per-axis lets the player slide along a wall instead of sticking.
        if not self._blocked(clipped_x, prev_y):
            state["player_x"] = clipped_x
        if not self._blocked(state["player_x"], clipped_y):
            state["player_y"] = clipped_y

    def _render_frame(self):
        self._init_pygame()
        self.screen.fill((30, 30, 30))
        state = self.game_state

        for obs in state["obstacles"]:
            pygame.draw.rect(
                self.screen, (100, 100, 100), (obs["x"], obs["y"], obs["w"], obs["h"])
            )

        for tgt in state["targets"]:
            pygame.draw.circle(self.screen, (255, 200, 0), (tgt["x"], tgt["y"]), tgt["radius"])

        pygame.draw.circle(self.screen, (0, 255, 0), (state["player_x"], state["player_y"]), 20)

        score_text = self._font.render(f"Score: {state['score']}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        return frame

    def _process_frame(self, frame):
        resized = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    # ------------------------------------------------------------------
    # Geodesic (obstacle-aware) distance field for reward shaping
    # ------------------------------------------------------------------

    def _ensure_distance_field(self):
        """(Re)compute the cached distance field when targets/obstacles change."""
        state = self.game_state
        sig = (tuple((t["x"], t["y"]) for t in state["targets"]),
               tuple((o["x"], o["y"], o["w"], o["h"]) for o in state["obstacles"]))
        if sig != self._dist_sig:
            self._dist_sig   = sig
            self._dist_field = self._compute_distance_field(state["targets"],
                                                            state["obstacles"])

    def _compute_distance_field(self, targets, obstacles):
        """Multi-source BFS distance (in cells) from all targets, around walls.

        4-connected to match the agent's 4-directional movement, so the value
        is the true number of steps to the nearest reachable target.  Obstacles
        are inflated by the player radius (configuration space) so paths keep
        the agent's body clear of walls.
        """
        cs = self._dist_cell_size
        gw = self.screen_width  // cs
        gh = self.screen_height // cs
        pr = self._player_radius

        # Blocked cells: cell center within player_radius of any obstacle edge.
        cols = np.arange(gw) * cs + cs / 2.0
        rows = np.arange(gh) * cs + cs / 2.0
        cx = np.broadcast_to(cols, (gh, gw))
        cy = np.broadcast_to(rows[:, None], (gh, gw))
        blocked = np.zeros((gh, gw), dtype=bool)
        for o in obstacles:
            qx = np.clip(cx, o["x"], o["x"] + o["w"])
            qy = np.clip(cy, o["y"], o["y"] + o["h"])
            blocked |= ((cx - qx) ** 2 + (cy - qy) ** 2) < pr * pr

        INF = np.iinfo(np.int32).max
        dist = np.full((gh, gw), INF, dtype=np.int32)
        q = deque()
        for t in targets:
            c = min(int(t["x"] // cs), gw - 1)
            r = min(int(t["y"] // cs), gh - 1)
            if dist[r, c] != 0:
                dist[r, c] = 0          # seed even if "blocked": target is reachable
                q.append((r, c))
        while q:
            r, c = q.popleft()
            nd = dist[r, c] + 1
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < gh and 0 <= nc < gw and not blocked[nr, nc] and dist[nr, nc] > nd:
                    dist[nr, nc] = nd
                    q.append((nr, nc))

        # Fill blocked/unreachable cells so bilinear interpolation stays smooth
        # right up to a wall: each such cell inherits (min free neighbor + 1),
        # propagated a couple of layers; anything left gets a large finite value.
        field = np.where(dist == INF, np.inf, dist).astype(np.float64)
        maxd = float(dist[dist != INF].max()) if np.any(dist != INF) else 0.0
        for _ in range(3):
            up    = np.full_like(field, np.inf); up[1:, :]    = field[:-1, :]
            down  = np.full_like(field, np.inf); down[:-1, :] = field[1:, :]
            left  = np.full_like(field, np.inf); left[:, 1:]  = field[:, :-1]
            right = np.full_like(field, np.inf); right[:, :-1] = field[:, 1:]
            neigh = np.minimum.reduce([up, down, left, right]) + 1.0
            fill  = np.isinf(field) & ~np.isinf(neigh)
            field = np.where(fill, neigh, field)
        field[np.isinf(field)] = maxd + 2.0
        return field

    def _geodesic_dist(self, px, py):
        """Distance (in pixels) to the nearest target, around walls.

        Bilinear interpolation of the (filled) cell field so the value is a
        smooth continuous function of position — every step changes it, and
        exact ties become measure-zero, so the shaping gradient never goes flat.
        """
        self._ensure_distance_field()
        field = self._dist_field
        if field is None:
            return 0.0
        cs = self._dist_cell_size
        gh, gw = field.shape
        # Continuous cell coordinates (cell centers sit at (col+0.5, row+0.5)).
        fx = px / cs - 0.5
        fy = py / cs - 0.5
        x0 = min(max(int(np.floor(fx)), 0), gw - 2)
        y0 = min(max(int(np.floor(fy)), 0), gh - 2)
        tx = min(max(fx - x0, 0.0), 1.0)
        ty = min(max(fy - y0, 0.0), 1.0)
        d = (field[y0,     x0]     * (1 - tx) * (1 - ty)
             + field[y0,     x0 + 1] * tx       * (1 - ty)
             + field[y0 + 1, x0]     * (1 - tx) * ty
             + field[y0 + 1, x0 + 1] * tx       * ty)
        return float(d) * cs

    def _compute_reward(self):
        reward = -0.01
        state = self.game_state
        px, py = state["player_x"], state["player_y"]

        # ── Target collection ─────────────────────────────────────────────────
        collected = []
        for i, tgt in enumerate(state["targets"]):
            dist = np.sqrt((px - tgt["x"]) ** 2 + (py - tgt["y"]) ** 2)
            if dist < tgt["radius"] + 20:
                collected.append(i)
                reward += 10.0

        for idx in sorted(collected, reverse=True):
            state["targets"].pop(idx)

        if len(state["targets"]) == 0:
            # New level: re-roll counts and layout. Respawn obstacles first
            # (avoiding the player so a wall can't spawn on top of it), then
            # targets (which already avoid obstacles).
            n_obs, n_tgt = self._sample_counts()
            state["obstacles"] = self._spawn_obstacles(n_obs, avoid=(px, py))
            state["targets"]   = self._spawn_targets(n_tgt)
            state["score"] += 1
            reward += 50.0

        # Obstacles are solid walls (movement is blocked in _apply_action), so
        # there is no collision penalty — the agent physically cannot enter one
        # and must route around. Perception of obstacles is still in the obs.

        # ── Geodesic approach shaping ─────────────────────────────────────────
        # Reward descending an obstacle-aware distance field (multi-source BFS
        # from all targets) toward the nearest *reachable* target.  Unlike
        # straight-line distance, "closer" here means real progress along a path
        # that routes around walls: pressing into a wall doesn't lower it (no
        # jamming), and every free cell has a downhill direction (no gradient-
        # less milling when the target is occluded).
        if not state["targets"]:
            state["prev_target_dist"] = None
        else:
            geo = self._geodesic_dist(px, py)
            # Skip the delta on collection steps — the target set (and thus the
            # field) just changed, so only reset the baseline.
            if state["prev_target_dist"] is not None and not collected:
                reward += 0.05 * (state["prev_target_dist"] - geo)
            state["prev_target_dist"] = geo

        return reward

    def _get_obs(self):
        frame = self._render_frame()
        return self._process_frame(frame)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_pygame()
        self._init_game()
        self.current_step = 0
        self._hit_wall = False

        obs = self._get_obs()
        info = {"game_state": self.game_state}

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, action):
        self._apply_action(action)
        self.current_step += 1

        reward = self._compute_reward()
        obs = self._get_obs()
        info = {"game_state": self.game_state}

        terminated = self.game_state["lives"] <= 0
        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self._render_frame()
            if self.clock is None:
                self.clock = pygame.time.Clock()
            self.clock.tick(self.metadata["render_fps"])
            pygame.event.pump()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            frame = self._render_frame()
            return self._process_frame(frame)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
