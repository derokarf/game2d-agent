import gymnasium
import numpy as np
import cv2
import pygame
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
    ):
        super().__init__()

        self.frame_size = frame_size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.n_obstacles = n_obstacles
        self.n_targets = n_targets

        self.screen_width = 640
        self.screen_height = 480
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

    def _init_pygame(self):
        if pygame.get_init() is False:
            pygame.init()
        if self.screen is None:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self._font = pygame.font.Font(None, 36)   # create once, reuse every step

    def _init_game(self):
        """Initialize your game state here."""
        self.game_state = {
            "player_x": self.screen_width // 2,
            "player_y": self.screen_height // 2,
            "player_speed": 8,
            "targets": [],
            "obstacles": self._spawn_obstacles(self.n_obstacles),
            "score": 0,
            "lives": 3,
            "prev_target_dist": None,
        }
        self.game_state["targets"] = self._spawn_targets(self.n_targets)
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

    def _spawn_obstacles(self, count):
        return [
            {
                "x": np.random.randint(50, self.screen_width - 50),
                "y": np.random.randint(50, self.screen_height - 50),
                "w": np.random.randint(60, 120),
                "h": np.random.randint(60, 120),
            }
            for _ in range(count)
        ]

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

    def _apply_action(self, action):
        state = self.game_state
        if action == 0:
            pass
        elif action == 1:
            state["player_x"] -= state["player_speed"]
        elif action == 2:
            state["player_x"] += state["player_speed"]
        elif action == 3:
            state["player_y"] -= state["player_speed"]
        elif action == 4:
            state["player_y"] += state["player_speed"]

        clipped_x = np.clip(state["player_x"], 0, self.screen_width)
        clipped_y = np.clip(state["player_y"], 0, self.screen_height)
        self._hit_wall = (clipped_x != state["player_x"]) or (clipped_y != state["player_y"])
        state["player_x"] = clipped_x
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

    def _compute_reward(self):
        reward = -0.01
        state = self.game_state
        px, py = state["player_x"], state["player_y"]

        # ── Target collection ─────────────────────────────────────────────────
        collected = []
        nearest_target_dist = float("inf")
        for i, tgt in enumerate(state["targets"]):
            dist = np.sqrt((px - tgt["x"]) ** 2 + (py - tgt["y"]) ** 2)
            nearest_target_dist = min(nearest_target_dist, dist)
            if dist < tgt["radius"] + 20:
                collected.append(i)
                reward += 10.0

        for idx in sorted(collected, reverse=True):
            state["targets"].pop(idx)

        if len(state["targets"]) == 0:
            state["targets"] = self._spawn_targets(self.n_targets)
            state["score"] += 1
            reward += 50.0
            nearest_target_dist = float("inf")  # recalc after respawn

        # ── Obstacle proximity ────────────────────────────────────────────────
        nearest_obstacle_dist = float("inf")
        hit_obstacle = False
        for obs in state["obstacles"]:
            # Distance to nearest edge of the obstacle rectangle
            cx = np.clip(px, obs["x"], obs["x"] + obs["w"])
            cy = np.clip(py, obs["y"], obs["y"] + obs["h"])
            dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            nearest_obstacle_dist = min(nearest_obstacle_dist, dist)

            if dist == 0:  # player is inside the obstacle
                if not hit_obstacle:
                    reward -= 20.0
                    state["lives"] -= 1
                    state["player_x"], state["player_y"] = self._safe_spawn()
                    hit_obstacle = True

        # ── Shaping ───────────────────────────────────────────────────────────
        # Approach reward k=0.01. Reset prev on collection so the jump to the
        # next target doesn't produce a negative penalty (which taught hovering).
        if collected:
            state["prev_target_dist"] = None
        elif nearest_target_dist != float("inf"):
            if state["prev_target_dist"] is not None:
                reward += 0.01 * (state["prev_target_dist"] - nearest_target_dist)
            state["prev_target_dist"] = nearest_target_dist
        else:
            state["prev_target_dist"] = None

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
