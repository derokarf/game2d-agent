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
    ):
        super().__init__()

        self.frame_size = frame_size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

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

    def _init_pygame(self):
        if pygame.get_init() is False:
            pygame.init()
        if self.screen is None:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

    def _init_game(self):
        """Initialize your game state here."""
        self.game_state = {
            "player_x": self.screen_width // 2,
            "player_y": self.screen_height // 2,
            "player_speed": 5,
            "targets": self._spawn_targets(5),
            "obstacles": self._spawn_obstacles(3),
            "score": 0,
            "lives": 3,
        }

    def _spawn_targets(self, count):
        return [
            {
                "x": np.random.randint(20, self.screen_width - 20),
                "y": np.random.randint(20, self.screen_height - 20),
                "radius": 10,
            }
            for _ in range(count)
        ]

    def _spawn_obstacles(self, count):
        return [
            {
                "x": np.random.randint(50, self.screen_width - 50),
                "y": np.random.randint(50, self.screen_height - 50),
                "w": np.random.randint(30, 80),
                "h": np.random.randint(30, 80),
            }
            for _ in range(count)
        ]

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

        state["player_x"] = np.clip(state["player_x"], 0, self.screen_width)
        state["player_y"] = np.clip(state["player_y"], 0, self.screen_height)

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

        pygame.draw.circle(self.screen, (0, 255, 0), (state["player_x"], state["player_y"]), 8)

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {state['score']}", True, (255, 255, 255))
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
        player_pos = (state["player_x"], state["player_y"])

        collected = []
        for i, tgt in enumerate(state["targets"]):
            dist = np.sqrt((player_pos[0] - tgt["x"]) ** 2 + (player_pos[1] - tgt["y"]) ** 2)
            if dist < tgt["radius"] + 8:
                collected.append(i)
                reward += 10.0

        for idx in sorted(collected, reverse=True):
            state["targets"].pop(idx)

        if len(state["targets"]) == 0:
            state["targets"] = self._spawn_targets(5)
            state["score"] += 1
            reward += 50.0

        for obs in state["obstacles"]:
            if (
                obs["x"] < player_pos[0] < obs["x"] + obs["w"]
                and obs["y"] < player_pos[1] < obs["y"] + obs["h"]
            ):
                reward -= 5.0
                state["lives"] -= 1
                state["player_x"] = self.screen_width // 2
                state["player_y"] = self.screen_height // 2
                break

        return reward

    def _get_obs(self):
        frame = self._render_frame()
        return self._process_frame(frame)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_pygame()
        self._init_game()
        self.current_step = 0

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
