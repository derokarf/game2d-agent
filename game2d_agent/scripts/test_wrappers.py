"""
Validation script for environment wrappers.

Run from the game2d_agent/ directory:
    python scripts/test_wrappers.py

Each section tests one wrapper in isolation, then the full make_env() stack.
No training happens — just observation/reward shape and dtype checks.
"""

from __future__ import annotations

import os
import sys

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs.pixel_env import PixelGameEnv
from envs.wrappers import (
    EpisodicLifeWrapper,
    FrameStackWrapper,
    GrayscaleWrapper,
    NormalizeObsWrapper,
    ResizeWrapper,
    RewardClipWrapper,
    make_env,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    msg = f"  {status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not condition:
        raise AssertionError(label)


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. Base environment sanity
# ---------------------------------------------------------------------------

def test_base_env():
    section("1. PixelGameEnv (base)")
    env = PixelGameEnv(frame_size=(84, 84))
    obs, info = env.reset()

    check("obs dtype is uint8", obs.dtype == np.uint8, str(obs.dtype))
    check("obs shape is (84, 84, 3)", obs.shape == (84, 84, 3), str(obs.shape))
    check("obs range [0, 255]", obs.min() >= 0 and obs.max() <= 255)
    check("info has game_state", "game_state" in info)

    obs2, reward, terminated, truncated, info2 = env.step(0)
    check("step obs shape matches", obs2.shape == obs.shape)
    check("reward is float", isinstance(reward, float))
    check("terminated is bool", isinstance(terminated, bool))

    env.close()
    print("  Base env OK")


# ---------------------------------------------------------------------------
# 2. GrayscaleWrapper
# ---------------------------------------------------------------------------

def test_grayscale():
    section("2. GrayscaleWrapper")
    env = GrayscaleWrapper(PixelGameEnv(frame_size=(84, 84)))
    obs, _ = env.reset()

    check("obs shape is (84, 84, 1)", obs.shape == (84, 84, 1), str(obs.shape))
    check("obs dtype is uint8", obs.dtype == np.uint8, str(obs.dtype))
    check("obs range [0, 255]", obs.min() >= 0 and obs.max() <= 255)

    obs2, *_ = env.step(2)
    check("step obs shape matches", obs2.shape == (84, 84, 1))
    env.close()


# ---------------------------------------------------------------------------
# 3. ResizeWrapper
# ---------------------------------------------------------------------------

def test_resize():
    section("3. ResizeWrapper")
    # Start with a non-84×84 base to confirm resize is actually doing work
    env = PixelGameEnv(frame_size=(128, 128))          # raw is 128×128×3
    env = GrayscaleWrapper(env)                         # 128×128×1
    env = ResizeWrapper(env, size=(42, 42))             # target 42×42×1

    obs, _ = env.reset()
    check("obs shape is (42, 42, 1)", obs.shape == (42, 42, 1), str(obs.shape))
    check("obs dtype is uint8", obs.dtype == np.uint8)

    obs2, *_ = env.step(1)
    check("step obs shape matches", obs2.shape == (42, 42, 1))
    env.close()


# ---------------------------------------------------------------------------
# 4. NormalizeObsWrapper
# ---------------------------------------------------------------------------

def test_normalize():
    section("4. NormalizeObsWrapper")
    env = PixelGameEnv(frame_size=(84, 84))
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, size=(84, 84))
    env = NormalizeObsWrapper(env)

    obs, _ = env.reset()
    check("obs dtype is float32", obs.dtype == np.float32, str(obs.dtype))
    check("obs shape is (84, 84, 1)", obs.shape == (84, 84, 1), str(obs.shape))
    check("obs min >= 0.0", float(obs.min()) >= 0.0, str(obs.min()))
    check("obs max <= 1.0", float(obs.max()) <= 1.0, str(obs.max()))

    obs2, *_ = env.step(3)
    check("step obs in [0,1]", obs2.min() >= 0.0 and obs2.max() <= 1.0)
    env.close()


# ---------------------------------------------------------------------------
# 5. FrameStackWrapper
# ---------------------------------------------------------------------------

def test_frame_stack():
    section("5. FrameStackWrapper")
    env = PixelGameEnv(frame_size=(84, 84))
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, size=(84, 84))
    env = NormalizeObsWrapper(env)
    env = FrameStackWrapper(env, num_stack=4)

    obs, _ = env.reset()
    check("obs shape is (84, 84, 4)", obs.shape == (84, 84, 4), str(obs.shape))
    check("obs dtype is float32", obs.dtype == np.float32)
    check("obs in [0,1]", obs.min() >= 0.0 and obs.max() <= 1.0)

    # First 4 channels should all be identical (initial frame repeated)
    frames = [obs[:, :, i] for i in range(4)]
    all_same = all(np.array_equal(frames[0], f) for f in frames[1:])
    check("reset fills buffer with identical frames", all_same)

    obs2, *_ = env.step(2)
    check("step obs shape matches", obs2.shape == (84, 84, 4))

    # After one step, frames 0-2 should equal frames 1-3 of the reset obs
    check(
        "oldest frame rotated out",
        np.array_equal(obs2[:, :, 0], obs[:, :, 1]),
    )

    env.close()


# ---------------------------------------------------------------------------
# 6. RewardClipWrapper
# ---------------------------------------------------------------------------

def test_reward_clip():
    section("6. RewardClipWrapper")

    # Build a tiny env and monkey-patch its reward to exceed bounds
    class BigRewardEnv(PixelGameEnv):
        def _compute_reward(self):
            return 999.0  # far above 1.0

    env = RewardClipWrapper(BigRewardEnv(frame_size=(84, 84)), low=-1.0, high=1.0)
    env.reset()
    _obs, reward, _term, _trunc, info = env.step(0)

    check("reward clipped to 1.0", reward == 1.0, str(reward))
    check("raw_reward preserved in info", info.get("raw_reward") == 999.0, str(info.get("raw_reward")))

    class NegRewardEnv(PixelGameEnv):
        def _compute_reward(self):
            return -500.0

    env2 = RewardClipWrapper(NegRewardEnv(frame_size=(84, 84)), low=-1.0, high=1.0)
    env2.reset()
    _obs, reward2, *_ = env2.step(0)
    check("negative reward clipped to -1.0", reward2 == -1.0, str(reward2))

    env.close()
    env2.close()


# ---------------------------------------------------------------------------
# 7. EpisodicLifeWrapper
# ---------------------------------------------------------------------------

def test_episodic_life():
    section("7. EpisodicLifeWrapper")

    # Patch the base env so we can control lives directly
    class ControlledEnv(PixelGameEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._injected_lives: int | None = None

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            if self._injected_lives is not None:
                info["game_state"]["lives"] = self._injected_lives
                if self._injected_lives <= 0:
                    terminated = True
            return obs, reward, terminated, truncated, info

    base = ControlledEnv(frame_size=(84, 84))
    env = EpisodicLifeWrapper(base)

    obs, info = env.reset()
    initial_lives = env._lives
    check("initial lives recorded", initial_lives > 0, str(initial_lives))

    # Normal step — no life loss
    base._injected_lives = initial_lives          # lives unchanged
    obs2, _r, terminated, truncated, info2 = env.step(0)
    check("no life loss → not terminated", not terminated)

    # Simulate one life loss (lives go from N to N-1)
    base._injected_lives = initial_lives - 1
    _obs3, _r, terminated2, _trunc, _info3 = env.step(0)
    check("life loss → agent sees terminated=True", terminated2)

    env.close()


# ---------------------------------------------------------------------------
# 8. Full make_env() stack
# ---------------------------------------------------------------------------

def test_make_env():
    section("8. Full make_env() stack")
    from envs.pixel_env import PixelGameEnv

    env = make_env(
        env_factory=lambda: PixelGameEnv(frame_size=(84, 84)),
        frame_size=(84, 84),
        num_stack=4,
        reward_clip=(-1.0, 1.0),
        episodic_life=True,
    )

    obs, info = env.reset()
    check("obs shape is (84, 84, 4)", obs.shape == (84, 84, 4), str(obs.shape))
    check("obs dtype is float32", obs.dtype == np.float32)
    check("obs in [0, 1]", obs.min() >= 0.0 and obs.max() <= 1.0)

    obs2, reward, terminated, truncated, info2 = env.step(2)
    check("step obs shape matches", obs2.shape == (84, 84, 4))
    check("reward in [-1, 1]", -1.0 <= reward <= 1.0, str(reward))
    check("raw_reward in info", "raw_reward" in info2)
    check("terminated is bool", isinstance(terminated, bool))

    # Run 50 steps to confirm no crashes
    for _ in range(50):
        a = env.action_space.sample()
        env.step(a)
    check("50 random steps without crash", True)

    env.close()


# ---------------------------------------------------------------------------
# 9. Observation space consistency
# ---------------------------------------------------------------------------

def test_obs_space_consistency():
    section("9. Observation space consistency (obs ∈ declared space)")
    from envs.pixel_env import PixelGameEnv

    env = make_env(lambda: PixelGameEnv(frame_size=(84, 84)))
    obs, _ = env.reset()

    in_space = env.observation_space.contains(obs)
    check("reset obs ∈ observation_space", in_space, str(obs.dtype))

    obs2, *_ = env.step(env.action_space.sample())
    in_space2 = env.observation_space.contains(obs2)
    check("step obs ∈ observation_space", in_space2)

    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_base_env,
        test_grayscale,
        test_resize,
        test_normalize,
        test_frame_stack,
        test_reward_clip,
        test_episodic_life,
        test_make_env,
        test_obs_space_consistency,
    ]

    failures: list[str] = []
    for test_fn in tests:
        try:
            test_fn()
        except AssertionError as e:
            failures.append(str(e))
        except Exception as e:
            failures.append(f"{test_fn.__name__}: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    if failures:
        print(f"  {len(failures)} test(s) FAILED:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    else:
        total = len(tests)
        print(f"  All {total} test groups passed.")
    print(f"{'='*60}\n")
