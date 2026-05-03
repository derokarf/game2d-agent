# 2D Game Pixel AI Agent

A reinforcement learning agent that plays custom 2D games using only pixel input (no game state access).

## Architecture

```
[Pixels] → [CNN Feature Extractor] → [DQN Policy] → [Action]
```

## Project Structure

```
game2d_agent/
├── envs/
│   ├── __init__.py
│   └── pixel_env.py          # Custom Gymnasium environment
├── scripts/
│   ├── train.py              # DQN training script
│   └── play.py               # Watch trained agent play
├── models/                   # Saved model checkpoints
├── logs/                     # TensorBoard logs
├── recordings/               # Gameplay videos
├── configs/                  # Configuration files
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train

```bash
cd game2d_agent
python scripts/train.py --total-timesteps 500000
```

**Key arguments:**
- `--total-timesteps` - Training steps (default: 500000)
- `--num-envs` - Parallel environments (default: 4)
- `--learning-rate` - Learning rate (default: 1e-4)
- `--seed` - Random seed

### Play

```bash
python scripts/play.py --model-path models/dqn/dqn_final.zip --save-video
```

**Key arguments:**
- `--model-path` - Path to trained model
- `--save-video` - Record gameplay as MP4
- `--no-render` - Run without visual display (faster)
- `--deterministic` - Use deterministic policy

### Monitor Training

```bash
tensorboard --logdir logs/dqn
```

## Customizing for Your Game

Edit `envs/pixel_env.py` to match your game:

1. **`_init_game()`** - Set up initial game state
2. **`_apply_action()`** - Map action IDs to game inputs (keyboard/mouse)
3. **`_render_frame()`** - Capture current game frame as RGB array
4. **`_compute_reward()`** - Define reward signal for learning

### Action Space (default 5 actions)

| Action | Description |
|--------|-------------|
| 0 | No-op |
| 1 | Move left |
| 2 | Move right |
| 3 | Move up |
| 4 | Move down |

To change actions, modify `self.action_space` in `__init__()` and `_apply_action()`.

### Integrating External Game (Screen Capture)

For games you don't have source code for, replace `_render_frame()` with screen capture:

```python
import mss

def _render_frame(self):
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 640, "height": 480}
        frame = np.array(sct.grab(monitor))
        return frame[:, :, :3]  # Remove alpha channel
```

Use `pynput` or `pyautogui` to send input actions.

## Tips for Better Training

1. **Increase timesteps** - 1M+ for complex games
2. **Tune reward** - Make it proportional to progress
3. **Add frame stacking** - See `FrameStackWrapper` in train.py
4. **Try PPO** - Replace `DQN` with `PPO` in train.py for more stable learning
5. **Reduce frame size** - 84x84 is fast, 128x128 for detail-heavy games
