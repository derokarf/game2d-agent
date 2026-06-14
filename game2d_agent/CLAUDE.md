# Project Rules

## Training scripts

**Never run training or evaluation scripts on behalf of the user.** This includes:
- `scripts/train.py`
- `scripts/train_lunar.py`
- `scripts/play_lunar.py`
- any future training or rollout scripts

The user always runs these themselves in their own terminal. When it is time to start a run, provide the exact command to paste and stop there.