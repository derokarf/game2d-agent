# Project Rules

## Training scripts

**Never run training or evaluation scripts on behalf of the user.** This includes:
- `scripts/train.py`
- `scripts/train_lunar.py`
- `scripts/play_lunar.py`
- `scripts/train_ppo_vision.py`
- `scripts/play_vision.py`
- any future training or rollout scripts

The user always runs these themselves in their own terminal. When it is time to start a run, provide the exact command to paste and stop there.

## Repository structure

See `docs/repo_structure.md` for the full layout. Key rules:

- **Source vs results:** `envs/`, `models/`, `scripts/` are **source only**. All training
  outputs (checkpoints, logs, recordings) live under
  `runs/<game>/<approach>/<run>/{ckpt/, train.log, media/}`. **Never write checkpoints into `models/`.**
- `<game>`: `collect2d` (current), `collect2d_b` (transfer game), `lunar`.
  `<approach>`: `state_mlp`, `state_gru`, `vision`, `local_map`.
- Checkpoint names: `checkpoint_best.pt`, `checkpoint_best_eval.pt` (usually the one to load),
  `checkpoint_final.pt`. Recordings: `media/<purpose>.gif` (`best`, `corner_test`, `lowtemp`,
  `transfer_zeroshot`, …).
- `runs/collect2d/vision/` is a retired track, loosely grouped (`_logs/`, `_media/`); don't rebind it.
- Always use the project venv: `.venv/bin/python`.

## Game mechanics reference

`docs/game_mechanics.md` is the single source of truth for game + perception mechanics
(world, obstacles, targets, reward, ray cast, compass, occupancy grid, geodesic field).
**When you change any mechanic in `envs/pixel_env.py`, `envs/state_env.py`, or
`envs/local_map_env.py` — a constant, spawn rule, reward term, or perception scheme —
update the matching section of `docs/game_mechanics.md` in the same change.** Do not keep
a second copy of these facts elsewhere; link to that doc instead.

## Maintenance

`docs/maintenance.md` + `scripts/cleanup.sh`. Safe cleanup (caches) = `bash scripts/cleanup.sh`;
destructive checkpoint pruning is opt-in via `--prune-ckpts`. Never auto-delete logs, recordings,
or best/final checkpoints.

## Current work

Gate 3 (transfer): the local-map CNN agent (`runs/collect2d/local_map/run1`) passed gates 1 & 2;
next is a game B for zero-shot + fine-tune transfer measurement. See `docs/agent_v2_local_map_spec.md`.