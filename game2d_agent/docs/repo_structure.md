# Repository structure

Orientation for anyone (human or agent) working in this repo. Read this before
creating files so new artifacts land in the right place.

## Top level
```
game2d_agent/
├── envs/         # SOURCE: environments (game logic + observation variants)
├── models/       # SOURCE ONLY: policy & PPO .py files — NO checkpoints here
├── scripts/      # SOURCE: train_*/play_* entry points + utilities
├── configs/      # static config files
├── data/         # datasets (e.g. collected vision data)
├── docs/         # design specs & this file
├── runs/         # ALL experiment outputs (checkpoints, logs, recordings)
├── requirements.txt   README.md   CLAUDE.md
└── .venv/        # project virtualenv (use .venv/bin/python)
```

**Golden rule:** `envs/`, `models/`, `scripts/` hold *source only*. Every training
*output* — checkpoints, logs, recordings — goes under `runs/`. Never write
checkpoints into `models/` again.

## Source layout
- `envs/pixel_env.py`     — base game (physics, spawning, reward incl. geodesic BFS teacher).
- `envs/state_env.py`     — flat state-vector observation (26/49-dim).
- `envs/local_map_env.py` — egocentric occupancy-map + goal-vector observation (agent v2).
- `envs/wrappers.py`      — gym wrappers (e.g. ActionRepeat).
- `models/*_policy.py`    — networks: `mlp_policy`, `gru_policy`, `cnn_map_policy`, `vision_encoder`.
- `models/ppo*.py`        — PPO variants: `ppo` (base), `ppo_mlp` (vector), `ppo_map` (dict obs).
- `scripts/train_<approach>.py`, `scripts/play_<approach>.py`.

## runs/ — experiment outputs
```
runs/<game>/<approach>/<run>/
    ├── ckpt/           # checkpoint_best.pt, checkpoint_best_eval.pt, checkpoint_final.pt
    ├── train.log       # the training log for this run
    └── media/          # *.gif recordings
```
- `<game>`     : `collect2d` (current), `collect2d_b` (gate-3 transfer game), `lunar`.
- `<approach>` : `state_mlp`, `state_gru`, `vision`, `local_map`.
- `<run>`      : `run1`, `run2`, … (or a label like `diag`, `radar`).

### Naming conventions
- **Checkpoints:** `checkpoint_best.pt` (best train ep_rew), `checkpoint_best_eval.pt`
  (best fixed-seed eval — usually the one to load), `checkpoint_final.pt`.
- **Log:** exactly one `train.log` per run (the path already identifies the run).
- **Recordings:** `media/<purpose>.gif` with a fixed vocabulary:
  `best`, `eval`, `corner_test`, `lowtemp`, `on_random_game`,
  `transfer_zeroshot`, `transfer_finetuned`.

### Exception: `runs/collect2d/vision/` (retired track, loosely grouped)
The old pixel/vision CNN track has a fuzzy log↔checkpoint↔recording binding, so it
is *not* per-run nested. Checkpoint dirs keep their original names; logs live in
`vision/_logs/`, recordings in `vision/_media/`, shared encoder weights in
`vision/encoders/`. Leave it as archival; don't try to rebind it.

## Script defaults
Active scripts default their output paths into the tree, e.g. `train_map.py`:
`--save-dir runs/collect2d/local_map/run1/ckpt`, `--log-file …/run1/train.log`.
For a new run, pass `--save-dir runs/<game>/<approach>/<runN>/ckpt` and the matching
`--log-file`/`--out`. Always run scripts with the project venv: `.venv/bin/python`.

## See also
- `docs/agent_v2_local_map_spec.md` — the current agent design, gates, and transfer plan.
- `docs/maintenance.md` — what is safe to clean periodically.
