# Maintenance — what to clean, and when

Use `scripts/cleanup.sh`. Run `--dry-run` first to preview.

## Always safe (regenerable) — clean anytime
Removed by `bash scripts/cleanup.sh` (the default):
- `**/__pycache__/` directories and `*.pyc` files — Python bytecode caches.
- (Outside the repo) the session scratchpad under `/tmp/claude-*/…/scratchpad/` —
  temporary frame dumps from GIF analysis. Safe to delete; not referenced by anything.

These never contain source, results, or anything you can't regenerate.

## Destructive — only on request (`--prune-ckpts`)
- **Intermediate checkpoints** inside `runs/<...>/ckpt/`: any `checkpoint_*.pt`
  that is *not* `checkpoint_best.pt`, `checkpoint_best_eval.pt`, or
  `checkpoint_final.pt`. Training that snapshots every N steps (the old pixel
  track did) leaves dozens of ~20 MB files; only best/final are usually worth
  keeping. `--prune-ckpts` deletes the rest.
  - This is how the one-time migration reclaimed ~2.4 GB from the dead pixel track.

## Never auto-clean (real results / source)
- `runs/**/train.log` — training logs are results; keep them.
- `runs/**/media/*.gif` — recordings; keep (delete individually if truly unwanted).
- `runs/**/ckpt/checkpoint_best*.pt`, `checkpoint_final.pt` — the models themselves.
- Anything in `envs/`, `models/*.py`, `scripts/`, `docs/`, `configs/`, `data/`.

## Suggested cadence
- **Every session / before committing:** `bash scripts/cleanup.sh` (caches).
- **When `du -sh runs` grows large:** review, then `bash scripts/cleanup.sh --prune-ckpts`
  after confirming the runs you care about still have their best/final.
- **Retiring an experiment:** keep its `train.log` + `best_eval`/`final`; prune the rest.
