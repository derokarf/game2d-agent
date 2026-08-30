#!/usr/bin/env bash
#
# Periodic maintenance. See docs/maintenance.md.
#
#   bash scripts/cleanup.sh              # safe: caches only (default)
#   bash scripts/cleanup.sh --dry-run    # show what safe-clean would remove
#   bash scripts/cleanup.sh --prune-ckpts        # ALSO prune intermediate checkpoints
#   bash scripts/cleanup.sh --prune-ckpts --dry-run
#
# Safe cleanup removes only regenerable caches (never source, logs, or results).
# --prune-ckpts additionally deletes non-best/final checkpoints inside runs/*/ckpt
# (keeps checkpoint_best.pt / checkpoint_best_eval.pt / checkpoint_final.pt). This
# is DESTRUCTIVE and off by default.
set -uo pipefail
cd "$(dirname "$0")/.."

DRY=0; PRUNE=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --prune-ckpts) PRUNE=1 ;;
    *) echo "unknown arg: $a"; exit 1 ;;
  esac
done
run() { if [ "$DRY" = 1 ]; then echo "[dry] $*"; else eval "$*"; fi; }

echo "== safe cleanup: python caches =="
while IFS= read -r d; do run "rm -rf '$d'"; done < <(find . -path ./.venv -prune -o -type d -name __pycache__ -print)
while IFS= read -r f; do run "rm -f '$f'"; done < <(find . -path ./.venv -prune -o -type f -name '*.pyc' -print)

if [ "$PRUNE" = 1 ]; then
  echo "== prune intermediate checkpoints (keep best / best_eval / final) =="
  # Only touches numbered/step checkpoints like checkpoint_1024000.pt or checkpoint_latest.pt
  while IFS= read -r f; do
    base=$(basename "$f")
    case "$base" in
      checkpoint_best.pt|checkpoint_best_eval.pt|checkpoint_final.pt) : ;;  # keep
      checkpoint_*.pt) run "rm -f '$f'" ;;
    esac
  done < <(find runs -type f -name 'checkpoint_*.pt')
fi

echo "== disk usage =="
du -sh runs models 2>/dev/null
if [ "$DRY" = 1 ]; then echo "done (dry-run)."; else echo "done."; fi
