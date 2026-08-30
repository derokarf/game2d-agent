#!/usr/bin/env bash
#
# One-time repo reorganization into runs/<game>/<approach>/<run>/{ckpt,train.log,media}.
# Preserves existing run identifiers (no renumbering). Prunes intermediate
# checkpoints from the dead pixel track (keeps best/final only).
#
# Safe to re-run: every move is guarded, missing sources are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."   # game2d_agent root

mvck() {  # move a checkpoint dir's contents into <dest>/ckpt, then drop the empty dir
  local src="$1" dest="$2"
  [ -d "$src" ] || return 0
  mkdir -p "$dest/ckpt"
  mv "$src"/* "$dest/ckpt"/ 2>/dev/null || true
  rmdir "$src" 2>/dev/null || true
}
mvlog() { [ -f "logs/$1" ] && mv "logs/$1" "$2/train.log" || true; }
mvgif() { local f="$1" dest="$2" name="$3"; [ -f "recordings/$f" ] && { mkdir -p "$dest/media"; mv "recordings/$f" "$dest/media/$name"; } || true; }

# ============================ lunar ============================
mvck models/lunar_ppo runs/lunar/ppo/run1
mvlog train_lunar.log runs/lunar/ppo/run1
mvgif lunar_best.gif  runs/lunar/ppo/run1 best.gif
mvgif lunar_final.gif runs/lunar/ppo/run1 final.gif

# ===================== collect2d / state_mlp ==================
mvck models/state_mlp runs/collect2d/state_mlp/run1        # run1 = unsuffixed dir
mvlog train_state_run1.log runs/collect2d/state_mlp/run1
for n in $(seq 2 16); do
  mvck "models/state_mlp_run$n" "runs/collect2d/state_mlp/run$n"
  mvlog "train_state_run$n.log" "runs/collect2d/state_mlp/run$n"
done
mvck models/state_mlp_diag  runs/collect2d/state_mlp/diag
mvlog train_state_diag.log  runs/collect2d/state_mlp/diag
mvck models/state_mlp_radar runs/collect2d/state_mlp/radar
mvlog train_state_radar.log runs/collect2d/state_mlp/radar

D=runs/collect2d/state_mlp
mvgif state_mlp_run2_best.gif  $D/run2  best.gif
mvgif state_mlp_run4_best.gif  $D/run4  best.gif
mvgif state_mlp_run7_best.gif  $D/run7  best.gif
mvgif state_mlp_run8_best.gif  $D/run8  best.gif
mvgif state_mlp_run9_best.gif  $D/run9  best.gif
mvgif state_mlp_run10_best.gif $D/run10 best.gif
mvgif state_mlp_run11_best.gif $D/run11 best.gif
mvgif state_mlp_run12_best.gif $D/run12 best.gif
mvgif state_mlp_run12_on_fixed_game.gif $D/run12 on_fixed_game.gif
mvgif state_mlp_run13_best.gif $D/run13 best.gif
mvgif run13_on_random_game.gif $D/run13 on_random_game.gif
mvgif run14_lowtemp.gif        $D/run14 lowtemp.gif
mvgif run15_milling_test.gif   $D/run15 milling_test.gif
mvgif run16_corner_test.gif    $D/run16 corner_test.gif
mvgif state_diag_best.gif      $D/diag  best.gif

# ===================== collect2d / state_gru =================
mvck models/state_gru runs/collect2d/state_gru/run1
mvlog train_state_gru_run1.log runs/collect2d/state_gru/run1
mvgif state_gru_best.gif runs/collect2d/state_gru/run1 best.gif

# ===================== collect2d / local_map ================
mvck models/map_run1 runs/collect2d/local_map/run1
mvlog train_map_run1.log runs/collect2d/local_map/run1
mvgif map_run1_corner_test.gif runs/collect2d/local_map/run1 corner_test.gif

# ================ collect2d / vision (loose) ================
# Prune the two heavy dirs: keep best/final only, delete ~2.4GB of intermediates.
for d in ppo ppo_run12; do
  if [ -d "models/$d" ]; then
    mkdir -p "runs/collect2d/vision/$d/ckpt"
    for keep in checkpoint_best.pt checkpoint_final.pt; do
      [ -f "models/$d/$keep" ] && mv "models/$d/$keep" "runs/collect2d/vision/$d/ckpt/" || true
    done
    rm -rf "models/$d"
  fi
done
for d in ppo_vision ppo_vision_run2 ppo_vision_run3 ppo_vision_run4 ppo_vision_run5 ppo_vision_gru_run1; do
  mvck "models/$d" "runs/collect2d/vision/$d"
done
mkdir -p runs/collect2d/vision/encoders runs/collect2d/vision/_logs runs/collect2d/vision/_media
mv models/vision_encoder.pt    runs/collect2d/vision/encoders/ 2>/dev/null || true
mv models/vision_encoder_v2.pt runs/collect2d/vision/encoders/ 2>/dev/null || true
mv logs/train_pixel_run*.log       runs/collect2d/vision/_logs/ 2>/dev/null || true
mv logs/train_vision_run*.log      runs/collect2d/vision/_logs/ 2>/dev/null || true
mv logs/train_vision_gru_run1.log  runs/collect2d/vision/_logs/ 2>/dev/null || true
for f in pixel.gif run4_deterministic.gif run4_eval.gif run4_temp05.gif run4_temp06.gif \
         run5_best.gif run7_best.gif run9_best.gif run11_best.gif gru_run1_best.gif; do
  mv "recordings/$f" runs/collect2d/vision/_media/ 2>/dev/null || true
done

echo "Migration done."
echo "Leftover in models/ (should be *.py source only):"
ls models/ | grep -v __pycache__
echo "Leftover logs/: $(ls logs 2>/dev/null | wc -l) | recordings/: $(ls recordings 2>/dev/null | wc -l)"
