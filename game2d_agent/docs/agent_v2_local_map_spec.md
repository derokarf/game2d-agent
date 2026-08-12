# Agent v2 — Local-Map CNN spec

## Design principle
Judge every choice by *"does this survive game B,"* not *"does this squeeze points on
game A."* The network must learn the **routing**; the environment only provides
**egocentric, relative, count-independent** perception. The geodesic BFS stays as a
**training-only teacher** (reward), never in the observation.

## Background / why this design
Two different perception schemes on the state agent (per-object obstacle slots, then a
32-ray "compass") both let the policy **freeze at dead-spots**: the interpolated geodesic
field is *followable* (a greedy follower reaches every target, 0 stalls), but the trained
policy can't perceive the route — it only sees straight-line target bearing + rays and must
*infer* the detour, which collapses in some configurations. Target reachability was verified
clean (0 unreachable of 9,455 across 800 seeds × ~4 levels), so the freeze is a
**perception-learnability** problem, not an environment bug. The fix is perception that makes
routing learnable (a local spatial map + CNN), keeping the geodesic reward as the teacher.

## Observation (the contract)
Two parts — a local map for *how to move*, a global vector for *where to go*.

### 1. Egocentric occupancy map — `2 × 17 × 17`
- Centered on the agent, **12px cells**, covering **±102px** (204px window) — enough to see
  the edges/corners of the nearest obstacle and the gaps beside it.
- **Channel 0 — blocked:** 1 if the cell is inside an obstacle *or* outside the screen
  (screen edge = wall). "Can't be here."
- **Channel 1 — target:** 1 if a target sits in the cell. "Goal here."
- Agent is always at the center → not encoded (implicit).
- **Raw** occupancy (not inflated by player radius) — the CNN learns the clearance itself;
  keeps it transfer-pure across agent sizes.

### 2. Global goal vector — `3` values
- Unit bearing to nearest target `(dx/|d|, dy/|d|)` — the straight-line "quest marker."
  *Points at the goal, possibly through a wall; the agent must still learn to route.*
  This is **not** the needle (the needle points along the solved path; the bearing does not).
- Normalized distance to nearest target.
- Handles targets *outside* the window; the local map handles nearby routing. Clean division
  of labor.

Everything is relative/egocentric — no absolute coordinates, no fixed entity slots.

## Architecture — small CNN actor-critic
```
map (2×17×17) ─ Conv 2→16  (3×3, p1)   ReLU  → 16×17×17
              ─ Conv 16→32 (3×3, s2,p1) ReLU  → 32×9×9
              ─ Conv 32→32 (3×3, s2,p1) ReLU  → 32×5×5
              ─ flatten (800) → FC 128 ReLU
global (3) ───────────────────────────────────┐
concat[128, 3] → FC 128 ReLU → ┬ policy head (→5 actions)
                               └ value head  (→1)
```
~0.3M params. The CNN *is* the answer to "where's the MLP" — it's the thing that learns
spatial routing.

## Reward — unchanged
Keep the **interpolated geodesic field** as the teacher (training-only, no runtime
dependency). This part already works; we're only fixing perception.

## What's new (files)
- `envs/local_map_env.py` — returns `{map, global}` obs (rasterize obstacles/targets into the
  egocentric window each step; cheap).
- `models/cnn_map_policy.py` — `CNNActorCritic(map + global → policy/value)`.
- `models/ppo_map.py` — rollout buffer that stores dict obs.
- `scripts/train_map.py`, `scripts/play_map.py`.
- Env physics, spawn logic, geodesic reward: reused as-is.

## Transfer plan (the actual deliverable)
1. Train on **game A** (current game).
2. **Zero-shot** eval on **game B** — same engine, deliberately different: larger board
   (e.g. 800×600), different obstacle sizes, and/or a different objective ("reach the exit" =
   one goal cell instead of collect-all). The contract (blocked+target channels, bearing)
   still applies unchanged.
3. **Fine-tune** on B; measure recovery speed.
- **Metrics:** zero-shot score vs from-scratch, and steps-to-recover. That curve *is* the
  result.

## Acceptance gates
1. **Stuck diagnostic** (temp 0.15 corner test): **no freezes.** Pass/fail. ← the thing we've
   been chasing.
2. Game-A eval competitive with run16 (~600+).
3. Zero-shot on B meaningfully above random ("a few good scores").
- Fallbacks if gate 1 leaves residue: **GRU** (commitment) → **mild anti-stall reward**.

## Decisions / defaults
| Choice | Default | Alternative |
|---|---|---|
| Window / cell | 17×17 @ 12px (±102px) | 21×21 @ 10px (finer, bigger, slower) |
| Channels | 2 (blocked, target) | +hazard channel (for future games) |
| Occupancy | raw (transfer-pure) | config-space inflated (easier to learn) |
| CNN depth | 3 conv | 2 conv (lighter) |
| Memory | none at first | add GRU if stuck-gate fails |

Build the defaults as-is (transfer-first choices); reach for alternatives (finer grid,
config-space, GRU) only if a gate fails, to avoid over-engineering before measuring.

## Status
- [x] `envs/local_map_env.py`
- [x] `models/cnn_map_policy.py` (134K params)
- [x] `models/ppo_map.py`
- [x] `scripts/train_map.py`, `scripts/play_map.py`
- [x] Pipeline smoke test (obs shapes, CNN forward, 2 PPO iters all finite, ~727 SPS CPU)
- [x] Gate 1: stuck diagnostic — PASS. 0 true freezes over 4196 frames (run16 had a
      675-frame dead-freeze). Residual: rare brief (~3s) hesitations at SCREEN EDGES only.
- [x] Gate 2: game-A eval — PASS. run1 best eval 1143 (mean ~1050-1100), vs run16's 640 (~1.8x).
      Plateaued by ~1.2M steps; 5M was more than needed.
- [ ] Game B + transfer measurement (zero-shot -> fine-tune)

## Known residual (map_run1)
Occasional brief hesitation when pinned against a SCREEN edge/corner with a target in the
corner. Hypothesis: the map encodes screen boundary in the same "blocked" channel as
obstacles, so edge vs wall are indistinguishable to the CNN, while they behave differently
(you can slide along an edge). Candidate fix if we chase it: a separate boundary channel,
or the spec fallbacks (GRU / mild anti-stall).
