# Game mechanics — technical reference

Single source of truth for how the game and the agent's perception actually work.

> **⚠ Keep this in sync with the code.** If you change any mechanic in
> `envs/pixel_env.py`, `envs/state_env.py`, or `envs/local_map_env.py` — a
> constant, a spawn rule, the reward, or a perception scheme — update the
> matching section here in the same change. Values below are quoted from the
> code with `file:line` anchors; verify them if something looks stale.

All three environments share one game (`PixelGameEnv`); they differ only in the
**observation** they return. Physics, spawning, and reward are identical.

---

## World & player  (`pixel_env.py`)
- **Board:** per-game via `screen_size` (`envs/games.py`): `collect2d` = 640 × 480,
  `collect2d_b` = 800 × 600. Assigned in `pixel_env.py:__init__`.
- **Player:** radius **20** px (`:51`), speed **8** px/step (`:93`). Starts center,
  safe-spawned away from obstacles.
- **Actions:** 5 discrete (`_apply_action :227`) — `0` noop, `1` left, `2` right,
  `3` up, `4` down. **4-directional + noop** (no diagonals → diagonal travel is a
  zigzag staircase, which is expected, not a bug).
- **Episode:** 1000 steps max (truncation). With solid obstacles `lives` never
  drops, so episodes run to the step cap.

## Obstacles — solid walls  (`pixel_env.py`)
- **Count:** `n_obstacles` (cap). With `--random-counts`, re-rolled per level to
  `randint(min_obstacles, n_obstacles+1)`, i.e. 1..cap (`_sample_counts :79`).
- **Size:** `w, h ∈ [obstacle_size)` px, per-game via `envs/games.py`
  (`collect2d` = [60,120), `collect2d_b` = [50,160)); used in `_spawn_obstacles`.
  **Position:** kept fully on-screen.
- **Passable-gap guarantee:** edge-to-edge gap between any two obstacles ≥
  `min_gap = 2*player_radius + 12 = 52` px (`:131`, `:158`). Ensures every gap is
  traversable — a thin ray or line-of-sight can't lure the agent into an
  impassable slot.
- **Solid behaviour:** movement is blocked **per-axis** if the new player center
  would come within `player_radius` (20 px) of an obstacle edge (`_apply_action`
  → `_blocked :180`). Per-axis blocking lets the player **slide along** a wall
  instead of sticking. No collision penalty, no phasing through.
- **Screen edges** are also walls (position clipped to the board).
- On a mid-episode board-clear respawn, obstacles avoid the player (within
  `player_radius + 30 = 50` px, `hits_player :146`) so a wall can't spawn on it.

## Targets ("rewards")  (`pixel_env.py`)
- **Count:** `n_targets` (cap); with `--random-counts`, 1..cap per level.
- **Radius:** 25 px (`:119`). **Spawn:** avoids obstacle edges by ≥35 px, 100
  retries (`_spawn_targets :106`).
- **Collection:** when player-center→target distance < `radius + 20 = 45` px →
  `+10` reward, target removed (`:382-384`).
- **Board clear:** when the last target is collected → `score += 1`, `+50` reward,
  and a **new level**: counts + layout re-rolled (obstacles first, avoiding the
  player; then targets, avoiding obstacles) (`:391-397`).

## Reward — the training teacher (never observed)  (`pixel_env.py:_compute_reward`)
- `−0.01` per step (time penalty).
- `+10` per target collected · `+50` per board clear.
- **Geodesic approach shaping:** `+0.05 × (prev_geo − geo)` (`:417`) — reward for
  real progress toward the nearest target along an *obstacle-avoiding* path
  (see geodesic field below). Skipped on collection steps (baseline reset).

---

## Perception schemes (what the agent sees)

### 1. Ray cast — clearance compass  (`state_env.py`, the state agent)
- **32 rays** evenly spaced `0..2π` from the player (`N_RAYS=32 :30`, `:54`).
- Each = distance to the first barrier (obstacle edge **or** screen wall) along
  the ray, via slab (ray-vs-AABB) intersection, normalized by the screen diagonal
  (√(640²+480²) ≈ 800) (`_cast_rays`).
- Full state obs = **49-dim**: player pos (2) + 5 targets ×(dx,dy,dist) (15) +
  32 rays (32) (`OBS_DIM :31`).

### 2. Compass — goal bearing  (`local_map_env.py`, the "global" vector)
- **Not** the ray cast, and **not** the geodesic needle. It is a straight-line
  "quest marker": unit bearing `(dx/|d|, dy/|d|)` to the **nearest** target +
  normalized distance = **3 values** (`_extract_obs`).
- Points *at* the goal, possibly through a wall — the agent must still learn to
  route around. `GLOBAL_DIM=3 :29`.

### 3. Grid — egocentric occupancy map  (`local_map_env.py`, the "map" channels)
- `MAP_SIZE=17`, `MAP_CELL=12` px → **±102 px** window (204 px), agent at the
  center cell (`:27-28`).
- **2 channels** (`_extract_obs`): `[0]` blocked = inside an obstacle OR outside
  the screen; `[1]` target present (nearest cell to each in-window target).
- Raw occupancy (not radius-inflated); shape `(2, 17, 17)`.
- Used by the local-map CNN (agent v2) alongside the compass vector.

### Geodesic distance field — reward internals only  (`pixel_env.py`)
Not part of any observation; used only to compute the shaping reward.
- Grid cell **10 px**; obstacles inflated by `player_radius` (configuration
  space) (`_compute_distance_field`, `_dist_cell_size=10 :68`).
- **Multi-source, 4-connected BFS** from *all* targets → distance-to-nearest-
  target field that routes around walls (4-connected matches the agent's
  movement). Cached; recomputed only when the target/obstacle set changes.
- Read via **bilinear interpolation** at the player's continuous position
  (`_geodesic_dist`) so the shaping gradient is smooth — this fixed the
  chunky/tie corner-stalls (see `docs/agent_v2_local_map_spec.md`).

## Which env → which observation
| Env | Observation | Agent / track |
|---|---|---|
| `pixel_env.py`     | 84×84×3 pixels          | vision CNN (retired) |
| `state_env.py`     | 49-dim vector (rays)    | state MLP / GRU |
| `local_map_env.py` | `{map(2,17,17), global(3)}` | local-map CNN (current) |
