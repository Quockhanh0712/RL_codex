# Orbit Wars — v4 + ML Shot Validator + topk1 (v3 update)

**Reference LB ~970** (v3, expected from local eval)

> **v3 update (May 2026)**: Reverted to `topk1` after a local sweep showed that for *this* self-contained validator (trained only on simple inline opponents), topk1 outperforms topk2 by **+11 pp** on LB-tier opponents (52% vs 41% in head-to-head). The earlier v2 attempt at topk2 — which had been validated against a much stronger validator trained on real LB-1450+ players' replays — did **not transfer** to this notebook's weaker self-trained validator. See section 6 for the full sweep.
>
> Data-collection improvements from v2 are **kept**: 1 extra inline opponent (`defender`) and ~70% more games (10 vs 6 seeds for opps, 30 vs 24 for self-play). These data changes added +7 pp by themselves with topk1 (52% vs 45%).

This notebook reproduces the agent `v4_mlhybrid_t30_topk1` (= v4 + ML validator + top-1 throttle) end-to-end. We train a small ML shot validator on top of a strong rule-based base agent, throttle moves down to one per turn, and produce a `submission.tar.gz` ready for the Orbit Wars competition — all inside this single Kaggle notebook.

## Acknowledgments

This work builds directly on excellent public Kaggle notebooks. **Huge thanks to the authors** for sharing their work:

| Component | Source | Author |
|---|---|---|
| `orbitbotnext` (rule-base foundation) | [orbit_botnext](https://www.kaggle.com/code/pascalledesma/orbit-botnext) | Pascal Ledesma |
| `orbitwork v14` (progenitor of obn) | [orbitwork-v14](https://www.kaggle.com/code/pascalledesma/orbitwork-v14) | Pascal Ledesma |
| `LB-1224 constants` (used in `obn_v4_exp004ish`) | [orbit-star-wars-lb-max-1224](https://www.kaggle.com/code/) | exp004 series |
| `LB-1100 constants` (alternative) | [distance-prioritized-agent](https://www.kaggle.com/code/) | exp004 series |
| `ow_proto` (alternative strong agent) | [orbit-wars-agent-ow-proto](https://www.kaggle.com/code/djenkivanov/orbit-wars-agent-ow-proto) | djenkivanov |

Without these public agents, the hybrid approach in this notebook would not exist.

## Pipeline Overview

1. **Strategy overview** — why a 3-layer hybrid works
2. **Environment setup** — install `kaggle-environments` (offline from a bundled wheel)
3. **Data collection** — v4 self-plays vs 5 opponents, every shot is recorded with a success/failure label
4. **Feature extraction** — 24-dim hand-crafted features per shot
5. **Model training** — small MLP with PyTorch (CUDA → CPU fallback)
6. **NumPy export** — convert weights so the Kaggle submission can run without PyTorch
7. **Agent assembly** — combine v4 source + numpy validator + topk1 wrapper into a single `main.py`
8. **`submission.tar.gz`** — ready to submit

## Strategy: 3-layer hybrid

```
v4 (rule-based foundation, ~3,400 lines)
   ↓ proposes ~6.7 moves per turn
ML shot validator (3-layer MLP, ~3,500 params)
   ↓ drops moves with success probability < 0.30
topk1 throttle (a one-line filter)
   ↓ keeps only the largest fleet move
final action: at most 1 move per turn
```

### Why this works

We analyzed 191 top replays:

| Agent | Moves / turn | Idle turns |
|---|---|---|
| `LB #1` | **1.3** | **63%** |
| `v4` baseline | 6.7 | 4% |

Top-tier players **wait, accumulate, and then strike decisively**. `v4` instead sprays small moves every turn, leaving its home garrison thin and vulnerable to counter-attacks.

- **`topk1`** keeps only the move with the largest ship count per turn — a 41-line wrapper around `v4`. This alone gives **+4.5 LB** over plain `v4`.
- **ML validator** learns "does this attack actually capture the planet?" and drops low-probability shots, letting `topk1` pick from a pre-filtered pool.

The two layers act on **different axes** (target quality vs. over-aggression), so combining them compounds: `topk1` alone reached LB 892, the ML+topk1 hybrid reached LB ~1049 ( **+75 LB** ).

## 0. Prerequisites (important)

To run this notebook you need:

1. **`obn_v4_exp004ish.py`** — the v4 base agent source (3,378 lines). This is `orbitbotnext` (by Pascal Ledesma) extended with 6 constant tunings drawn from public LB-1224 / LB-1100 notebooks. Place it in a Kaggle Dataset and attach.
2. **`kaggle_environments-1.28.1-py3-none-any.whl`** — needed because Kaggle's preinstalled `kaggle_environments` does not yet include the `orbit_wars` environment, AND competition-linked notebooks have internet disabled. We install it offline from a bundled wheel.

Both files are bundled in a single dataset (recommended slug: `orbit-wars-v4-base`). The next cell looks up the v4 source from that dataset.


```python
# Locate v4 source
import os
from pathlib import Path

# ↓ Adjust to match your dataset slug
DATASET_SLUG = "orbit-wars-v4-base"
V4_PATH = Path(f"/kaggle/input/datasets/konbu17/{DATASET_SLUG}/obn_v4_exp004ish.py")

# Fallback: if a copy already exists in /kaggle/working/, use that
if not V4_PATH.exists():
    fallback = Path("/kaggle/working/obn_v4_exp004ish.py")
    if fallback.exists():
        V4_PATH = fallback

assert V4_PATH.exists(), (
    f"v4 base not found.\n"
    f"Please attach a Kaggle Dataset slugged `{DATASET_SLUG}` containing\n"
    f"`obn_v4_exp004ish.py`, OR place the file at `/kaggle/working/obn_v4_exp004ish.py` directly."
)

# Copy to working dir for the rest of the pipeline
WORK = Path("/kaggle/working")
WORK.mkdir(exist_ok=True)
import shutil
V4_LOCAL = WORK / "obn_v4_exp004ish.py"
shutil.copy(V4_PATH, V4_LOCAL)
print(f"v4 base loaded: {V4_LOCAL} ({V4_LOCAL.stat().st_size:,} bytes)")
```

    v4 base loaded: /kaggle/working/obn_v4_exp004ish.py (113,252 bytes)
    

## 1. Environment setup

The `orbit_wars` environment was added in `kaggle-environments` 1.28+, but the Kaggle default image ships an older version. **Competition-linked notebooks also have internet disabled**, so we install the wheel offline from the bundled dataset.


```python
# Kaggle's preinstalled kaggle_environments doesn't include orbit_wars,
# AND competition-linked notebooks have internet disabled.
# So we install from the wheel bundled in our dataset.
import sys, subprocess, glob, os as _os

WHEEL_GLOB = "/kaggle/input/datasets/konbu17/orbit-wars-v4-base/kaggle_environments-*.whl"
wheels = glob.glob(WHEEL_GLOB)
assert wheels, f"wheel not found at {WHEEL_GLOB} — make sure dataset 'orbit-wars-v4-base' is attached"
wheel = wheels[0]
print(f"Installing {wheel}")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", wheel], check=True)

# Reload to pick up the new version (in case kaggle_environments was already imported)
import importlib, kaggle_environments
importlib.reload(kaggle_environments)
envs = sorted(_os.listdir(_os.path.join(_os.path.dirname(kaggle_environments.__file__), "envs")))
print(f"kaggle_environments {kaggle_environments.__version__}; orbit_wars available: {'orbit_wars' in envs}")
assert "orbit_wars" in envs, f"orbit_wars not in envs: {envs}"

import torch
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
```

    Installing /kaggle/input/datasets/konbu17/orbit-wars-v4-base/kaggle_environments-1.28.1-py3-none-any.whl
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO: Successfully loaded OpenSpiel environments: 16.
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_backgammon
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_checkers
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_chess
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_connect_four
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_gin_rummy
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_go
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_goofspiel
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_hearts
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_hex
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_matching_pennies_3p
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_othello
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_repeated_game
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_tic_tac_toe
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_universal_poker
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_repeated_poker
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    open_spiel_python_repeated_pokerkit
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO: OpenSpiel games skipped: 1.
    [kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO:    snake
    kaggle_environments 1.28.1; orbit_wars available: True
    torch 2.10.0+cpu, cuda available: False
    

## 2. Define baseline opponents for self-play

We need opponents to generate training data. The original `exp_arch_001` pipeline uses 5 stronger agents (`v1_sniper`, `v2_structured`, `exp007_tier3/4`, `orbitbotnext`), but for a self-contained public notebook we use 5 simple agents written inline:

- `nearest_sniper` — attack the nearest non-allied planet (the official sample agent)
- `weakest_first` — attack the planet with the fewest defenders
- `production_first` — prioritize high-production planets
- `defender` — consolidate ships toward your own planets
- `random_play` — purely random shots (negative-class generator)

A stronger opponent pool produces a stronger validator. If you have access to better agents, swap them in by replacing the entries in `OPPONENT_CODES` below or pointing `OPPONENT_PATHS` at external files.


```python
import math, random
from pathlib import Path

OPPONENTS_DIR = WORK / "opponents"
OPPONENTS_DIR.mkdir(exist_ok=True)

OPPONENT_CODES = {
    "nearest_sniper.py": '''
import math
def agent(obs, config=None):
    me = obs["player"] if isinstance(obs, dict) else obs.player
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    moves = []
    for src in planets:
        sid, owner, sx, sy, sr, ships, _ = src
        if int(owner) != int(me) or int(ships) < 2: continue
        best_pid, best_d, best_pxy = -1, 1e9, None
        for tgt in planets:
            tid, towner, tx, ty, tr, tships, _ = tgt
            if int(tid) == int(sid): continue
            if int(towner) == int(me): continue
            d = math.hypot(float(tx)-float(sx), float(ty)-float(sy))
            if d < best_d:
                best_d, best_pid, best_pxy = d, int(tid), (float(tx), float(ty), int(tships))
        if best_pid < 0 or best_pxy is None: continue
        ang = math.atan2(best_pxy[1]-float(sy), best_pxy[0]-float(sx))
        send = min(int(ships)-1, best_pxy[2]+1)
        if send > 0: moves.append([int(sid), float(ang), int(send)])
    return moves
''',
    "weakest_first.py": '''
import math
def agent(obs, config=None):
    me = obs["player"] if isinstance(obs, dict) else obs.player
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    moves = []
    for src in planets:
        sid, owner, sx, sy, sr, ships, _ = src
        if int(owner) != int(me) or int(ships) < 5: continue
        candidates = []
        for tgt in planets:
            tid, towner, tx, ty, tr, tships, _ = tgt
            if int(tid) == int(sid) or int(towner) == int(me): continue
            candidates.append((int(tships), float(tx), float(ty), int(tid)))
        if not candidates: continue
        candidates.sort()
        ts, tx, ty, tid = candidates[0]
        ang = math.atan2(ty-float(sy), tx-float(sx))
        send = min(int(ships)-1, ts+2)
        if send > 0: moves.append([int(sid), float(ang), int(send)])
    return moves
''',
    "production_first.py": '''
import math
def agent(obs, config=None):
    me = obs["player"] if isinstance(obs, dict) else obs.player
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    moves = []
    for src in planets:
        sid, owner, sx, sy, sr, ships, _ = src
        if int(owner) != int(me) or int(ships) < 5: continue
        best = None
        for tgt in planets:
            tid, towner, tx, ty, tr, tships, prod = tgt
            if int(tid) == int(sid) or int(towner) == int(me): continue
            score = float(prod) - 0.05 * int(tships)
            if best is None or score > best[0]:
                best = (score, float(tx), float(ty), int(tid), int(tships))
        if best is None: continue
        _, tx, ty, tid, tships = best
        ang = math.atan2(ty-float(sy), tx-float(sx))
        send = min(int(ships)-1, tships+3)
        if send > 0: moves.append([int(sid), float(ang), int(send)])
    return moves
''',
    "defender.py": '''
import math
def agent(obs, config=None):
    me = obs["player"] if isinstance(obs, dict) else obs.player
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    fleets = obs["fleets"] if isinstance(obs, dict) else obs.fleets
    mine = [p for p in planets if int(p[1]) == int(me)]
    if not mine: return []
    # Identify which of my planets has an incoming enemy fleet within 40 units
    threats = set()
    for f in fleets:
        if int(f[1]) == int(me): continue
        for p in mine:
            if math.hypot(float(f[2])-float(p[2]), float(f[3])-float(p[3])) < 40:
                threats.add(int(p[0]))
                break
    moves = []
    for src in mine:
        sid, _, sx, sy, sr, ships, _ = src
        if int(ships) < 8: continue
        if int(sid) in threats:
            # Reinforce: send to nearest threatened ally
            best = None
            for p in mine:
                if int(p[0]) == int(sid) or int(p[0]) not in threats: continue
                d = math.hypot(float(p[2])-float(sx), float(p[3])-float(sy))
                if best is None or d < best[0]:
                    best = (d, float(p[2]), float(p[3]), int(p[0]))
            if best is not None:
                _, tx, ty, tid = best
                ang = math.atan2(ty-float(sy), tx-float(sx))
                send = max(int(ships)//2, 1)
                moves.append([int(sid), float(ang), int(send)])
            continue
        # Otherwise attack nearest non-ally
        best = None
        for tgt in planets:
            tid, towner, tx, ty, tr, tships, _ = tgt
            if int(tid) == int(sid) or int(towner) == int(me): continue
            d = math.hypot(float(tx)-float(sx), float(ty)-float(sy))
            if best is None or d < best[0]:
                best = (d, float(tx), float(ty), int(tid), int(tships))
        if best is None: continue
        _, tx, ty, tid, tships = best
        ang = math.atan2(ty-float(sy), tx-float(sx))
        send = min(int(ships)-1, tships+2)
        if send > 0: moves.append([int(sid), float(ang), int(send)])
    return moves
''',
}

OPPONENT_PATHS = []
for fname, code in OPPONENT_CODES.items():
    p = OPPONENTS_DIR / fname
    p.write_text(code)
    OPPONENT_PATHS.append(str(p))
    print(f"  wrote {p}")

# Critical: include v4 self-play as the strongest opponent.
# Without this, v4 dominates the simple agents above and shot pos_rate skews
# toward 90%+. With pos_rate that high, training's pos_weight becomes tiny
# and the validator outputs uniformly low probabilities for every shot,
# so threshold=0.30 ends up filtering almost everything → topk1 sends nothing.
# v4-vs-v4 produces evenly matched games (~50% pos_rate) so the validator
# learns a meaningful decision boundary.
OPPONENT_PATHS.append(str(V4_LOCAL))
print(f"  added v4 self-play: {V4_LOCAL}")
```

      wrote /kaggle/working/opponents/nearest_sniper.py
      wrote /kaggle/working/opponents/weakest_first.py
      wrote /kaggle/working/opponents/production_first.py
      wrote /kaggle/working/opponents/defender.py
      added v4 self-play: /kaggle/working/obn_v4_exp004ish.py
    

## 3. Data collection (self-play)

We run v4 against 5 opponents × 6 seeds × 2 sides = **60 games**. Every move v4 issues is recorded as a *shot* with three pieces:

- **Source** = `(src_id, ang, ships)` — the move v4 picked
- **Features** = a 24-dim vector describing the board state at the moment of the shot (next section)
- **Label** = was this shot ultimately **successful**? Specifically: did v4 own the target planet within 10 turns of the projected fleet arrival?

So the validator learns a binary classifier: **"will this attack actually capture the target?"**

**Time budget**: roughly 5–10 minutes on Kaggle CPU.


```python
# Data collection
import math, time, multiprocessing as mp
import numpy as np
from kaggle_environments import make

BOARD = 100.0; MAX_SPEED = 6.0

def fleet_speed(ships):
    if ships <= 0: return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000.0)) ** 1.5

def find_target_via_ray(src_xy, send_angle, planets, ray_horizon=200.0, perp_margin=1.0):
    """Recover the (likely) target planet of a shot from its (src, angle).

    v4 emits actions as (src_id, angle, ships) — the target is implicit. We
    project a ray from src along `angle` and find the closest planet whose
    bounding circle the ray crosses.
    """
    sx, sy = src_xy; fx, fy = math.cos(send_angle), math.sin(send_angle)
    best_pid, best_perp = -1, 1e9
    for p in planets:
        pid, _, px, py, pr, _, _ = p
        pid = int(pid); px = float(px); py = float(py); pr = float(pr)
        dx = px - sx; dy = py - sy
        t = dx * fx + dy * fy
        if t <= 0 or t > ray_horizon: continue
        perp = abs(dx * fy - dy * fx)
        if perp <= pr + perp_margin and perp < best_perp:
            best_perp = perp; best_pid = pid
    return best_pid

def label_outcome(env_steps, target_id, side, arrival_turn, window=10):
    """Label = 1 iff `side` owns `target_id` at any turn in [arrival_turn, arrival_turn+window]."""
    end_t = min(arrival_turn + window, len(env_steps) - 1)
    start_t = min(arrival_turn, end_t)
    for t in range(start_t, end_t + 1):
        s = env_steps[t][side].observation
        if s is None: continue
        for p in s["planets"]:
            if int(p[0]) == target_id and int(p[1]) == side: return 1
    return 0

# 24-dim hand-crafted feature extractor for a single shot.
FEATURE_DIM = 24
def encode_shot(obs, src_id, target_id, ships_sent):
    pdict = {int(p[0]): p for p in obs["planets"]}
    if src_id not in pdict or target_id not in pdict: return None
    src = pdict[src_id]; tgt = pdict[target_id]
    me = int(obs.get("player", 0))
    fleets = obs.get("fleets", [])
    planets = obs["planets"]
    my_ships_total = sum(int(p[5]) for p in planets if int(p[1]) == me)
    enemy_ships_total = sum(int(p[5]) for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    my_planets = sum(1 for p in planets if int(p[1]) == me)
    enemy_planets = sum(1 for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    sx, sy, sr, sships = float(src[2]), float(src[3]), float(src[4]), int(src[5])
    tx, ty, tr, tships = float(tgt[2]), float(tgt[3]), float(tgt[4]), int(tgt[5])
    sprod, tprod = float(src[6]), float(tgt[6])
    dx, dy = tx - sx, ty - sy
    dist = max(math.hypot(dx, dy) - sr - tr, 0.0)
    speed = fleet_speed(ships_sent)
    eta = dist / max(speed, 0.5)
    own_self = 1.0 if int(tgt[1]) == me else 0.0
    own_neutral = 1.0 if int(tgt[1]) < 0 else 0.0
    own_enemy = 1.0 if (int(tgt[1]) >= 0 and int(tgt[1]) != me) else 0.0
    ship_frac = ships_sent / max(sships, 1)
    ally_n = sum(1 for f in fleets if int(f[1]) == me)
    ally_s = sum(int(f[6]) for f in fleets if int(f[1]) == me)
    enemy_n = sum(1 for f in fleets if int(f[1]) != me)
    enemy_s = sum(int(f[6]) for f in fleets if int(f[1]) != me)
    turn = int(obs.get("step", 0))
    return np.array([
        sships/100.0, sprod/5.0, sr/4.0,
        tships/100.0, tprod/5.0, tr/4.0,
        own_self, own_neutral, own_enemy,
        ships_sent/100.0, ship_frac,
        dist/BOARD, eta/60.0, speed/MAX_SPEED,
        ally_n/10.0, ally_s/100.0, enemy_n/10.0, enemy_s/100.0,
        turn/500.0, my_ships_total/200.0, enemy_ships_total/200.0,
        (my_ships_total - enemy_ships_total)/200.0,
        my_planets/20.0, enemy_planets/20.0,
    ], dtype=np.float32)

def collect_one_game(args):
    """Run one v4 vs opponent game and return (features, labels) for every v4 shot."""
    teacher_path, opponent_path, seed, side, game_id = args
    paths = [teacher_path, opponent_path] if side == 0 else [opponent_path, teacher_path]
    env = make("orbit_wars", configuration={"randomSeed": seed}, debug=False)
    try:
        env.run(paths)
    except Exception as e:
        return [], game_id, str(e)
    rows = []
    for step_idx, st in enumerate(env.steps):
        s = st[side]
        obs = s.observation
        action = s.action or []
        if obs is None or not action: continue
        planets = obs["planets"]
        src_xy = {int(p[0]): (float(p[2]), float(p[3])) for p in planets}
        for mv in action:
            try:
                src_id, ang, ships = int(mv[0]), float(mv[1]), int(mv[2])
            except Exception:
                continue
            if src_id not in src_xy: continue
            tgt_id = find_target_via_ray(src_xy[src_id], ang, planets)
            if tgt_id < 0 or tgt_id == src_id: continue
            tgt_owner = next((int(p[1]) for p in planets if int(p[0]) == tgt_id), -2)
            if tgt_owner == side: continue  # skip own-planet reinforcements (trivially "successful")
            feat = encode_shot(obs, src_id, tgt_id, ships)
            if feat is None: continue
            tx, ty, tr = next(((float(p[2]), float(p[3]), float(p[4])) for p in planets if int(p[0]) == tgt_id), (0,0,0))
            sx, sy = src_xy[src_id]
            sr = next((float(p[4]) for p in planets if int(p[0]) == src_id), 0)
            dist = max(math.hypot(tx-sx, ty-sy) - sr - tr, 0.0)
            speed = fleet_speed(ships)
            eta_turns = max(int(math.ceil(dist / max(speed, 0.5))), 1)
            arrival_turn = step_idx + eta_turns
            label = label_outcome(env.steps, tgt_id, side, arrival_turn, window=10)
            rows.append((feat, label, game_id, step_idx))
    return rows, game_id, None

# Build job list: 5 opponents × 6 seeds × 2 sides = 60 games
TEACHER = str(V4_LOCAL)
SEEDS = list(range(101, 111))  # 10 seeds (was 6 in v1)
SELFPLAY_SEEDS = list(range(101, 131))  # 30 self-play seeds (was 24 in v1)
jobs = []
gid = 0
for opp in OPPONENT_PATHS:
    for seed in (SELFPLAY_SEEDS if opp == TEACHER else SEEDS):
        for side in (0, 1):
            gid += 1
            jobs.append((TEACHER, opp, seed, side, gid))
print(f"Jobs: {len(jobs)} games (teacher=v4, {len(OPPONENT_PATHS)} opps, {len(SEEDS)} seeds × 2 sides)")
```

    Jobs: 140 games (teacher=v4, 5 opps, 10 seeds × 2 sides)
    


```python
# Run jobs in parallel (Kaggle notebooks have ~4 CPU cores)
all_rows = []
failed = 0
t0 = time.time()
with mp.Pool(processes=4) as pool:
    for i, (rows, gid_, err) in enumerate(pool.imap_unordered(collect_one_game, jobs)):
        if err is not None:
            failed += 1
            print(f"  [WARN] game {gid_} failed: {err[:80]}")
        else:
            all_rows.extend(rows)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(jobs)} games, rows={len(all_rows)}, t={time.time()-t0:.0f}s", flush=True)

print(f"\nDone: {len(all_rows)} shots collected ({failed} games failed)")
feats = np.stack([r[0] for r in all_rows]).astype(np.float32)
labels = np.asarray([r[1] for r in all_rows], dtype=np.float32)
meta_game = np.asarray([r[2] for r in all_rows], dtype=np.int32)
pos_rate = labels.mean()
print(f"  features: {feats.shape}, labels: {labels.shape}")
print(f"  positive rate: {pos_rate*100:.1f}%")

# Diagnostic: pos_rate is the most predictive number for whether the validator
# will produce a useful threshold. Healthy range is 50-75%. Above 85% means
# opponents are too weak (v4 dominates), training pos_weight will be too small,
# the validator will under-confidence almost every shot, and topk1 will end up
# sending almost nothing on the LB. If you see pos_rate > 85%, add stronger
# opponents or increase the v4-self-play seed count.
if pos_rate > 0.85:
    print(f"  WARNING: pos_rate is high ({pos_rate*100:.1f}%). Validator may be over-cautious.")
    print(f"           Consider stronger opponents or more v4 self-play games.")
elif pos_rate < 0.40:
    print(f"  WARNING: pos_rate is low ({pos_rate*100:.1f}%). Opponents may be too strong.")

# Save dataset for inspection
np.savez_compressed(WORK / "shot_dataset.npz", features=feats, labels=labels.astype(np.int64), meta_game=meta_game)
print(f"  saved {WORK}/shot_dataset.npz")
```

      5/140 games, rows=352, t=48s
      10/140 games, rows=729, t=122s
      15/140 games, rows=1030, t=295s
      20/140 games, rows=1468, t=373s
      25/140 games, rows=2002, t=454s
      30/140 games, rows=2535, t=532s
      35/140 games, rows=2948, t=595s
      40/140 games, rows=3449, t=689s
      45/140 games, rows=3944, t=822s
      50/140 games, rows=4376, t=876s
      55/140 games, rows=4883, t=920s
      60/140 games, rows=5378, t=953s
      65/140 games, rows=5741, t=1045s
      70/140 games, rows=6233, t=1329s
      75/140 games, rows=6604, t=1410s
      80/140 games, rows=7195, t=1611s
      85/140 games, rows=7981, t=1866s
      90/140 games, rows=8866, t=2063s
      95/140 games, rows=9768, t=2269s
      100/140 games, rows=10793, t=2455s
      105/140 games, rows=11759, t=2573s
      110/140 games, rows=12666, t=2779s
      115/140 games, rows=13730, t=3042s
      120/140 games, rows=14871, t=3198s
      125/140 games, rows=16113, t=3517s
      130/140 games, rows=17134, t=3665s
      135/140 games, rows=18148, t=3943s
      140/140 games, rows=19192, t=4106s
    
    Done: 19192 shots collected (0 games failed)
      features: (19192, 24), labels: (19192,)
      positive rate: 73.1%
      saved /kaggle/working/shot_dataset.npz
    

## 4. Train the ML shot validator

**Model**: a tiny 3-layer MLP (24 → 64 → 32 → 1), about 3,500 parameters total. Small enough to inference in ~1 ms on CPU during the Kaggle 1-second-per-turn match.

**Training settings**:
- Loss: `BCEWithLogitsLoss` with `pos_weight = (1 - pos_rate) / pos_rate` to correct class imbalance
- Optimizer: Adam, lr=1e-3
- Epochs: 40
- **Validation**: 20% holdout split **at the game level** — random row-level split would leak, since shots within the same game are temporally correlated.

Runs in ~30 seconds on CPU; ~5 seconds on GPU. (We auto-fall-back to CPU if CUDA fails, since the model is tiny anyway.)


```python
import torch
import torch.nn as nn

class ShotValidator(nn.Module):
    def __init__(self, in_dim=24, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# Try CUDA, fall back to CPU on any compatibility issue.
# This MLP is tiny (3.5k params, 10k rows) so CPU is fine — full training
# completes in ~30 seconds either way.
def _select_device():
    if not torch.cuda.is_available():
        return torch.device("cpu"), "cuda not available"
    try:
        # Sanity-check that CUDA actually works on this device
        _ = torch.zeros(4, device="cuda") + 1
        torch.cuda.synchronize()
        return torch.device("cuda"), "cuda OK"
    except Exception as e:
        return torch.device("cpu"), f"cuda failed ({type(e).__name__}: {str(e)[:60]}), using CPU"

device, why = _select_device()
print(f"Training on: {device} ({why})")

# Game-level val split (random row split would leak — shots within a game are correlated)
rng = np.random.default_rng(42)
games = np.unique(meta_game)
rng.shuffle(games)
n_val = max(1, int(len(games) * 0.2))
val_games = set(games[:n_val].tolist())
val_mask = np.array([g in val_games for g in meta_game], dtype=bool)
Xt, yt = feats[~val_mask], labels[~val_mask]
Xv, yv = feats[val_mask], labels[val_mask]
print(f"  train: {len(Xt)} shots ({len(games)-n_val} games), val: {len(Xv)} shots ({n_val} games)")
print(f"  train pos: {yt.mean()*100:.1f}%, val pos: {yv.mean()*100:.1f}%")

# Class-balanced pos_weight
pr = max(yt.mean(), 1e-6)
pos_weight = torch.tensor([(1.0 - pr) / pr], device=device)
print(f"  pos_weight (neg/pos): {pos_weight.item():.3f}")
```

    Training on: cpu (cuda not available)
      train: 15385 shots (112 games), val: 3807 shots (28 games)
      train pos: 72.9%, val pos: 73.9%
      pos_weight (neg/pos): 0.371
    


```python
# Training loop
model = ShotValidator(in_dim=24, hidden=64).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

Xt_t = torch.from_numpy(Xt).to(device)
yt_t = torch.from_numpy(yt).to(device)
Xv_t = torch.from_numpy(Xv).to(device)
yv_t = torch.from_numpy(yv).to(device)

EPOCHS = 40
BATCH = 512

best_acc03 = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = torch.randperm(len(Xt_t), device=device)
    losses = []
    for i in range(0, len(idx), BATCH):
        b = idx[i:i+BATCH]
        logits = model(Xt_t[b])
        loss = crit(logits, yt_t[b])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    tl = float(np.mean(losses))
    model.eval()
    with torch.no_grad():
        vlogits = model(Xv_t)
        vloss = float(crit(vlogits, yv_t))
        vprob = torch.sigmoid(vlogits)
        v_acc05 = float(((vprob > 0.5).float() == yv_t).float().mean())
        v_acc03 = float(((vprob > 0.3).float() == yv_t).float().mean())
    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"epoch {epoch:3d} | t_loss={tl:.4f} v_loss={vloss:.4f} "
              f"v_acc(0.5)={v_acc05:.3f} v_acc(0.3)={v_acc03:.3f}")

print("\nTraining complete.")
```

    /tmp/ipykernel_16/923667046.py:24: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
    Consider using tensor.detach() first. (Triggered internally at /pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:836.)
      losses.append(float(loss))
    

    epoch   1 | t_loss=0.3557 v_loss=0.3221 v_acc(0.5)=0.701 v_acc(0.3)=0.769
    epoch   5 | t_loss=0.2987 v_loss=0.2951 v_acc(0.5)=0.735 v_acc(0.3)=0.775
    epoch  10 | t_loss=0.2891 v_loss=0.2846 v_acc(0.5)=0.755 v_acc(0.3)=0.788
    epoch  15 | t_loss=0.2797 v_loss=0.2789 v_acc(0.5)=0.749 v_acc(0.3)=0.794
    epoch  20 | t_loss=0.2813 v_loss=0.2717 v_acc(0.5)=0.750 v_acc(0.3)=0.795
    epoch  25 | t_loss=0.2756 v_loss=0.2714 v_acc(0.5)=0.749 v_acc(0.3)=0.796
    epoch  30 | t_loss=0.2681 v_loss=0.2700 v_acc(0.5)=0.763 v_acc(0.3)=0.804
    epoch  35 | t_loss=0.2644 v_loss=0.2670 v_acc(0.5)=0.750 v_acc(0.3)=0.804
    epoch  40 | t_loss=0.2655 v_loss=0.2685 v_acc(0.5)=0.744 v_acc(0.3)=0.799
    
    Training complete.
    

## 5. Export weights to NumPy

The Kaggle submission environment does not include PyTorch. We convert the trained weights to a NumPy `.npz` file and inference becomes a simple chain of matmuls + ReLU + sigmoid (1 ms or less per turn).


```python
model.eval()
sd = model.state_dict()
weights = {
    "l0_w": sd["net.0.weight"].cpu().numpy().astype(np.float32),
    "l0_b": sd["net.0.bias"].cpu().numpy().astype(np.float32),
    "l1_w": sd["net.2.weight"].cpu().numpy().astype(np.float32),
    "l1_b": sd["net.2.bias"].cpu().numpy().astype(np.float32),
    "l2_w": sd["net.4.weight"].cpu().numpy().astype(np.float32),
    "l2_b": sd["net.4.bias"].cpu().numpy().astype(np.float32),
}
for k, v in weights.items():
    print(f"  {k}: {v.shape}, {v.dtype}")

weights_path = WORK / "weights.npz"
np.savez(weights_path, **weights)
print(f"\nSaved {weights_path} ({weights_path.stat().st_size:,} bytes)")

# Sanity check: parity between PyTorch and numpy forward
def numpy_forward(x, W):
    h = np.maximum(0, x @ W["l0_w"].T + W["l0_b"])
    h = np.maximum(0, h @ W["l1_w"].T + W["l1_b"])
    logit = h @ W["l2_w"].T + W["l2_b"]
    return 1.0 / (1.0 + np.exp(-logit))

x_test = Xv[:5]
with torch.no_grad():
    p_torch = torch.sigmoid(model(torch.from_numpy(x_test).to(device))).cpu().numpy()
p_numpy = numpy_forward(x_test, weights).squeeze()
print(f"\nParity check (PyTorch vs numpy):")
print(f"  torch: {p_torch[:5]}")
print(f"  numpy: {p_numpy[:5]}")
print(f"  max diff: {np.abs(p_torch - p_numpy).max():.6e}")
```

      l0_w: (64, 24), float32
      l0_b: (64,), float32
      l1_w: (32, 64), float32
      l1_b: (32,), float32
      l2_w: (1, 32), float32
      l2_b: (1,), float32
    
    Saved /kaggle/working/weights.npz (16,314 bytes)
    
    Parity check (PyTorch vs numpy):
      torch: [0.49568418 0.8834993  0.5709377  0.60510427 0.5349362 ]
      numpy: [0.49568424 0.8834993  0.5709377  0.6051042  0.5349362 ]
      max diff: 5.960464e-08
    

## 6. Assemble `main.py`

We splice three components into a single submission file:

1. The full v4 source (3,378 lines), with its `agent` function renamed to `_v4_agent_internal` so we can wrap it
2. A `_NumpyValidator`-style forward function (24 → 64 → 32 → 1 with ReLU + sigmoid)
3. A new `agent(obs, config)` that:
   - calls `_v4_agent_internal` to get v4's full proposal list
   - sends each move's features through the validator and **drops moves with P < 0.30**
   - keeps the **single largest-ship-count move** that survives (= topk1)

### Why topk1 (and not topk2 / topk2_src)?

A local sweep over throttle variants — same v3 data + 0.30 threshold, evaluated against 8 LB-tier opponents × 4 seeds × 2 sides = 64 games each:

| variant | overall wr | vs LB-tier (56g) |
|---|---|---|
| **topk1** | **51.6%** ⭐ | **52%** |
| topk2 | 42.2% | 41% |
| topk2_src (distinct sources) | 39.1% | 38% |
| topk2_cond (2nd if strong) | 37.5% | 39% |

topk1 wins by **+9 pp** locally on this validator. The intuition: with a 24-dim feature set and a tiny MLP trained only on simple-opponent self-play data, the validator's discrimination is limited (val_acc ≈ 0.76, pos_rate ≈ 0.74). Sending fewer moves per turn (topk1) tolerates that imprecision better than letting through marginally validated shots (topk2+).

> A larger validator trained on real LB-1450+ replay shots scores higher with topk2 (LB ~1016 vs ~967 in our internal experiments). If you build that stronger validator, switch back to topk2 — but with *this* notebook's self-contained pipeline, topk1 is the better default.


```python
# Read v4 source
v4_source = V4_LOCAL.read_text()
print(f"v4 source: {len(v4_source.splitlines()):,} lines, {len(v4_source):,} chars")

# Hybrid wrapper (renames v4's `agent` to `_v4_agent_internal` so we can wrap it)
import re
v4_renamed = re.sub(
    r"^def agent\(obs, config=None\):",
    "def _v4_agent_internal(obs, config=None):",
    v4_source, count=1, flags=re.MULTILINE,
)
assert "_v4_agent_internal" in v4_renamed, "failed to rename v4 agent"

# Hybrid wrapper code (validator + topk1)
HYBRID_CODE = '''
# ============================================================
# ML Shot Validator (numpy) + topk1 throttle (v3: reverted from topk2)
# ============================================================
import os as _os_h
from pathlib import Path as _Path_h
import numpy as _np_h
import math as _math_h

_VAL_THRESHOLD = 0.30  # the "sweet spot" found via threshold sweep

def _find_weights():
    cands = [
        _Path_h("/kaggle_simulations/agent/weights.npz"),
        _Path_h.cwd() / "weights.npz",
        _Path_h("weights.npz"),
    ]
    try:
        cands.insert(0, _Path_h(__file__).resolve().parent / "weights.npz")
    except NameError:
        pass
    for p in cands:
        if p.exists(): return p
    return None

_W_PATH = _find_weights()
_W = _np_h.load(_W_PATH) if _W_PATH is not None else None

def _validator_proba(x):
    if _W is None: return None
    h = _np_h.maximum(0, x @ _W["l0_w"].T + _W["l0_b"])
    h = _np_h.maximum(0, h @ _W["l1_w"].T + _W["l1_b"])
    logit = h @ _W["l2_w"].T + _W["l2_b"]
    return 1.0 / (1.0 + _np_h.exp(-logit))

_BOARD_H = 100.0; _MAX_SPEED_H = 6.0

def _fleet_speed_h(ships):
    if ships <= 0: return 1.0
    return 1.0 + (_MAX_SPEED_H - 1.0) * (_math_h.log(max(ships, 1)) / _math_h.log(1000.0)) ** 1.5

def _find_target_ray_h(src_xy, ang, planets, ray_horizon=200.0, perp_margin=1.0):
    sx, sy = src_xy; fx, fy = _math_h.cos(ang), _math_h.sin(ang)
    best_pid, best_perp = -1, 1e9
    for p in planets:
        pid, _, px, py, pr, _, _ = p
        pid = int(pid); px = float(px); py = float(py); pr = float(pr)
        dx = px - sx; dy = py - sy
        t = dx * fx + dy * fy
        if t <= 0 or t > ray_horizon: continue
        perp = abs(dx * fy - dy * fx)
        if perp <= pr + perp_margin and perp < best_perp:
            best_perp = perp; best_pid = pid
    return best_pid

def _encode_shot_h(obs, src_id, target_id, ships_sent):
    pdict = {int(p[0]): p for p in obs["planets"]}
    if src_id not in pdict or target_id not in pdict: return None
    src = pdict[src_id]; tgt = pdict[target_id]
    me = int(obs.get("player", 0))
    fleets = obs.get("fleets", [])
    planets = obs["planets"]
    my_t = sum(int(p[5]) for p in planets if int(p[1]) == me)
    en_t = sum(int(p[5]) for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    my_p = sum(1 for p in planets if int(p[1]) == me)
    en_p = sum(1 for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    sx, sy, sr, sships = float(src[2]), float(src[3]), float(src[4]), int(src[5])
    tx, ty, tr, tships = float(tgt[2]), float(tgt[3]), float(tgt[4]), int(tgt[5])
    sprod, tprod = float(src[6]), float(tgt[6])
    dx, dy = tx - sx, ty - sy
    dist = max(_math_h.hypot(dx, dy) - sr - tr, 0.0)
    speed = _fleet_speed_h(ships_sent); eta = dist / max(speed, 0.5)
    own_self = 1.0 if int(tgt[1]) == me else 0.0
    own_neutral = 1.0 if int(tgt[1]) < 0 else 0.0
    own_enemy = 1.0 if (int(tgt[1]) >= 0 and int(tgt[1]) != me) else 0.0
    sf = ships_sent / max(sships, 1)
    an = sum(1 for f in fleets if int(f[1]) == me)
    a_s = sum(int(f[6]) for f in fleets if int(f[1]) == me)
    en = sum(1 for f in fleets if int(f[1]) != me)
    e_s = sum(int(f[6]) for f in fleets if int(f[1]) != me)
    turn = int(obs.get("step", 0))
    return _np_h.array([
        sships/100.0, sprod/5.0, sr/4.0,
        tships/100.0, tprod/5.0, tr/4.0,
        own_self, own_neutral, own_enemy,
        ships_sent/100.0, sf,
        dist/_BOARD_H, eta/60.0, speed/_MAX_SPEED_H,
        an/10.0, a_s/100.0, en/10.0, e_s/100.0,
        turn/500.0, my_t/200.0, en_t/200.0,
        (my_t - en_t)/200.0,
        my_p/20.0, en_p/20.0,
    ], dtype=_np_h.float32)

def agent(obs, config=None):
    moves = _v4_agent_internal(obs, config)
    if not moves or _W is None:
        # If validator missing or v4 didn't propose anything, return raw v4 output
        return moves
    side = int(obs.get("player", 0))
    planets = obs["planets"]
    src_xy = {int(p[0]): (float(p[2]), float(p[3])) for p in planets}
    owner_by_id = {int(p[0]): int(p[1]) for p in planets}
    feats, idxs = [], []
    for i, mv in enumerate(moves):
        try:
            sid, ang, ships = int(mv[0]), float(mv[1]), int(mv[2])
        except Exception:
            continue
        if sid not in src_xy: continue
        tid = _find_target_ray_h(src_xy[sid], ang, planets)
        if tid < 0 or tid == sid: continue
        if owner_by_id.get(tid, -2) == side:
            continue  # never filter own-planet reinforcements
        f = _encode_shot_h(obs, sid, tid, ships)
        if f is None: continue
        feats.append(f); idxs.append(i)
    if feats:
        x = _np_h.stack(feats)
        probs = _validator_proba(x).squeeze(-1)
        keep = [True] * len(moves)
        for j, p in zip(idxs, probs):
            if p < _VAL_THRESHOLD:
                keep[j] = False
        moves = [m for k, m in enumerate(moves) if keep[k]]
    if not moves: return []
    # topk1: keep only the largest-ship-count move (best for self-trained validator)
    best = max(moves, key=lambda m: int(m[2]))
    return [best]
'''

# Concatenate v4 (renamed) + hybrid wrapper
main_py = v4_renamed + "\n\n" + HYBRID_CODE

# Write to /kaggle/working/main.py — Kaggle picks up `agent` automatically
(WORK / "main.py").write_text(main_py)
print(f"main.py: {(WORK/'main.py').stat().st_size:,} bytes ({len(main_py.splitlines()):,} lines)")
```

    v4 source: 3,378 lines, 113,252 chars
    main.py: 118,375 bytes (3,508 lines)
    

## 7. Sanity check

Run a single game with the freshly-built `main.py` to make sure it loads, runs, and finishes without errors.


```python
# Sanity test: run a 1-game match to confirm the agent works
from kaggle_environments import make

env = make("orbit_wars", debug=True)
result = env.run([str(WORK / "main.py"), OPPONENT_PATHS[0]])  # vs nearest_sniper
final = env.steps[-1]
print(f"Game finished in {len(env.steps)} steps:")
for i, s in enumerate(final):
    print(f"  P{i}: reward={s.reward}, status={s.status}")
```

    Game finished in 175 steps:
      P0: reward=1, status=DONE
      P1: reward=-1, status=DONE
    

## 8. Build `submission.tar.gz`

Bundle `main.py` + `weights.npz` into the tar.gz format that Kaggle expects.


```python
import tarfile

tar_path = WORK / "submission.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(WORK / "main.py", arcname="main.py")
    tar.add(WORK / "weights.npz", arcname="weights.npz")

size = tar_path.stat().st_size
print(f"\n✅ Submission ready: {tar_path} ({size:,} bytes = {size/1024:.1f} KB)")
with tarfile.open(tar_path) as tar:
    for m in tar.getmembers():
        print(f"  {m.name}: {m.size:,} bytes")
```

    
    ✅ Submission ready: /kaggle/working/submission.tar.gz (37,668 bytes = 36.8 KB)
      main.py: 118,375 bytes
      weights.npz: 16,314 bytes
    

## 9. How to submit

Two options:

### A. Submit this notebook directly (recommended)

1. Click **Save Version** at the top right of the notebook → "Save & Run All (Commit)"
2. From the notebook page, click **Submit to Competition** on the right → choose `submission.tar.gz`

Kaggle automatically picks up `/kaggle/working/submission.tar.gz`.

### B. Download and submit via CLI

1. Download `submission.tar.gz` from the notebook output
2. Locally:
   ```bash
   kaggle competitions submit orbit-wars -f submission.tar.gz -m "v4 + ML validator (t=0.30) + topk1"
   ```

## 10. Expected score

- **LB ~970 ballpark** with v3 (topk1 + the v2 data improvements). The previous v2 attempt with topk2 scored 841.8; the original v1 (no defender opp, 6 seeds) was 926.5. v3 should land **above v1** thanks to the additional training data, while reverting the throttle mistake from v2.
- The opponent pool here is a self-contained set of **simple** inline agents, weaker than the original `exp_arch_001` pool. With stronger external opponents (e.g. `orbit-botnext`), scores tend to be higher.
- To push higher, try:
  - **Threshold sweep**: 0.25 / 0.30 / 0.35 around the current default. Looser (0.25) keeps more shots; tighter (0.35) drops more — pick by validation accuracy or local win rate.
  - **Stronger opponents in self-play**: include the public `orbit-botnext` notebook by Pascal Ledesma — adds noticeable LB pts in our experiments.
  - **More data**: scale to 200+ games (vs current ~140).
  - **Reconsider topk2**: only after you have a stronger validator (e.g. trained on real LB-≥1450 player replays). With this notebook's simple-opponent training, topk1 is better — see section 6 sweep.

## 11. Improvement ideas

1. **Train against stronger opponents** — the agent's known weakness is matchups vs LB-≥1000 opponents (~20% win rate). Including stronger agents in the data-collection pool should help directly.
2. **Richer features** — add comet info, planet orbital velocity, and rotation phase to the 24-dim feature vector.
3. **Validator ensemble** — train one MLP on the v1 dataset and one on a hard-negatives augmented dataset, accept a shot only if both approve.
4. **Dynamic threshold** — vary `_VAL_THRESHOLD` by game phase (stricter early, looser late, or vice versa).

## Final acknowledgments

This notebook stands on the shoulders of the public Kaggle community. Particular thanks to:
- **Pascal Ledesma** for [orbit-botnext](https://www.kaggle.com/code/pascalledesma/orbit-botnext) and [orbitwork-v14](https://www.kaggle.com/code/pascalledesma/orbitwork-v14)
- **djenkivanov** for [orbit-wars-agent-ow-proto](https://www.kaggle.com/code/djenkivanov/orbit-wars-agent-ow-proto)
- The authors of the LB-1224 / LB-1100 public notebooks whose constants we incorporated

If this notebook helps you, please give an upvote to **their** notebooks too — without their work this hybrid would not exist.
