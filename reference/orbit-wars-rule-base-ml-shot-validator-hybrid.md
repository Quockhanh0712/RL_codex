# 🛰️ Orbit Wars — Rule-base × ML Shot Validator Hybrid

This notebook ships a **hybrid agent** that pairs a strong public rule-based agent (the
Tamrazov × Ykhnkf line, descended from `pilkwang/structured-baseline`) with a small
**numpy-only "Shot Validator" MLP** that filters out attacks the rule-base would otherwise
make but which an ML model has learned tend to fail.

The validator is intentionally **conservative**: it only ever *rejects* shots, never proposes
new ones. Worst case it does nothing and the agent is identical to the rule-base. Best case
it removes wasteful attacks against well-defended targets and lets the agent conserve ships
for the next opening.

In local 2P play (8 seeds × 5 opponents × 2 sides = 80 games per side), the hybrid wins
**84%** vs **65%** for the rule-base alone — a **+19pp** swing driven mostly by the harder
opponent classes (tier3 +25pp, tier4 +43pp).


## 1. Why a Hybrid?

A pure ML approach (PPO from scratch, SFT distillation from teachers, multi-teacher SFT)
keeps hitting the same ceiling against the strongest public rule-based agents — win rate
against tier3+ opponents collapses to ~0% even after 1000+ PPO updates. Five separate ML
attempts ran into the same wall.

The mechanism is the usual sparse-reward trap: the network only gets a +1 / -1 signal at
end-of-game, against a strong opponent it loses almost every rollout, and the gradient
ends up pushing the policy toward defensive no-ops rather than discovering tier3-beating
strategies.

A pure rule-base approach (the path our parent notebook took) has the opposite ceiling:
it plays a coherent strategy out of the box, but every constant-tweak we tried either
helped against one opponent and hurt another, or had no measurable effect.

This notebook takes the **third path**:

```
[obs] ──► rule-base (v4 lineage) ──► candidate moves
                                     │
                                     ▼
                              ML shot validator ──► drop low-P(success) shots
                                     │
                                     ▼
                                 final action
```

The rule-base does the heavy strategic lifting. The ML model only votes *no* on individual
shots. Because the rule-base's coherent strategy is preserved, the ML doesn't need to
discover anything from scratch — it just needs to learn one local question:

> *Given this source planet, this target, and this fleet size — does this shot usually
> end with us owning the target 10 turns later?*

That question has a clean per-shot binary label, dense across every game (~180 shots / game),
and the wrong answer just leaves a v4-equivalent action in place.


## 2. Shot Validator design

### Inputs (24-dim float32)

For every shot the rule-base proposes, we encode:

| group | features |
| --- | --- |
| **source planet** | ships, production, radius |
| **target planet** | ships, production, radius, owner one-hot (mine / neutral / enemy) |
| **shot** | ships sent, ship fraction (sent / source ships), distance, ETA in turns, computed fleet speed |
| **in-flight** | count + ship total of allied & enemy fleets |
| **meta** | turn number, my total ships, enemy total ships, ship diff, my planet count, enemy planet count |

All scalars are normalised to roughly [0, 1] so the MLP doesn't have to learn per-feature
scales.

### Label

For each shot, walk forward in the played-out game from the expected arrival turn `t` for
`t+10` turns. Label is `1` iff the target planet's owner is *us* on any of those turns,
else `0`.

Crucially, **shots that reinforce our own planets are excluded from the dataset entirely**.
Self-reinforcement is trivially "successful" (we already own the target) and would dilute
the signal — without filtering, the positive rate is ~96%. After filtering, the positive
rate drops to **70.8%**, which leaves real negative signal for the model to learn from.

### Model

A tiny three-layer MLP, ~5k parameters total:

```
input (24) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) → sigmoid → P(success)
```

Trained with `BCEWithLogitsLoss(pos_weight = neg/pos)` on 8.8k training shots from
games against five different opponents, validated by *game id* (not row) to prevent
leakage between train and val.

After 40 epochs:

- val accuracy at threshold 0.5: **76.8%**
- val accuracy at threshold 0.3: **80.8%**
- mean P(positive) given true positive: **0.68**
- mean P(positive) given true negative: **0.38**

The 0.30 separation is small in absolute terms, but it doesn't need to be large — every
correctly rejected wasteful shot saves ships, and every incorrect rejection just leaves a
v4 default in place.

### Inference: threshold gate

At inference time we rebuild the same 24-dim feature for every shot the rule-base wants to
take, run the MLP, and **drop** any shot whose predicted P(success) is below a threshold.
Self-reinforcement always passes through.

We swept four thresholds:

| threshold | local win rate |
|---|---|
| 0.2 (lenient) | 76% |
| 0.3 | 78% |
| **0.4** | **84%** ⭐ |
| 0.5 (strict) | 57% (over-rejects) |

0.4 is the sweet spot — strict enough to remove the bad tail, lenient enough not to reject
the merely uncertain.


## 3. Validation findings

### 3.1 Per-opponent win rate (8 seeds × 2 sides = 16 games / cell)

| opponent | hybrid (t=0.4) | rule-base only | Δ |
|---|---|---|---|
| `v1_sniper` | 16/16 (100%) | 16/16 (100%) | 0 |
| `v2_structured` | 13/16 (81%) | 12/16 (75%) | +6pp |
| `exp007_tier3` | **13/16 (81%)** | 9/16 (56%) | **+25pp** |
| `exp007_tier4` | **9/16 (56%)** → **13/16 (81%)** | 6/16 (38%) | **+43pp** |
| `orbitbotnext` | 11/16 (69%) → 12/16 (75%) | 9/16 (56%) | +13–19pp |
| **overall** | **67/80 (84%)** | **52/80 (65%)** | **+19pp** |

The pattern is the one the design predicted: against weak opponents (sniper) there's
nothing for the validator to do, both agents win comfortably. Against the strong opponents
where v4 alone struggles (tier4 38%), the validator's ship conservation lets the agent
trade more efficiently and the win rate roughly doubles.

### 3.2 What changes turn-by-turn

Across an average game, the validator drops on the order of **3–10%** of the rule-base's
shots. The dropped shots cluster around two patterns:

- **Late-game over-extension**: trying to capture a target that the model thinks will be
  re-taken before our reinforcement arrives.
- **Defended-target attacks with marginal ship counts**: shots where ship_fraction is high
  enough that the source becomes vulnerable, but the model has low confidence the target
  will actually fall.

In both cases the rejection conserves ships for the next turn's rule-base decision, and
the rule-base picks a better target with the saved garrison.

### 3.3 No regression vs the rule-base

Every opponent class shows hybrid ≥ rule-base. There is no opponent for which adding the
validator hurts. This is the design's core safety property: rejection-only overrides cannot
introduce a worse action than the rule-base's own choice — they can only fail to improve it.


## 4. Build `weights.npz`

The trained MLP weights are tiny (~15 KB) so we embed them as base64 here and decode
back into a `weights.npz` file next to `submission.py` at submission time.



```python
%%writefile decode_weights.py
# auto-generated: decodes the embedded base64 MLP weights into weights.npz
import base64, pathlib

WEIGHTS_B64 = (
'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'
'AC46AAAAAA=='
)

pathlib.Path("weights.npz").write_bytes(base64.b64decode("".join(WEIGHTS_B64)))
print("wrote weights.npz", pathlib.Path("weights.npz").stat().st_size, "bytes")

```

    Writing decode_weights.py
    


```python
!python decode_weights.py

```

    wrote weights.npz 15340 bytes


## 5. The agent

A single self-contained `submission.py`. Everything is pure stdlib + numpy — no torch,
no extra dependencies. The rule-base body (~3300 lines) is inlined verbatim and its entry
function is renamed to `_v4_agent_internal`; our `agent(obs, config)` wraps it and applies
the validator.

Layout of the file:

1. Imports + the numpy `_NumpyValidator` class (forward only)
2. `_encode_shot_np(...)` — 24-dim feature builder
3. `_find_target_ray(...)` — rebuild target id from `(src, angle)` via ray projection
4. The full inlined rule-base agent (renamed to `_v4_agent_internal`)
5. `agent(obs, config)` — calls `_v4_agent_internal`, then drops shots below threshold



```python
%%writefile submission.py
"""Hybrid agent: v4 (obn_v4_exp004ish) + Shot Validator override.

build_submission.py で auto-generated。手動編集しないこと。
"""
import math as _math_hybrid
import os as _os_hybrid
from pathlib import Path as _Path_hybrid
import numpy as _np_hybrid

# ---- Shot Validator (numpy) ----
def _find_weights_path():
    candidates = [
        _Path_hybrid("/kaggle_simulations/agent/weights.npz"),
        _Path_hybrid.cwd() / "weights.npz",
        _Path_hybrid("weights.npz"),
    ]
    try:
        candidates.insert(0, _Path_hybrid(__file__).resolve().parent / "weights.npz")
    except NameError:
        pass
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

_WEIGHTS_PATH = _find_weights_path()
_VAL_THRESHOLD = 0.4000

class _NumpyValidator:
    def __init__(self, w_path):
        npz = _np_hybrid.load(str(w_path))
        self.w0 = npz["w0"]; self.b0 = npz["b0"]
        self.w2 = npz["w2"]; self.b2 = npz["b2"]
        self.w4 = npz["w4"]; self.b4 = npz["b4"]
    def forward(self, x):
        # x: (B, in_dim)
        h = _np_hybrid.maximum(0.0, x @ self.w0.T + self.b0)
        h = _np_hybrid.maximum(0.0, h @ self.w2.T + self.b2)
        return (h @ self.w4.T + self.b4).reshape(-1)
    def proba(self, x):
        z = self.forward(x)
        return 1.0 / (1.0 + _np_hybrid.exp(-z))

try:
    _VALIDATOR = _NumpyValidator(_WEIGHTS_PATH) if _WEIGHTS_PATH.exists() else None
except Exception as _e:
    _VALIDATOR = None

_FEATURE_DIM = 24

def _encode_shot_np(obs, src_id, target_id, ships_sent):
    BOARD = 100.0; MAX_SPEED = 6.0
    pdict = {}
    for p in obs["planets"]:
        pid = int(p[0])
        pdict[pid] = (int(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[5]), float(p[6]))
    if src_id not in pdict or target_id not in pdict:
        return None
    src = pdict[src_id]; tgt = pdict[target_id]
    me = int(obs.get("player", 0))
    fleets = obs.get("fleets", [])
    planets = obs["planets"]
    my_ships_total = sum(int(p[5]) for p in planets if int(p[1]) == me)
    enemy_ships_total = sum(int(p[5]) for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    my_planets = sum(1 for p in planets if int(p[1]) == me)
    enemy_planets = sum(1 for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    src_owner, sx, sy, sr, ss, sp = src
    tgt_owner, tx, ty, tr, ts, tp = tgt
    dx = tx - sx; dy = ty - sy
    dist = max(_math_hybrid.hypot(dx, dy) - sr - tr, 0.0)
    if ships_sent <= 0:
        speed = 1.0
    else:
        speed = 1.0 + (MAX_SPEED - 1.0) * (_math_hybrid.log(max(ships_sent, 1)) / _math_hybrid.log(1000.0)) ** 1.5
    eta = dist / max(speed, 0.5)
    own_self = 1.0 if tgt_owner == me else 0.0
    own_neutral = 1.0 if tgt_owner < 0 else 0.0
    own_enemy = 1.0 if (tgt_owner >= 0 and tgt_owner != me) else 0.0
    ship_frac = ships_sent / max(ss, 1)
    ally_n = 0; ally_s = 0; enemy_n = 0; enemy_s = 0
    for f in fleets:
        owner = int(f[1]); shp = int(f[6])
        if owner == me:
            ally_n += 1; ally_s += shp
        else:
            enemy_n += 1; enemy_s += shp
    turn = int(obs.get("step", 0))
    feat = _np_hybrid.array([
        ss / 100.0, sp / 5.0, sr / 4.0,
        ts / 100.0, tp / 5.0, tr / 4.0,
        own_self, own_neutral, own_enemy,
        ships_sent / 100.0, ship_frac,
        dist / BOARD, eta / 60.0, speed / MAX_SPEED,
        ally_n / 10.0, ally_s / 100.0,
        enemy_n / 10.0, enemy_s / 100.0,
        turn / 500.0,
        my_ships_total / 200.0, enemy_ships_total / 200.0,
        (my_ships_total - enemy_ships_total) / 200.0,
        my_planets / 20.0, enemy_planets / 20.0,
    ], dtype=_np_hybrid.float32)
    return feat

def _find_target_ray(src_xy, send_angle, planets, ray_horizon=200.0, perp_margin=1.0):
    sx, sy = src_xy
    fx = _math_hybrid.cos(send_angle); fy = _math_hybrid.sin(send_angle)
    best_pid = -1; best_perp = 1e9
    for p in planets:
        pid = int(p[0]); px = float(p[2]); py = float(p[3]); pr = float(p[4])
        dx = px - sx; dy = py - sy
        t = dx * fx + dy * fy
        if t <= 0 or t > ray_horizon:
            continue
        perp = abs(dx * fy - dy * fx)
        if perp <= pr + perp_margin and perp < best_perp:
            best_perp = perp; best_pid = pid
    return best_pid

# ---- v4 source (inlined below) ----

import math
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field

# ============================================================
# Shared Configuration
# ============================================================

BOARD = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
SUN_SAFETY = 1.5
ROTATION_LIMIT = 50.0
TOTAL_STEPS = 500
SIM_HORIZON = 110
ROUTE_SEARCH_HORIZON = 60
HORIZON = 180
LAUNCH_CLEARANCE = 0.1
FLEET_SWEEP_HORIZON = 90

EARLY_TURN_LIMIT = 40
OPENING_TURN_LIMIT = 80
LATE_REMAINING_TURNS = 70
VERY_LATE_REMAINING_TURNS = 25
TOTAL_WAR_ENABLED = False
TOTAL_WAR_REMAINING_TURNS = 55
TOTAL_WAR_MIN_SEND = 5

SAFE_NEUTRAL_MARGIN = 2
CONTESTED_NEUTRAL_MARGIN = 2
INTERCEPT_TOLERANCE = 1

SAFE_OPENING_PROD_THRESHOLD = 4
SAFE_OPENING_TURN_LIMIT = 10
ROTATING_OPENING_MAX_TURNS = 13
ROTATING_OPENING_LOW_PROD = 2
FOUR_PLAYER_ROTATING_REACTION_GAP = 3
FOUR_PLAYER_ROTATING_SEND_RATIO = 0.55
FOUR_PLAYER_ROTATING_TURN_LIMIT = 14

COMET_MAX_CHASE_TURNS = 10

ATTACK_COST_TURN_WEIGHT = 0.50
SNIPE_COST_TURN_WEIGHT = 0.45
INDIRECT_VALUE_SCALE = 0.15
INDIRECT_FRIENDLY_WEIGHT = 0.35
INDIRECT_NEUTRAL_WEIGHT = 0.9
INDIRECT_ENEMY_WEIGHT = 1.25

STATIC_NEUTRAL_VALUE_MULT = 1.4
STATIC_HOSTILE_VALUE_MULT = 1.65
ROTATING_OPENING_VALUE_MULT = 0.9
HOSTILE_TARGET_VALUE_MULT = 2.05
OPENING_HOSTILE_TARGET_VALUE_MULT = 1.55
SAFE_NEUTRAL_VALUE_MULT = 1.2
CONTESTED_NEUTRAL_VALUE_MULT = 0.7
EARLY_NEUTRAL_VALUE_MULT = 1.2
COMET_VALUE_MULT = 0.65
SNIPE_VALUE_MULT = 1.13
SWARM_VALUE_MULT = 1.05
REINFORCE_VALUE_MULT = 1.35
CRASH_EXPLOIT_VALUE_MULT = 1.18
FINISHING_HOSTILE_VALUE_MULT = 1.3
BEHIND_ROTATING_NEUTRAL_VALUE_MULT = 0.92
FFA_OPPORTUNISM_ENABLED = False
EXPOSED_PLANET_VALUE_MULT = 1.55
BLOOD_IN_WATER_VALUE_MULT = 1.28
WEAKEST_ENEMY_VALUE_MULT = 1.12
LET_THEM_FIGHT_PENALTY = 0.82
LEADER_DENIAL_ENABLED = False
LEADER_DENIAL_PROD_GAP = 4
LEADER_DENIAL_STRENGTH_GAP = 30
LEADER_DENIAL_VALUE_MULT = 1.35
LEADER_DENIAL_SCORE_MULT = 1.12
LEADER_DENIAL_PRODUCTION_BONUS = 1.8

NEUTRAL_MARGIN_BASE = 2
NEUTRAL_MARGIN_PROD_WEIGHT = 2
NEUTRAL_MARGIN_CAP = 8
HOSTILE_MARGIN_BASE = 3
HOSTILE_MARGIN_PROD_WEIGHT = 2
HOSTILE_MARGIN_CAP = 12
HOSTILE_REINFORCE_ENABLED = False
HOSTILE_REINFORCE_HORIZON = 8
HOSTILE_REINFORCE_RATIO = 0.22
HOSTILE_REINFORCE_CAP = 12
STATIC_TARGET_MARGIN = 4
CONTESTED_TARGET_MARGIN = 5
FOUR_PLAYER_TARGET_MARGIN = 2
LONG_TRAVEL_MARGIN_START = 18
LONG_TRAVEL_MARGIN_DIVISOR = 3
LONG_TRAVEL_MARGIN_CAP = 8
COMET_MARGIN_RELIEF = 6
FINISHING_HOSTILE_SEND_BONUS = 3

STATIC_TARGET_SCORE_MULT = 1.18
EARLY_STATIC_NEUTRAL_SCORE_MULT = 1.25
FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT = 0.92
DENSE_STATIC_NEUTRAL_COUNT = 4
DENSE_ROTATING_NEUTRAL_SCORE_MULT = 0.86
SNIPE_SCORE_MULT = 1.12
SWARM_SCORE_MULT = 1.06
CRASH_EXPLOIT_SCORE_MULT = 1.05
EXPOSED_PLANET_SCORE_MULT = 1.18
BLOOD_IN_WATER_SCORE_MULT = 1.12
WEAKEST_ENEMY_SCORE_MULT = 1.08

FOLLOWUP_MIN_SHIPS = 8
LOW_VALUE_COMET_PRODUCTION = 1
LATE_CAPTURE_BUFFER = 5
VERY_LATE_CAPTURE_BUFFER = 3

DEFENSE_LOOKAHEAD_TURNS = 28
DEFENSE_COST_TURN_WEIGHT = 0.4
DEFENSE_FRONTIER_SCORE_MULT = 1.12
DEFENSE_SEND_MARGIN_BASE = 1
DEFENSE_SEND_MARGIN_PROD_WEIGHT = 1
DEFENSE_SHIP_VALUE = 0.55

REINFORCE_ENABLED = True
REINFORCE_MIN_PRODUCTION = 2
REINFORCE_MAX_TRAVEL_TURNS = 22
REINFORCE_SAFETY_MARGIN = 2
REINFORCE_MAX_SOURCE_FRACTION = 0.75
REINFORCE_MIN_FUTURE_TURNS = 40
REINFORCE_HOLD_LOOKAHEAD = 20
REINFORCE_COST_TURN_WEIGHT = 0.35

RECAPTURE_LOOKAHEAD_TURNS = 10
RECAPTURE_COST_TURN_WEIGHT = 0.52
RECAPTURE_VALUE_MULT = 0.88
RECAPTURE_FRONTIER_MULT = 1.08
RECAPTURE_PRODUCTION_WEIGHT = 0.6
RECAPTURE_IMMEDIATE_WEIGHT = 0.4

REAR_SOURCE_MIN_SHIPS = 16
REAR_DISTANCE_RATIO = 1.25
REAR_STAGE_PROGRESS = 0.78
REAR_SEND_RATIO_TWO_PLAYER = 0.62
REAR_SEND_RATIO_FOUR_PLAYER = 0.7
REAR_SEND_MIN_SHIPS = 10
REAR_MAX_TRAVEL_TURNS = 40

PARTIAL_SOURCE_MIN_SHIPS = 6
MULTI_SOURCE_TOP_K = 10
MULTI_SOURCE_ETA_TOLERANCE = 2
MULTI_SOURCE_PLAN_PENALTY = 0.97
HOSTILE_SWARM_ETA_TOLERANCE = 1
THREE_SOURCE_SWARM_ENABLED = True
THREE_SOURCE_MIN_TARGET_SHIPS = 20
THREE_SOURCE_ETA_TOLERANCE = 2
THREE_SOURCE_PLAN_PENALTY = 0.94

WAIT_STRIKE_ENABLED = True
WAIT_STRIKE_DELAYS = (0, 2, 4, 6)
WAIT_STRIKE_MAX_TARGETS = 6

FOUR_SOURCE_SWARM_ENABLED = False
FOUR_SOURCE_ETA_TOLERANCE = 2
FOUR_SOURCE_MIN_TARGET_SHIPS = 40
FOUR_SOURCE_PLAN_PENALTY = 0.91

PROACTIVE_DEFENSE_HORIZON = 12
PROACTIVE_DEFENSE_RATIO = 0.18
MULTI_ENEMY_PROACTIVE_HORIZON = 14
MULTI_ENEMY_PROACTIVE_RATIO = 0.22
MULTI_ENEMY_STACK_WINDOW = 3
REACTION_SOURCE_TOP_K_MY = 4
REACTION_SOURCE_TOP_K_ENEMY = 4
PROACTIVE_ENEMY_TOP_K = 3

CRASH_EXPLOIT_ENABLED = True
CRASH_EXPLOIT_MIN_TOTAL_SHIPS = 10
CRASH_EXPLOIT_ETA_WINDOW = 2
CRASH_EXPLOIT_POST_CRASH_DELAY = 1

LATE_IMMEDIATE_SHIP_VALUE = 0.6
WEAK_ENEMY_THRESHOLD = 45
ELIMINATION_BONUS = 18.0
FFA_ELIMINATION_SHIPS = 55
FFA_LET_FIGHT_MIN_SHIPS = 14
EXPOSED_OUTBOUND_MIN_SHIPS = 12
EXPOSED_OUTBOUND_RATIO = 0.8

BEHIND_DOMINATION = -0.20
AHEAD_DOMINATION = 0.18
FINISHING_DOMINATION = 0.35
FINISHING_PROD_RATIO = 1.25
AHEAD_ATTACK_MARGIN_BONUS = 0.08
BEHIND_ATTACK_MARGIN_PENALTY = 0.05
FINISHING_ATTACK_MARGIN_BONUS = 0.08

DOOMED_EVAC_TURN_LIMIT = 24
DOOMED_MIN_SHIPS = 8

SOFT_ACT_DEADLINE = 0.82
HEAVY_PHASE_MIN_TIME = 0.16
OPTIONAL_PHASE_MIN_TIME = 0.08
HEAVY_ROUTE_PLANET_LIMIT = 32


# ============================================================
# Shared Types
# ============================================================

Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)


@dataclass(frozen=True)
class ShotOption:
    score: float
    src_id: int
    target_id: int
    angle: float
    turns: int
    needed: int
    send_cap: int
    mission: str = "capture"
    anchor_turn: int | None = None


@dataclass
class Mission:
    kind: str
    score: float
    target_id: int
    turns: int
    options: list[ShotOption] = field(default_factory=list)

# ============================================================
# Physics
# ============================================================

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def orbital_radius(planet):
    return dist(planet.x, planet.y, CENTER_X, CENTER_Y)


def is_static_planet(planet):
    return orbital_radius(planet) + planet.radius >= ROTATION_LIMIT


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio**1.5)


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return dist(px, py, proj_x, proj_y)


def segment_hits_sun(x1, y1, x2, y2, safety=SUN_SAFETY):
    return point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R + safety


def launch_point(sx, sy, sr, angle):
    clearance = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * clearance, sy + math.sin(angle) * clearance


def actual_path_geometry(sx, sy, sr, tx, ty, tr):
    angle = math.atan2(ty - sy, tx - sx)
    start_x, start_y = launch_point(sx, sy, sr, angle)
    hit_distance = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    end_x = start_x + math.cos(angle) * hit_distance
    end_y = start_y + math.sin(angle) * hit_distance
    return angle, start_x, start_y, end_x, end_y, hit_distance


def safe_angle_and_distance(sx, sy, sr, tx, ty, tr):
    # Launch from the source boundary and time the route to the first hit on
    # the target circle.
    angle, start_x, start_y, end_x, end_y, hit_distance = actual_path_geometry(
        sx,
        sy,
        sr,
        tx,
        ty,
        tr,
    )
    if segment_hits_sun(start_x, start_y, end_x, end_y):
        return None
    return angle, hit_distance


def predict_planet_position(planet, initial_by_id, angular_velocity, turns):
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    return (
        CENTER_X + r * math.cos(new_ang),
        CENTER_Y + r * math.sin(new_ang),
    )


def predict_comet_position(planet_id, comets, turns):
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        future_idx = path_index + int(turns)
        if 0 <= future_idx < len(path):
            return path[future_idx][0], path[future_idx][1]
        return None
    return None


def comet_remaining_life(planet_id, comets):
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0


def estimate_arrival(sx, sy, sr, tx, ty, tr, ships):
    # Use one boundary-aware ETA model for routing, ranking, reserve, and
    # launch decisions.
    safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
    return angle, turns


def travel_time(sx, sy, sr, tx, ty, tr, ships):
    est = estimate_arrival(sx, sy, sr, tx, ty, tr, ships)
    if est is None:
        return 10**9
    return est[1]


def predict_target_position(target, turns, initial_by_id, ang_vel, comets, comet_ids):
    if target.id in comet_ids:
        return predict_comet_position(target.id, comets, turns)
    return predict_planet_position(target, initial_by_id, ang_vel, turns)


def target_can_move(target, initial_by_id, comet_ids):
    if target.id in comet_ids:
        return True
    init = initial_by_id.get(target.id)
    if init is None:
        return False
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    return r + init.radius < ROTATION_LIMIT


def search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    # If the direct line is unsafe, scan future positions and keep the earliest
    # viable intercept window.
    best = None
    best_score = None
    max_turns = min(HORIZON, ROUTE_SEARCH_HORIZON)
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))

    for candidate_turns in range(1, max_turns + 1):
        pos = predict_target_position(
            target,
            candidate_turns,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
        )
        if pos is None:
            continue
        est = estimate_arrival(src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships)
        if est is None:
            continue
        _, turns = est
        if abs(turns - candidate_turns) > INTERCEPT_TOLERANCE:
            continue

        actual_turns = max(turns, candidate_turns)
        actual_pos = predict_target_position(
            target,
            actual_turns,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
        )
        if actual_pos is None:
            continue

        confirm = estimate_arrival(
            src.x,
            src.y,
            src.radius,
            actual_pos[0],
            actual_pos[1],
            target.radius,
            ships,
        )
        if confirm is None:
            continue

        delta = abs(confirm[1] - actual_turns)
        if delta > INTERCEPT_TOLERANCE:
            continue

        score = (delta, confirm[1], candidate_turns)
        if best is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], actual_pos[0], actual_pos[1])

    return best


def aim_with_prediction(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    # Iterate toward a self-consistent moving-target intercept, then fall back
    # to a later safe window if needed.
    est = estimate_arrival(src.x, src.y, src.radius, target.x, target.y, target.radius, ships)
    if est is None:
        if not target_can_move(target, initial_by_id, comet_ids):
            return None
        return search_safe_intercept(
            src,
            target,
            ships,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
        )

    tx, ty = target.x, target.y
    for _ in range(5):
        _, turns = est
        pos = predict_target_position(target, turns, initial_by_id, ang_vel, comets, comet_ids)
        if pos is None:
            return None
        ntx, nty = pos
        next_est = estimate_arrival(src.x, src.y, src.radius, ntx, nty, target.radius, ships)
        if next_est is None:
            if not target_can_move(target, initial_by_id, comet_ids):
                return None
            return search_safe_intercept(
                src,
                target,
                ships,
                initial_by_id,
                ang_vel,
                comets,
                comet_ids,
            )
        if (
            abs(ntx - tx) < 0.3
            and abs(nty - ty) < 0.3
            and abs(next_est[1] - turns) <= INTERCEPT_TOLERANCE
        ):
            return next_est[0], next_est[1], ntx, nty
        tx, ty = ntx, nty
        est = next_est

    final_est = estimate_arrival(src.x, src.y, src.radius, tx, ty, target.radius, ships)
    if final_est is None:
        return search_safe_intercept(
            src,
            target,
            ships,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
        )
    return final_est[0], final_est[1], tx, ty

# ============================================================
# World Model
# ============================================================

def fleet_target_planet(
    fleet,
    planets,
    initial_by_id=None,
    ang_vel=0.0,
    comets=(),
    comet_ids=(),
):
    # Project in-flight fleets by ray-circle hit timing to build a usable
    # arrival ledger. Static planets can use an analytic ray hit; rotating
    # planets and comets need a bounded future sweep against predicted centers.
    initial_by_id = initial_by_id or {}
    comet_ids = set(comet_ids or ())
    best_planet = None
    best_time = 1e9
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)
    moving_targets = []

    for planet in planets:
        if planet.id == fleet.from_planet_id:
            continue
        if target_can_move(planet, initial_by_id, comet_ids):
            moving_targets.append(planet)
            continue

        dx = planet.x - fleet.x
        dy = planet.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        radius_sq = planet.radius * planet.radius
        if perp_sq >= radius_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
        turns = hit_d / speed
        if turns <= HORIZON and turns < best_time:
            best_time = turns
            best_planet = planet

    if moving_targets:
        max_turns = min(HORIZON, FLEET_SWEEP_HORIZON)
        prev_x = fleet.x
        prev_y = fleet.y
        for turn in range(1, max_turns + 1):
            if turn > best_time:
                break
            cur_x = fleet.x + dir_x * speed * turn
            cur_y = fleet.y + dir_y * speed * turn
            for planet in moving_targets:
                pos = predict_target_position(
                    planet,
                    turn,
                    initial_by_id,
                    ang_vel,
                    comets,
                    comet_ids,
                )
                if pos is None:
                    continue
                if (
                    point_to_segment_distance(
                        pos[0],
                        pos[1],
                        prev_x,
                        prev_y,
                        cur_x,
                        cur_y,
                    )
                    <= planet.radius
                    and turn < best_time
                ):
                    best_time = turn
                    best_planet = planet
            prev_x = cur_x
            prev_y = cur_y

    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_time))


def build_arrival_ledger(
    fleets,
    planets,
    initial_by_id=None,
    ang_vel=0.0,
    comets=(),
    comet_ids=(),
):
    arrivals_by_planet = {planet.id: [] for planet in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(
            fleet,
            planets,
            initial_by_id=initial_by_id,
            ang_vel=ang_vel,
            comets=comets,
            comet_ids=comet_ids,
        )
        if target is None:
            continue
        arrivals_by_planet[target.id].append((eta, fleet.owner, int(fleet.ships)))
    return arrivals_by_planet


def resolve_arrival_event(owner, garrison, arrivals):
    # Match the environment's same-turn combat order: aggregate by owner, let
    # the top two attackers cancel, then resolve the survivor against garrison.
    by_owner = {}
    for _, attacker_owner, ships in arrivals:
        by_owner[attacker_owner] = by_owner.get(attacker_owner, 0) + ships

    if not by_owner:
        return owner, max(0.0, garrison)

    sorted_players = sorted(by_owner.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = sorted_players[0]

    if len(sorted_players) > 1:
        second_ships = sorted_players[1][1]
        if top_ships == second_ships:
            survivor_owner = -1
            survivor_ships = 0
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships - second_ships
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return owner, max(0.0, garrison)

    if owner == survivor_owner:
        return owner, garrison + survivor_ships

    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison


def normalize_arrivals(arrivals, horizon):
    events = []
    for turns, owner, ships in arrivals:
        if ships <= 0:
            continue
        eta = max(1, int(math.ceil(turns)))
        if eta > horizon:
            continue
        events.append((eta, owner, int(ships)))
    events.sort(key=lambda item: item[0])
    return events


def simulate_planet_timeline(planet, arrivals, player, horizon):
    # Build one reusable future timeline so defense, capture, and evacuation
    # all query the same state model.
    horizon = max(0, int(math.ceil(horizon)))
    events = normalize_arrivals(arrivals, horizon)
    by_turn = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    min_owned = garrison if owner == player else 0.0
    first_enemy = None
    fall_turn = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production

        group = by_turn.get(turn, [])
        prev_owner = owner
        if group:
            if prev_owner == player and first_enemy is None:
                if any(item[1] not in (-1, player) for item in group):
                    first_enemy = turn
            owner, garrison = resolve_arrival_event(owner, garrison, group)
            if prev_owner == player and owner != player and fall_turn is None:
                fall_turn = turn

        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)
        if owner == player:
            min_owned = min(min_owned, garrison)

    keep_needed = 0
    holds_full = True

    if planet.owner == player:

        def survives_with_keep(keep):
            sim_owner = planet.owner
            sim_garrison = float(keep)
            for turn in range(1, horizon + 1):
                if sim_owner != -1:
                    sim_garrison += planet.production
                group = by_turn.get(turn, [])
                if group:
                    sim_owner, sim_garrison = resolve_arrival_event(sim_owner, sim_garrison, group)
                    if sim_owner != player:
                        return False
            return sim_owner == player

        if survives_with_keep(int(planet.ships)):
            lo, hi = 0, int(planet.ships)
            while lo < hi:
                mid = (lo + hi) // 2
                if survives_with_keep(mid):
                    hi = mid
                else:
                    lo = mid + 1
            keep_needed = lo
        else:
            holds_full = False
            keep_needed = int(planet.ships)

    return {
        "owner_at": owner_at,
        "ships_at": ships_at,
        "keep_needed": keep_needed,
        "min_owned": max(0, int(math.floor(min_owned))) if planet.owner == player else 0,
        "first_enemy": first_enemy,
        "fall_turn": fall_turn,
        "holds_full": holds_full,
        "horizon": horizon,
    }


def state_at_timeline(timeline, arrival_turn):
    turn = max(0, int(math.ceil(arrival_turn)))
    turn = min(turn, timeline["horizon"])
    owner = timeline["owner_at"].get(turn, timeline["owner_at"][timeline["horizon"]])
    ships = timeline["ships_at"].get(turn, timeline["ships_at"][timeline["horizon"]])
    return owner, max(0.0, ships)


def count_players(planets, fleets):
    owners = set()
    for planet in planets:
        if planet.owner != -1:
            owners.add(planet.owner)
    for fleet in fleets:
        owners.add(fleet.owner)
    return max(2, len(owners))


def nearest_distance_to_set(px, py, planets):
    if not planets:
        return 10**9
    return min(dist(px, py, planet.x, planet.y) for planet in planets)


def indirect_features(planet, planets, player):
    friendly = 0.0
    neutral = 0.0
    enemy = 0.0
    for other in planets:
        if other.id == planet.id:
            continue
        d = dist(planet.x, planet.y, other.x, other.y)
        if d < 1:
            continue
        factor = other.production / (d + 12.0)
        if other.owner == player:
            friendly += factor
        elif other.owner == -1:
            neutral += factor
        else:
            enemy += factor
    return friendly, neutral, enemy


def detect_exposed_enemy_planets(fleets, enemy_planets):
    exposed = set()
    for planet in enemy_planets:
        outbound = sum(
            int(fleet.ships)
            for fleet in fleets
            if fleet.owner == planet.owner
            and fleet.from_planet_id == planet.id
            and fleet.ships >= 5
        )
        if (
            outbound >= EXPOSED_OUTBOUND_MIN_SHIPS
            and outbound >= planet.ships * EXPOSED_OUTBOUND_RATIO
        ):
            exposed.add(planet.id)
    return exposed


def detect_enemy_fights(arrivals_by_planet, player):
    contested = {}
    for planet_id, arrivals in arrivals_by_planet.items():
        enemy_owners = set()
        enemy_ships = 0
        for _, owner, ships in arrivals:
            if owner not in (-1, player):
                enemy_owners.add(owner)
                enemy_ships += int(ships)
        if len(enemy_owners) >= 2 and enemy_ships >= FFA_LET_FIGHT_MIN_SHIPS:
            contested[planet_id] = enemy_ships
    return contested


class WorldModel:
    def __init__(self, player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids):
        self.player = player
        self.step = step
        self.planets = planets
        self.fleets = fleets
        self.initial_by_id = initial_by_id
        self.ang_vel = ang_vel
        self.comets = comets
        self.comet_ids = set(comet_ids)

        self.planet_by_id = {planet.id: planet for planet in planets}
        self.my_planets = [planet for planet in planets if planet.owner == player]
        self.enemy_planets = [planet for planet in planets if planet.owner not in (-1, player)]
        self.neutral_planets = [planet for planet in planets if planet.owner == -1]
        self.static_neutral_planets = [
            planet for planet in self.neutral_planets if is_static_planet(planet)
        ]

        self.num_players = count_players(planets, fleets)
        self.remaining_steps = max(1, TOTAL_STEPS - step)
        self.is_early = step < EARLY_TURN_LIMIT
        self.is_opening = step < OPENING_TURN_LIMIT
        self.is_late = self.remaining_steps < LATE_REMAINING_TURNS
        self.is_very_late = self.remaining_steps < VERY_LATE_REMAINING_TURNS
        self.is_total_war = TOTAL_WAR_ENABLED and self.remaining_steps < TOTAL_WAR_REMAINING_TURNS
        self.is_four_player = self.num_players >= 4

        self.owner_strength = defaultdict(int)
        self.owner_production = defaultdict(int)
        for planet in planets:
            if planet.owner != -1:
                self.owner_strength[planet.owner] += int(planet.ships)
                self.owner_production[planet.owner] += int(planet.production)
        for fleet in fleets:
            self.owner_strength[fleet.owner] += int(fleet.ships)

        self.my_total = self.owner_strength.get(player, 0)
        self.enemy_total = sum(
            strength for owner, strength in self.owner_strength.items() if owner != player
        )
        self.max_enemy_strength = max(
            (strength for owner, strength in self.owner_strength.items() if owner != player),
            default=0,
        )
        self.my_prod = self.owner_production.get(player, 0)
        self.enemy_prod = sum(
            production
            for owner, production in self.owner_production.items()
            if owner != player
        )
        enemy_owners = [owner for owner in self.owner_strength if owner != player]
        if enemy_owners:
            self.weakest_enemy = min(enemy_owners, key=lambda owner: self.owner_strength[owner])
            self.weakest_enemy_strength = self.owner_strength[self.weakest_enemy]
            self.weakest_enemy_prod = self.owner_production.get(self.weakest_enemy, 0)
            self.leader_enemy = max(
                enemy_owners,
                key=lambda owner: (self.owner_production.get(owner, 0), self.owner_strength[owner]),
            )
            self.leader_enemy_strength = self.owner_strength[self.leader_enemy]
            self.leader_enemy_prod = self.owner_production.get(self.leader_enemy, 0)
        else:
            self.weakest_enemy = None
            self.weakest_enemy_strength = 0
            self.weakest_enemy_prod = 0
            self.leader_enemy = None
            self.leader_enemy_strength = 0
            self.leader_enemy_prod = 0
        self.leader_enemy_is_threat = (
            self.leader_enemy is not None
            and (
                self.leader_enemy_prod >= self.my_prod + LEADER_DENIAL_PROD_GAP
                or self.leader_enemy_strength >= self.my_total + LEADER_DENIAL_STRENGTH_GAP
            )
        )
        self.blood_in_water_owners = {
            owner
            for owner in enemy_owners
            if self.owner_strength[owner] <= FFA_ELIMINATION_SHIPS
        }
        self.opp_planets = defaultdict(list)
        for planet in self.enemy_planets:
            self.opp_planets[planet.owner].append(planet)

        self.arrivals_by_planet = build_arrival_ledger(
            fleets,
            planets,
            initial_by_id=initial_by_id,
            ang_vel=ang_vel,
            comets=comets,
            comet_ids=comet_ids,
        )
        self.base_timeline = {
            planet.id: simulate_planet_timeline(
                planet,
                self.arrivals_by_planet[planet.id],
                player,
                HORIZON,
            )
            for planet in planets
        }
        self.keep_needed_map = {
            planet.id: self.base_timeline[planet.id]["keep_needed"] for planet in planets
        }
        self.min_owned_map = {
            planet.id: self.base_timeline[planet.id]["min_owned"] for planet in planets
        }
        self.first_enemy_map = {
            planet.id: self.base_timeline[planet.id]["first_enemy"] for planet in planets
        }
        self.fall_turn_map = {
            planet.id: self.base_timeline[planet.id]["fall_turn"] for planet in planets
        }
        self.holds_full_map = {
            planet.id: self.base_timeline[planet.id]["holds_full"] for planet in planets
        }
        self.indirect_feature_map = {
            planet.id: indirect_features(planet, planets, player) for planet in planets
        }
        if FFA_OPPORTUNISM_ENABLED:
            self.exposed_planet_ids = detect_exposed_enemy_planets(fleets, self.enemy_planets)
            self.enemy_fights = detect_enemy_fights(self.arrivals_by_planet, player)
        else:
            self.exposed_planet_ids = set()
            self.enemy_fights = {}
        self.shot_cache = {}
        self.probe_candidate_cache = {}
        self.best_probe_cache = {}
        self.reaction_cache = {}
        self.exact_need_cache = {}

        self.total_visible_ships = sum(int(planet.ships) for planet in planets) + sum(
            int(fleet.ships) for fleet in fleets
        )
        self.total_production = sum(int(planet.production) for planet in planets)

    def is_static(self, planet_id):
        return is_static_planet(self.planet_by_id[planet_id])

    def comet_life(self, planet_id):
        return comet_remaining_life(planet_id, self.comets)

    def source_inventory_left(self, source_id, spent_total):
        return max(0, int(self.planet_by_id[source_id].ships) - spent_total[source_id])

    def plan_shot(self, src_id, target_id, ships):
        ships = int(ships)
        key = (src_id, target_id, ships)
        cached = self.shot_cache.get(key)
        if key in self.shot_cache:
            return cached
        src = self.planet_by_id[src_id]
        target = self.planet_by_id[target_id]
        result = aim_with_prediction(
            src,
            target,
            ships,
            self.initial_by_id,
            self.ang_vel,
            self.comets,
            self.comet_ids,
        )
        self.shot_cache[key] = result
        return result

    def probe_ship_candidates(self, src_id, target_id, source_cap, hints=()):
        cache = getattr(self, "probe_candidate_cache", None)
        if cache is None:
            cache = {}
            self.probe_candidate_cache = cache
        source_cap = max(1, int(source_cap))
        normalized_hints = tuple(
            int(math.ceil(hint))
            for hint in hints
            if hint is not None
        )
        cache_key = (src_id, target_id, source_cap, normalized_hints)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        target = self.planet_by_id[target_id]
        target_ships = max(1, int(math.ceil(target.ships)))

        values = set(range(1, min(6, source_cap) + 1))
        values.update(
            {
                source_cap,
                max(1, source_cap // 2),
                max(1, source_cap // 3),
                min(source_cap, PARTIAL_SOURCE_MIN_SHIPS),
                min(source_cap, target_ships + 1),
                min(source_cap, target_ships + 2),
                min(source_cap, target_ships + 4),
                min(source_cap, target_ships + 8),
            }
        )

        for hint in normalized_hints:
            base = max(1, min(source_cap, hint))
            for delta in (-2, -1, 0, 1, 2):
                candidate = base + delta
                if 1 <= candidate <= source_cap:
                    values.add(candidate)

        result = sorted(values)
        cache[cache_key] = result
        return result

    def best_probe_aim(
        self,
        src_id,
        target_id,
        source_cap,
        hints=(),
        min_turn=None,
        max_turn=None,
        anchor_turn=None,
        max_anchor_diff=None,
    ):
        cache_key = (
            src_id,
            target_id,
            max(1, int(source_cap)),
            tuple(hints),
            min_turn,
            max_turn,
            anchor_turn,
            max_anchor_diff,
        )
        cache = getattr(self, "best_probe_cache", None)
        if cache is None:
            cache = {}
            self.best_probe_cache = cache
        if cache_key in cache:
            return cache[cache_key]

        best = None
        best_key = None

        for ships in self.probe_ship_candidates(src_id, target_id, source_cap, hints=hints):
            aim = self.plan_shot(src_id, target_id, ships)
            if aim is None:
                continue

            angle, turns, dist_to_target, path_target = aim
            if min_turn is not None and turns < min_turn:
                continue
            if max_turn is not None and turns > max_turn:
                continue
            if (
                anchor_turn is not None
                and max_anchor_diff is not None
                and abs(turns - anchor_turn) > max_anchor_diff
            ):
                continue

            if anchor_turn is None:
                key = (turns, ships)
            else:
                key = (abs(turns - anchor_turn), turns, ships)

            if best_key is None or key < best_key:
                best_key = key
                best = (ships, (angle, turns, dist_to_target, path_target))

        cache[cache_key] = best
        return best

    def reaction_times(self, target_id):
        cached = self.reaction_cache.get(target_id)
        if cached is not None:
            return cached

        target = self.planet_by_id[target_id]
        my_t = 10**9
        for planet in self.my_planets:
            seeded = self.best_probe_aim(planet.id, target.id, max(1, int(planet.ships)))
            if seeded is None:
                continue
            _, aim = seeded
            my_t = min(my_t, aim[1])

        enemy_t = 10**9
        for planet in self.enemy_planets:
            seeded = self.best_probe_aim(planet.id, target.id, max(1, int(planet.ships)))
            if seeded is None:
                continue
            _, aim = seeded
            enemy_t = min(enemy_t, aim[1])

        cached = (my_t, enemy_t)
        self.reaction_cache[target_id] = cached
        return cached

    def projected_state(self, target_id, arrival_turn, planned_commitments=None, extra_arrivals=()):
        planned_commitments = planned_commitments or {}
        cutoff = max(1, int(math.ceil(arrival_turn)))
        if not planned_commitments.get(target_id) and not extra_arrivals:
            return state_at_timeline(self.base_timeline[target_id], cutoff)

        arrivals = [
            item
            for item in self.arrivals_by_planet.get(target_id, [])
            if item[0] <= cutoff
        ]
        arrivals.extend(
            item
            for item in planned_commitments.get(target_id, [])
            if item[0] <= cutoff
        )
        arrivals.extend(item for item in extra_arrivals if item[0] <= cutoff)

        target = self.planet_by_id[target_id]
        dyn = simulate_planet_timeline(target, arrivals, self.player, cutoff)
        return state_at_timeline(dyn, cutoff)

    def projected_timeline(self, target_id, horizon, planned_commitments=None, extra_arrivals=()):
        planned_commitments = planned_commitments or {}
        horizon = max(1, int(math.ceil(horizon)))
        arrivals = [
            item for item in self.arrivals_by_planet.get(target_id, []) if item[0] <= horizon
        ]
        arrivals.extend(
            item for item in planned_commitments.get(target_id, []) if item[0] <= horizon
        )
        arrivals.extend(item for item in extra_arrivals if item[0] <= horizon)
        target = self.planet_by_id[target_id]
        return simulate_planet_timeline(target, arrivals, self.player, horizon)

    def hold_status(self, target_id, planned_commitments=None, horizon=HORIZON):
        planned_commitments = planned_commitments or {}
        if planned_commitments.get(target_id):
            tl = self.projected_timeline(
                target_id,
                horizon,
                planned_commitments=planned_commitments,
            )
        else:
            tl = self.base_timeline[target_id]
        return {
            "keep_needed": tl["keep_needed"],
            "min_owned": tl["min_owned"],
            "first_enemy": tl["first_enemy"],
            "fall_turn": tl["fall_turn"],
            "holds_full": tl["holds_full"],
        }

    def _ownership_search_cap(self, eval_turn):
        productive_cap = self.total_production * max(2, eval_turn + 2)
        return max(32, int(self.total_visible_ships + productive_cap + 32))

    def min_ships_to_own_by(
        self,
        target_id,
        eval_turn,
        attacker_owner,
        arrival_turn=None,
        planned_commitments=None,
        extra_arrivals=(),
        upper_bound=None,
    ):
        planned_commitments = planned_commitments or {}
        eval_turn = max(1, int(math.ceil(eval_turn)))
        arrival_turn = eval_turn if arrival_turn is None else max(1, int(math.ceil(arrival_turn)))
        if arrival_turn > eval_turn:
            if upper_bound is not None:
                return max(1, int(upper_bound)) + 1
            return self._ownership_search_cap(eval_turn) + 1

        normalized_extra = tuple(
            (
                max(1, int(math.ceil(turns))),
                owner,
                int(ships),
            )
            for turns, owner, ships in extra_arrivals
            if ships > 0 and max(1, int(math.ceil(turns))) <= eval_turn
        )

        cache_key = None
        if (
            arrival_turn == eval_turn
            and not planned_commitments.get(target_id)
            and not normalized_extra
        ):
            cache_key = (target_id, eval_turn, attacker_owner)
            cached = self.exact_need_cache.get(cache_key)
            if cached is not None:
                return cached

        owner_before, ships_before = self.projected_state(
            target_id,
            eval_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=normalized_extra,
        )
        if owner_before == attacker_owner:
            if cache_key is not None:
                self.exact_need_cache[cache_key] = 0
            return 0

        def owns_at(ships):
            owner_after, _ = self.projected_state(
                target_id,
                eval_turn,
                planned_commitments=planned_commitments,
                extra_arrivals=normalized_extra + ((arrival_turn, attacker_owner, int(ships)),),
            )
            return owner_after == attacker_owner

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not owns_at(hi):
                return hi + 1
        else:
            hi = max(1, int(math.ceil(ships_before)) + 1)
            search_cap = self._ownership_search_cap(eval_turn)
            while hi <= search_cap and not owns_at(hi):
                hi *= 2
            if hi > search_cap:
                hi = search_cap
                if not owns_at(hi):
                    return hi + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if owns_at(mid):
                hi = mid
            else:
                lo = mid + 1

        if cache_key is not None:
            self.exact_need_cache[cache_key] = lo
        return lo

    def min_ships_to_own_at(
        self,
        target_id,
        arrival_turn,
        attacker_owner,
        planned_commitments=None,
        extra_arrivals=(),
        upper_bound=None,
    ):
        return self.min_ships_to_own_by(
            target_id,
            arrival_turn,
            attacker_owner,
            arrival_turn=arrival_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
            upper_bound=upper_bound,
        )

    def reinforcement_needed_to_hold_until(
        self,
        planet_id,
        arrival_turn,
        hold_until,
        planned_commitments=None,
        upper_bound=None,
    ):
        planned_commitments = planned_commitments or {}
        target = self.planet_by_id[planet_id]
        arrival_turn = max(1, int(math.ceil(arrival_turn)))
        hold_until = max(arrival_turn, int(math.ceil(hold_until)))

        if target.owner != self.player:
            return self.min_ships_to_own_by(
                planet_id,
                hold_until,
                self.player,
                arrival_turn=arrival_turn,
                planned_commitments=planned_commitments,
                upper_bound=upper_bound,
            )

        def holds_with_reinforcement(ships):
            timeline = self.projected_timeline(
                planet_id,
                hold_until,
                planned_commitments=planned_commitments,
                extra_arrivals=((arrival_turn, self.player, int(ships)),),
            )
            for turn in range(arrival_turn, hold_until + 1):
                if timeline["owner_at"].get(turn) != self.player:
                    return False
            return True

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not holds_with_reinforcement(hi):
                return hi + 1
        else:
            hi = 1
            search_cap = self._ownership_search_cap(hold_until)
            while hi <= search_cap and not holds_with_reinforcement(hi):
                hi *= 2
            if hi > search_cap:
                hi = search_cap
                if not holds_with_reinforcement(hi):
                    return hi + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if holds_with_reinforcement(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def ships_needed_to_capture(
        self,
        target_id,
        arrival_turn,
        planned_commitments=None,
        extra_arrivals=(),
    ):
        return self.min_ships_to_own_at(
            target_id,
            arrival_turn,
            self.player,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
        )

# ============================================================
# Strategy
# ============================================================

def planet_distance(first, second):
    return math.hypot(first.x - second.x, first.y - second.y)


def nearest_sources_to_target(target, sources, top_k):
    if top_k <= 0 or len(sources) <= top_k:
        return sources
    return sorted(
        sources,
        key=lambda src: (planet_distance(src, target), -int(src.ships), src.id),
    )[:top_k]


def min_legal_reaction_time(target, sources, world):
    best = 10**9
    for src in sources:
        seeded = world.best_probe_aim(src.id, target.id, max(1, int(src.ships)))
        if seeded is None:
            continue
        _, aim = seeded
        best = min(best, aim[1])
    return best


def policy_reaction_times(target_id, policy):
    return policy["reaction_time_map"].get(target_id, (10**9, 10**9))


def candidate_time_valid(target, turns, world, remaining_buffer):
    if turns > world.remaining_steps - remaining_buffer:
        return False
    if target.id in world.comet_ids:
        life = world.comet_life(target.id)
        if turns >= life or turns > COMET_MAX_CHASE_TURNS:
            return False
    return True


def stacked_enemy_proactive_keep(planet, world):
    threats = []
    for enemy in world.enemy_planets:
        seeded = world.best_probe_aim(
            enemy.id,
            planet.id,
            max(1, int(enemy.ships)),
        )
        if seeded is None:
            continue
        _, aim = seeded
        eta = aim[1]
        if eta > MULTI_ENEMY_PROACTIVE_HORIZON:
            continue
        threats.append((eta, int(enemy.ships)))

    if not threats:
        return 0

    threats.sort()
    best_stacked = 0
    left = 0
    running = 0
    for right in range(len(threats)):
        running += threats[right][1]
        while threats[right][0] - threats[left][0] > MULTI_ENEMY_STACK_WINDOW:
            running -= threats[left][1]
            left += 1
        best_stacked = max(best_stacked, running)

    return int(best_stacked * MULTI_ENEMY_PROACTIVE_RATIO)


def swarm_eta_tolerance(options, target, world):
    if len(options) >= 3:
        return THREE_SOURCE_ETA_TOLERANCE
    if target.owner not in (-1, world.player):
        return HOSTILE_SWARM_ETA_TOLERANCE
    return MULTI_SOURCE_ETA_TOLERANCE


def detect_enemy_crashes(world):
    crashes = []
    for target_id, arrivals in world.arrivals_by_planet.items():
        enemy_events = [
            (int(math.ceil(eta)), owner, int(ships))
            for eta, owner, ships in arrivals
            if owner not in (-1, world.player) and ships > 0
        ]
        enemy_events.sort()
        for i in range(len(enemy_events)):
            eta_a, owner_a, ships_a = enemy_events[i]
            for j in range(i + 1, len(enemy_events)):
                eta_b, owner_b, ships_b = enemy_events[j]
                if owner_a == owner_b:
                    continue
                if abs(eta_a - eta_b) > CRASH_EXPLOIT_ETA_WINDOW:
                    break
                if ships_a + ships_b < CRASH_EXPLOIT_MIN_TOTAL_SHIPS:
                    continue
                crashes.append(
                    {
                        "target_id": target_id,
                        "crash_turn": max(eta_a, eta_b),
                        "owners": (owner_a, owner_b),
                        "ships": (ships_a, ships_b),
                    }
                )
    return crashes


def build_policy_state(world, deadline=None):
    def expired():
        return deadline is not None and time.perf_counter() > deadline

    indirect_wealth_map = {}
    for target_id, features in world.indirect_feature_map.items():
        friendly, neutral, enemy = features
        indirect_wealth_map[target_id] = (
            friendly * INDIRECT_FRIENDLY_WEIGHT
            + neutral * INDIRECT_NEUTRAL_WEIGHT
            + enemy * INDIRECT_ENEMY_WEIGHT
        )

    reserve = {}
    attack_budget = {}
    reaction_time_map = {}

    for target in world.planets:
        if expired():
            break
        if target.owner == world.player:
            continue
        my_sources = nearest_sources_to_target(target, world.my_planets, REACTION_SOURCE_TOP_K_MY)
        enemy_sources = nearest_sources_to_target(target, world.enemy_planets, REACTION_SOURCE_TOP_K_ENEMY)
        my_t = min_legal_reaction_time(target, my_sources, world)
        enemy_t = min_legal_reaction_time(target, enemy_sources, world)
        reaction_time_map[target.id] = (my_t, enemy_t)

    for planet in world.my_planets:
        if expired():
            break
        exact_keep = world.keep_needed_map.get(planet.id, 0)

        proactive_keep = 0
        for enemy in nearest_sources_to_target(planet, world.enemy_planets, PROACTIVE_ENEMY_TOP_K):
            enemy_aim = world.plan_shot(enemy.id, planet.id, max(1, int(enemy.ships)))
            if enemy_aim is None:
                continue
            enemy_eta = enemy_aim[1]
            if enemy_eta > PROACTIVE_DEFENSE_HORIZON:
                continue
            proactive_keep = max(
                proactive_keep,
                int(enemy.ships * PROACTIVE_DEFENSE_RATIO),
            )
        proactive_keep = max(proactive_keep, stacked_enemy_proactive_keep(planet, world))

        if world.is_total_war:
            exact_keep = min(exact_keep, max(1, exact_keep // 2))
            proactive_keep = min(proactive_keep, max(1, proactive_keep // 2))

        reserve[planet.id] = min(int(planet.ships), max(exact_keep, proactive_keep))
        attack_budget[planet.id] = max(0, int(planet.ships) - reserve[planet.id])

    return {
        "indirect_wealth_map": indirect_wealth_map,
        "reserve": reserve,
        "attack_budget": attack_budget,
        "reaction_time_map": reaction_time_map,
    }


def build_modes(world):
    domination = (world.my_total - world.enemy_total) / max(1, world.my_total + world.enemy_total)
    is_behind = domination < BEHIND_DOMINATION
    is_ahead = domination > AHEAD_DOMINATION
    is_dominating = is_ahead or (
        world.max_enemy_strength > 0 and world.my_total > world.max_enemy_strength * 1.25
    )
    is_finishing = (
        domination > FINISHING_DOMINATION
        and world.my_prod > world.enemy_prod * FINISHING_PROD_RATIO
        and world.step > 100
    )

    attack_margin_mult = 1.0
    if is_ahead:
        attack_margin_mult += AHEAD_ATTACK_MARGIN_BONUS
    if is_behind:
        attack_margin_mult -= BEHIND_ATTACK_MARGIN_PENALTY
    if is_finishing:
        attack_margin_mult += FINISHING_ATTACK_MARGIN_BONUS

    return {
        "domination": domination,
        "is_behind": is_behind,
        "is_ahead": is_ahead,
        "is_dominating": is_dominating,
        "is_finishing": is_finishing,
        "attack_margin_mult": attack_margin_mult,
    }


def is_safe_neutral(target, policy):
    if target.owner != -1:
        return False
    my_t, enemy_t = policy_reaction_times(target.id, policy)
    return my_t <= enemy_t - SAFE_NEUTRAL_MARGIN


def is_contested_neutral(target, policy):
    if target.owner != -1:
        return False
    my_t, enemy_t = policy_reaction_times(target.id, policy)
    return abs(my_t - enemy_t) <= CONTESTED_NEUTRAL_MARGIN


def opening_filter(target, arrival_turns, needed, src_available, world, policy):
    if not world.is_opening or target.owner != -1:
        return False
    if target.id in world.comet_ids:
        return False
    if world.is_static(target.id):
        return False

    my_t, enemy_t = policy_reaction_times(target.id, policy)
    reaction_gap = enemy_t - my_t
    if (
        target.production >= SAFE_OPENING_PROD_THRESHOLD
        and arrival_turns <= SAFE_OPENING_TURN_LIMIT
        and reaction_gap >= SAFE_NEUTRAL_MARGIN
    ):
        return False

    if world.is_four_player:
        affordable = needed <= max(
            PARTIAL_SOURCE_MIN_SHIPS,
            int(src_available * FOUR_PLAYER_ROTATING_SEND_RATIO),
        )
        if (
            affordable
            and arrival_turns <= FOUR_PLAYER_ROTATING_TURN_LIMIT
            and reaction_gap >= FOUR_PLAYER_ROTATING_REACTION_GAP
        ):
            return False
        return True

    return arrival_turns > ROTATING_OPENING_MAX_TURNS or target.production <= ROTATING_OPENING_LOW_PROD


def target_value(target, arrival_turns, mission, world, modes, policy):
    turns_profit = max(1, world.remaining_steps - arrival_turns)
    if target.id in world.comet_ids:
        life = world.comet_life(target.id)
        turns_profit = max(0, min(turns_profit, life - arrival_turns))
        if turns_profit <= 0:
            return -1.0

    value = target.production * turns_profit
    value += policy["indirect_wealth_map"][target.id] * turns_profit * INDIRECT_VALUE_SCALE

    if world.is_static(target.id):
        value *= STATIC_NEUTRAL_VALUE_MULT if target.owner == -1 else STATIC_HOSTILE_VALUE_MULT
    else:
        value *= ROTATING_OPENING_VALUE_MULT if world.is_opening else 1.0

    if target.owner not in (-1, world.player):
        value *= OPENING_HOSTILE_TARGET_VALUE_MULT if world.is_opening else HOSTILE_TARGET_VALUE_MULT

    if (
        LEADER_DENIAL_ENABLED
        and world.is_four_player
        and world.leader_enemy_is_threat
        and target.owner == world.leader_enemy
    ):
        value *= LEADER_DENIAL_VALUE_MULT
        value += target.production * LEADER_DENIAL_PRODUCTION_BONUS

    if target.owner == -1:
        if is_safe_neutral(target, policy):
            value *= SAFE_NEUTRAL_VALUE_MULT
        elif is_contested_neutral(target, policy):
            value *= CONTESTED_NEUTRAL_VALUE_MULT
        if world.is_early:
            value *= EARLY_NEUTRAL_VALUE_MULT

    if target.id in world.comet_ids:
        value *= COMET_VALUE_MULT

    if mission == "snipe":
        value *= SNIPE_VALUE_MULT
    elif mission == "swarm":
        value *= SWARM_VALUE_MULT
    elif mission == "reinforce":
        value *= REINFORCE_VALUE_MULT
    elif mission == "crash_exploit":
        value *= CRASH_EXPLOIT_VALUE_MULT

    if FFA_OPPORTUNISM_ENABLED and world.is_four_player:
        if target.owner in world.blood_in_water_owners:
            value *= BLOOD_IN_WATER_VALUE_MULT
            value += ELIMINATION_BONUS * 0.6
        if target.id in world.exposed_planet_ids:
            value *= EXPOSED_PLANET_VALUE_MULT
        if target.owner == world.weakest_enemy and target.owner not in (-1, world.player):
            value *= WEAKEST_ENEMY_VALUE_MULT
        if target.owner == -1 and target.id in world.enemy_fights:
            value *= LET_THEM_FIGHT_PENALTY

    if world.is_late:
        value += max(0, target.ships) * LATE_IMMEDIATE_SHIP_VALUE
        if target.owner not in (-1, world.player):
            enemy_strength = world.owner_strength.get(target.owner, 0)
            if enemy_strength <= WEAK_ENEMY_THRESHOLD:
                value += ELIMINATION_BONUS

    if modes["is_finishing"] and target.owner not in (-1, world.player):
        value *= FINISHING_HOSTILE_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and not world.is_static(target.id):
        value *= BEHIND_ROTATING_NEUTRAL_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and is_safe_neutral(target, policy):
        value *= 1.08
    if modes["is_dominating"] and target.owner == -1 and is_contested_neutral(target, policy):
        value *= 0.92

    return value


def reinforce_value(target, hold_until, world, policy):
    saved_turns = max(1, world.remaining_steps - hold_until)
    value = target.production * saved_turns + max(0, target.ships) * DEFENSE_SHIP_VALUE
    if world.enemy_planets and nearest_distance_to_set(target.x, target.y, world.enemy_planets) < 22:
        value *= DEFENSE_FRONTIER_SCORE_MULT
    value += policy["indirect_wealth_map"][target.id] * saved_turns * INDIRECT_VALUE_SCALE * 0.35
    return value * REINFORCE_VALUE_MULT


def preferred_send(target, base_needed, arrival_turns, src_available, world, modes, policy):
    send = max(base_needed, int(math.ceil(base_needed * modes["attack_margin_mult"])))
    margin = 0
    if target.owner == -1:
        margin += min(
            NEUTRAL_MARGIN_CAP,
            NEUTRAL_MARGIN_BASE + target.production * NEUTRAL_MARGIN_PROD_WEIGHT,
        )
    else:
        margin += min(
            HOSTILE_MARGIN_CAP,
            HOSTILE_MARGIN_BASE + target.production * HOSTILE_MARGIN_PROD_WEIGHT,
        )
        if HOSTILE_REINFORCE_ENABLED:
            reinforce_est = 0
            for enemy_source in world.opp_planets.get(target.owner, []):
                if enemy_source.id == target.id:
                    continue
                enemy_aim = world.plan_shot(
                    enemy_source.id,
                    target.id,
                    max(1, int(enemy_source.ships)),
                )
                if enemy_aim is None:
                    continue
                if enemy_aim[1] <= arrival_turns + HOSTILE_REINFORCE_HORIZON:
                    reinforce_est += max(0, int(enemy_source.ships) - 3)
            margin += min(
                HOSTILE_REINFORCE_CAP,
                int(reinforce_est * HOSTILE_REINFORCE_RATIO),
            )
    if world.is_static(target.id):
        margin += STATIC_TARGET_MARGIN
    if is_contested_neutral(target, policy):
        margin += CONTESTED_TARGET_MARGIN
    if world.is_four_player:
        margin += FOUR_PLAYER_TARGET_MARGIN
    if arrival_turns > LONG_TRAVEL_MARGIN_START:
        margin += min(LONG_TRAVEL_MARGIN_CAP, arrival_turns // LONG_TRAVEL_MARGIN_DIVISOR)
    if target.id in world.comet_ids:
        margin = max(0, margin - COMET_MARGIN_RELIEF)
    if modes["is_finishing"] and target.owner not in (-1, world.player):
        margin += FINISHING_HOSTILE_SEND_BONUS
    return min(src_available, send + margin)


def apply_score_modifiers(base_score, target, mission, world):
    score = base_score
    if world.is_static(target.id):
        score *= STATIC_TARGET_SCORE_MULT
    if world.is_early and target.owner == -1 and world.is_static(target.id):
        score *= EARLY_STATIC_NEUTRAL_SCORE_MULT
    if world.is_four_player and target.owner == -1 and not world.is_static(target.id):
        score *= FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT
    if (
        LEADER_DENIAL_ENABLED
        and world.is_four_player
        and world.leader_enemy_is_threat
        and target.owner == world.leader_enemy
    ):
        score *= LEADER_DENIAL_SCORE_MULT
    if (
        len(world.static_neutral_planets) >= DENSE_STATIC_NEUTRAL_COUNT
        and target.owner == -1
        and not world.is_static(target.id)
    ):
        score *= DENSE_ROTATING_NEUTRAL_SCORE_MULT
    if mission == "snipe":
        score *= SNIPE_SCORE_MULT
    elif mission == "swarm":
        score *= SWARM_SCORE_MULT
    elif mission == "crash_exploit":
        score *= CRASH_EXPLOIT_SCORE_MULT
    if FFA_OPPORTUNISM_ENABLED and world.is_four_player:
        if target.id in world.exposed_planet_ids:
            score *= EXPOSED_PLANET_SCORE_MULT
        if target.owner in world.blood_in_water_owners:
            score *= BLOOD_IN_WATER_SCORE_MULT
        if target.owner == world.weakest_enemy and target.owner not in (-1, world.player):
            score *= WEAKEST_ENEMY_SCORE_MULT
    return score


def settle_plan(
    src,
    target,
    src_cap,
    send_guess,
    world,
    planned_commitments,
    modes,
    policy,
    mission="capture",
    eval_turn_fn=None,
    anchor_turn=None,
    anchor_tolerance=None,
    max_iter=4,
):
    if src_cap < 1:
        return None

    seed_hint = max(1, min(src_cap, int(send_guess)))
    eval_turn_fn = eval_turn_fn or (lambda turns: turns)
    anchor_tolerance = (
        anchor_tolerance
        if anchor_tolerance is not None
        else (1 if mission == "snipe" else None)
    )
    tested = {}
    tested_order = []

    def evaluate(send):
        send = max(1, min(src_cap, int(send)))
        cached = tested.get(send)
        if cached is not None or send in tested:
            return cached

        aim = world.plan_shot(src.id, target.id, send)
        if aim is None:
            tested[send] = None
            return None

        angle, turns, _, _ = aim
        if mission == "crash_exploit" and anchor_turn is not None and turns < anchor_turn:
            tested[send] = None
            return None
        raw_eval_turn = int(math.ceil(eval_turn_fn(turns)))
        if raw_eval_turn < turns:
            tested[send] = None
            return None
        eval_turn = raw_eval_turn
        need = world.min_ships_to_own_by(
            target.id,
            eval_turn,
            world.player,
            arrival_turn=turns,
            planned_commitments=planned_commitments,
            upper_bound=src_cap,
        )
        if need <= 0 or need > src_cap:
            tested[send] = None
            return None

        if mission in ("snipe", "crash_exploit"):
            desired = need
        elif mission == "rescue":
            desired = min(
                src_cap,
                max(
                    need,
                    need + DEFENSE_SEND_MARGIN_BASE + target.production * DEFENSE_SEND_MARGIN_PROD_WEIGHT,
                ),
            )
        else:
            desired = min(
                src_cap,
                max(need, preferred_send(target, need, turns, src_cap, world, modes, policy)),
            )

        result = (angle, turns, eval_turn, need, send, desired)
        tested[send] = result
        tested_order.append(send)
        return result

    initial_candidates = sorted(
        world.probe_ship_candidates(
            src.id,
            target.id,
            src_cap,
            hints=(seed_hint,),
        ),
        key=lambda send: (abs(send - seed_hint), send),
    )

    current_send = None
    for seed in initial_candidates:
        result = evaluate(seed)
        if result is None:
            continue
        if (
            anchor_turn is not None
            and anchor_tolerance is not None
            and abs(result[1] - anchor_turn) > anchor_tolerance
        ):
            continue
        current_send = seed
        break

    if current_send is None:
        return None

    for _ in range(max_iter):
        result = evaluate(current_send)
        if result is None:
            break

        angle, turns, eval_turn, need, actual_send, desired = result
        if desired == actual_send:
            if (
                anchor_turn is not None
                and anchor_tolerance is not None
                and abs(turns - anchor_turn) > anchor_tolerance
            ):
                return None
            if mission == "rescue" and turns > eval_turn:
                return None
            return angle, turns, eval_turn, need, actual_send

        next_send = max(1, min(src_cap, int(desired)))
        if next_send in tested:
            current_send = next_send
            break
        current_send = next_send

    candidate_sends = sorted(
        [send for send in tested_order if tested.get(send) is not None],
        key=lambda send: (
            0
            if mission != "snipe" or anchor_turn is None
            else abs(tested[send][1] - anchor_turn),
            abs(send - seed_hint),
            tested[send][1],
            send,
        ),
    )

    seen = set()
    for send in candidate_sends:
        if send in seen:
            continue
        seen.add(send)
        result = tested.get(send)
        if result is None:
            continue
        angle, turns, eval_turn, need, actual_send, _ = result
        if actual_send < need:
            continue
        if (
            anchor_turn is not None
            and anchor_tolerance is not None
            and abs(turns - anchor_turn) > anchor_tolerance
        ):
            continue
        if mission == "rescue" and turns > eval_turn:
            continue
        return angle, turns, eval_turn, need, actual_send

    return None


def settle_reinforce_plan(
    src,
    target,
    src_cap,
    send_guess,
    world,
    planned_commitments,
    hold_until,
    max_arrival_turn,
    max_iter=4,
):
    if src_cap < 1:
        return None

    seed_hint = max(1, min(src_cap, int(send_guess)))
    tested = {}
    tested_order = []

    def evaluate(send):
        send = max(1, min(src_cap, int(send)))
        cached = tested.get(send)
        if cached is not None or send in tested:
            return cached

        aim = world.plan_shot(src.id, target.id, send)
        if aim is None:
            tested[send] = None
            return None

        angle, turns, _, _ = aim
        if turns > max_arrival_turn:
            tested[send] = None
            return None

        need = world.reinforcement_needed_to_hold_until(
            target.id,
            turns,
            hold_until,
            planned_commitments=planned_commitments,
            upper_bound=src_cap,
        )
        if need <= 0 or need > src_cap:
            tested[send] = None
            return None

        desired = min(src_cap, need + REINFORCE_SAFETY_MARGIN)
        result = (angle, turns, hold_until, need, send, desired)
        tested[send] = result
        tested_order.append(send)
        return result

    initial_candidates = sorted(
        world.probe_ship_candidates(
            src.id,
            target.id,
            src_cap,
            hints=(seed_hint,),
        ),
        key=lambda send: (abs(send - seed_hint), send),
    )

    current_send = None
    for seed in initial_candidates:
        result = evaluate(seed)
        if result is None:
            continue
        current_send = seed
        break

    if current_send is None:
        return None

    for _ in range(max_iter):
        result = evaluate(current_send)
        if result is None:
            break

        angle, turns, eval_turn, need, actual_send, desired = result
        if desired == actual_send:
            return angle, turns, eval_turn, need, actual_send

        next_send = max(1, min(src_cap, int(desired)))
        if next_send in tested:
            current_send = next_send
            break
        current_send = next_send

    candidate_sends = sorted(
        [send for send in tested_order if tested.get(send) is not None],
        key=lambda send: (abs(send - seed_hint), tested[send][1], send),
    )
    for send in candidate_sends:
        result = tested.get(send)
        if result is None:
            continue
        angle, turns, eval_turn, need, actual_send, _ = result
        if actual_send < need or turns > max_arrival_turn:
            continue
        return angle, turns, eval_turn, need, actual_send

    return None


def build_snipe_mission(src, target, src_available, world, planned_commitments, modes, policy):
    if target.owner != -1:
        return None

    enemy_etas = sorted(
        {
            int(math.ceil(eta))
            for eta, owner, ships in world.arrivals_by_planet.get(target.id, [])
            if owner not in (-1, world.player) and ships > 0
        }
    )
    if not enemy_etas:
        return None

    best = None
    for enemy_eta in enemy_etas[:3]:
        seeded = world.best_probe_aim(
            src.id,
            target.id,
            src_available,
            hints=(int(target.ships) + 1, int(target.ships) + 8),
            anchor_turn=enemy_eta,
            max_anchor_diff=1,
        )
        if seeded is None:
            continue

        probe, rough = seeded
        sync_turn = max(rough[1], enemy_eta)
        if target.id in world.comet_ids:
            life = world.comet_life(target.id)
            if sync_turn >= life or sync_turn > COMET_MAX_CHASE_TURNS:
                continue

        plan = settle_plan(
            src,
            target,
            src_available,
            probe,
            world,
            planned_commitments,
            modes,
            policy,
            mission="snipe",
            eval_turn_fn=lambda turns, enemy_eta=enemy_eta: max(turns, enemy_eta),
            anchor_turn=enemy_eta,
        )
        if plan is None:
            continue

        angle, turns, sync_turn, need, send_pref = plan
        if target.id in world.comet_ids:
            life = world.comet_life(target.id)
            if sync_turn >= life or sync_turn > COMET_MAX_CHASE_TURNS:
                continue

        value = target_value(target, sync_turn, "snipe", world, modes, policy)
        if value <= 0:
            continue

        score = apply_score_modifiers(
            value / (send_pref + sync_turn * SNIPE_COST_TURN_WEIGHT + 1.0),
            target,
            "snipe",
            world,
        )
        option = ShotOption(
            score=score,
            src_id=src.id,
            target_id=target.id,
            angle=angle,
            turns=turns,
            needed=need,
            send_cap=send_pref,
            mission="snipe",
            anchor_turn=enemy_eta,
        )
        mission_obj = Mission(
            kind="snipe",
            score=score,
            target_id=target.id,
            turns=sync_turn,
            options=[option],
        )
        if best is None or mission_obj.score > best.score:
            best = mission_obj

    return best


def build_rescue_missions(world, policy, planned_commitments, modes):
    missions = []

    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None or fall_turn > DEFENSE_LOOKAHEAD_TURNS:
            continue

        for src in world.my_planets:
            if src.id == target.id:
                continue

            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(target.production + DEFENSE_SEND_MARGIN_BASE + 2,),
                max_turn=fall_turn,
            )
            if seeded is None:
                continue
            probe, probe_aim = seeded
            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="rescue",
                eval_turn_fn=lambda _turns, fall_turn=fall_turn: fall_turn,
                anchor_turn=fall_turn,
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            saved_turns = max(1, world.remaining_steps - fall_turn)
            value = target.production * saved_turns + max(0, target.ships) * DEFENSE_SHIP_VALUE
            if world.enemy_planets and nearest_distance_to_set(target.x, target.y, world.enemy_planets) < 22:
                value *= DEFENSE_FRONTIER_SCORE_MULT
            score = value / (send_pref + turns * DEFENSE_COST_TURN_WEIGHT + 1.0)

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="rescue",
                anchor_turn=fall_turn,
            )
            missions.append(
                Mission(
                    kind="rescue",
                    score=score,
                    target_id=target.id,
                    turns=fall_turn,
                    options=[option],
                )
            )

    return missions


def build_recapture_missions(world, policy, planned_commitments, modes):
    missions = []

    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None or fall_turn > DEFENSE_LOOKAHEAD_TURNS:
            continue

        for src in world.my_planets:
            if src.id == target.id:
                continue

            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(target.production + DEFENSE_SEND_MARGIN_BASE + 2,),
                min_turn=fall_turn + 1,
                max_turn=fall_turn + RECAPTURE_LOOKAHEAD_TURNS,
            )
            if seeded is None:
                continue
            probe, probe_aim = seeded
            probe_turns = probe_aim[1]

            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="capture",
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            if turns <= fall_turn or turns - fall_turn > RECAPTURE_LOOKAHEAD_TURNS:
                continue

            saved_turns = max(1, world.remaining_steps - turns)
            value = (
                RECAPTURE_PRODUCTION_WEIGHT * target.production * saved_turns
                + RECAPTURE_IMMEDIATE_WEIGHT * max(0, target.ships)
            )
            if world.enemy_planets and nearest_distance_to_set(target.x, target.y, world.enemy_planets) < 22:
                value *= RECAPTURE_FRONTIER_MULT
            value *= RECAPTURE_VALUE_MULT
            score = value / (send_pref + turns * RECAPTURE_COST_TURN_WEIGHT + 1.0)

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="recapture",
                anchor_turn=fall_turn,
            )
            missions.append(
                Mission(
                    kind="recapture",
                    score=score,
                    target_id=target.id,
                    turns=turns,
                    options=[option],
                )
            )

    return missions


def build_reinforce_missions(world, policy, planned_commitments, modes, inventory_left_fn):
    if not REINFORCE_ENABLED:
        return []

    missions = []
    if world.remaining_steps < REINFORCE_MIN_FUTURE_TURNS:
        return missions

    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None:
            continue
        if target.production < REINFORCE_MIN_PRODUCTION:
            continue

        hold_until = min(HORIZON, fall_turn + REINFORCE_HOLD_LOOKAHEAD)
        max_arrival_turn = min(fall_turn, REINFORCE_MAX_TRAVEL_TURNS)

        for src in world.my_planets:
            if src.id == target.id:
                continue

            budget = inventory_left_fn(src.id)
            source_cap = min(budget, int(src.ships * REINFORCE_MAX_SOURCE_FRACTION))
            if source_cap < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                source_cap,
                hints=(target.production + REINFORCE_SAFETY_MARGIN + 2,),
                max_turn=max_arrival_turn,
            )
            if seeded is None:
                continue
            probe, _ = seeded

            plan = settle_reinforce_plan(
                src,
                target,
                source_cap,
                probe,
                world,
                planned_commitments,
                hold_until,
                max_arrival_turn,
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            value = reinforce_value(target, hold_until, world, policy)
            score = value / (send_pref + turns * REINFORCE_COST_TURN_WEIGHT + 1.0)

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="reinforce",
                anchor_turn=hold_until,
            )
            missions.append(
                Mission(
                    kind="reinforce",
                    score=score,
                    target_id=target.id,
                    turns=fall_turn,
                    options=[option],
                )
            )

    return missions


def build_crash_exploit_missions(world, policy, planned_commitments, modes):
    if not CRASH_EXPLOIT_ENABLED or not world.is_four_player:
        return []

    missions = []
    for crash in detect_enemy_crashes(world):
        target = world.planet_by_id[crash["target_id"]]
        if target.owner == world.player:
            continue
        desired_arrival = crash["crash_turn"] + CRASH_EXPLOIT_POST_CRASH_DELAY

        for src in world.my_planets:
            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(12, int(target.ships) + 1),
                anchor_turn=desired_arrival,
                max_anchor_diff=CRASH_EXPLOIT_ETA_WINDOW,
            )
            if seeded is None:
                continue
            probe, _ = seeded

            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="crash_exploit",
                eval_turn_fn=lambda turns, desired_arrival=desired_arrival: max(turns, desired_arrival),
                anchor_turn=desired_arrival,
                anchor_tolerance=CRASH_EXPLOIT_ETA_WINDOW,
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                continue
            value = target_value(target, turns, "crash_exploit", world, modes, policy)
            if value <= 0:
                continue

            score = apply_score_modifiers(
                value / (send_pref + turns * SNIPE_COST_TURN_WEIGHT + 1.0),
                target,
                "crash_exploit",
                world,
            )
            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="crash_exploit",
                anchor_turn=desired_arrival,
            )
            missions.append(
                Mission(
                    kind="crash_exploit",
                    score=score,
                    target_id=target.id,
                    turns=turns,
                    options=[option],
                )
            )

    return missions


def plan_moves(world, deadline=None):
    def expired():
        return deadline is not None and time.perf_counter() > deadline

    def time_left():
        if deadline is None:
            return 10**9
        return deadline - time.perf_counter()

    def allow_heavy_phase():
        return time_left() > HEAVY_PHASE_MIN_TIME and len(world.planets) <= HEAVY_ROUTE_PLANET_LIMIT

    def allow_optional_phase():
        return time_left() > OPTIONAL_PHASE_MIN_TIME

    modes = build_modes(world)
    policy = build_policy_state(world, deadline=deadline)
    planned_commitments = defaultdict(list)
    source_options_by_target = defaultdict(list)
    missions = []
    moves = []
    spent_total = defaultdict(int)

    def source_inventory_left(source_id):
        return world.source_inventory_left(source_id, spent_total)

    def source_attack_left(source_id):
        budget = policy["attack_budget"].get(source_id, 0)
        return max(0, budget - spent_total[source_id])

    def append_move(src_id, angle, ships):
        send = min(int(ships), source_inventory_left(src_id))
        if send < 1:
            return 0
        moves.append([src_id, float(angle), int(send)])
        spent_total[src_id] += send
        return send

    def finalize_moves():
        final_moves = []
        used_final = defaultdict(int)
        for src_id, angle, ships in moves:
            source = world.planet_by_id[src_id]
            max_allowed = int(source.ships) - used_final[src_id]
            send = min(int(ships), max_allowed)
            if send >= 1:
                final_moves.append([src_id, float(angle), int(send)])
                used_final[src_id] += send
        return final_moves

    def compute_live_doomed():
        doomed = set()
        for planet in world.my_planets:
            status = world.hold_status(
                planet.id,
                planned_commitments=planned_commitments,
                horizon=DOOMED_EVAC_TURN_LIMIT,
            )
            if (
                not status["holds_full"]
                and status["fall_turn"] is not None
                and status["fall_turn"] <= DOOMED_EVAC_TURN_LIMIT
                and source_inventory_left(planet.id) >= DOOMED_MIN_SHIPS
            ):
                doomed.add(planet.id)
        return doomed

    def time_filters_pass(target, turns, needed, src_cap):
        if not candidate_time_valid(target, turns, world, VERY_LATE_CAPTURE_BUFFER if world.is_very_late else LATE_CAPTURE_BUFFER):
            return False
        if opening_filter(target, turns, needed, src_cap, world, policy):
            return False
        return True

    if allow_heavy_phase():
        missions.extend(
            build_reinforce_missions(
                world,
                policy,
                planned_commitments,
                modes,
                source_inventory_left,
            )
        )
    missions.extend(build_rescue_missions(world, policy, planned_commitments, modes))
    missions.extend(build_recapture_missions(world, policy, planned_commitments, modes))

    # Only build candidates after solving an intercept so timing decisions come
    # from a real route.
    for src in world.my_planets:
        if expired():
            return finalize_moves()
        src_available = source_attack_left(src.id)
        if src_available <= 0:
            continue

        for target in world.planets:
            if expired():
                return finalize_moves()
            if target.id == src.id or target.owner == world.player:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(int(target.ships) + 1,),
            )
            if seeded is None:
                continue
            _, rough_aim = seeded

            rough_turns = rough_aim[1]
            if not candidate_time_valid(
                target,
                rough_turns,
                world,
                VERY_LATE_CAPTURE_BUFFER if world.is_very_late else LATE_CAPTURE_BUFFER,
            ):
                continue

            global_needed = world.min_ships_to_own_at(
                target.id,
                rough_turns,
                world.player,
                planned_commitments=planned_commitments,
            )
            if global_needed <= 0:
                continue
            if opening_filter(target, rough_turns, global_needed, src_available, world, policy):
                continue

            partial_send_cap = min(
                src_available,
                preferred_send(
                    target,
                    global_needed,
                    rough_turns,
                    src_available,
                    world,
                    modes,
                    policy,
                ),
            )
            if partial_send_cap >= PARTIAL_SOURCE_MIN_SHIPS:
                partial_seed = world.best_probe_aim(
                    src.id,
                    target.id,
                    partial_send_cap,
                    hints=(partial_send_cap, global_needed, int(target.ships) + 1),
                )
                if partial_seed is not None:
                    _, partial_aim = partial_seed
                    p_angle, p_turns, _, _ = partial_aim
                    if time_filters_pass(target, p_turns, global_needed, src_available):
                        partial_value = target_value(target, p_turns, "swarm", world, modes, policy)
                        if partial_value > 0:
                            partial_score = apply_score_modifiers(
                                partial_value / (partial_send_cap + p_turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                                target,
                                "swarm",
                                world,
                            )
                            source_options_by_target[target.id].append(
                                ShotOption(
                                    score=partial_score,
                                    src_id=src.id,
                                    target_id=target.id,
                                    angle=p_angle,
                                    turns=p_turns,
                                    needed=global_needed,
                                    send_cap=partial_send_cap,
                                    mission="swarm",
                                )
                            )

            if global_needed <= src_available:
                send_guess = preferred_send(
                    target,
                    global_needed,
                    rough_turns,
                    src_available,
                    world,
                    modes,
                    policy,
                )
                plan = settle_plan(
                    src,
                    target,
                    src_available,
                    send_guess,
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )
                if plan is None:
                    continue

                angle, turns, _, needed, send_cap = plan
                if not time_filters_pass(target, turns, needed, src_available):
                    continue
                if send_cap < 1:
                    continue

                value = target_value(target, turns, "capture", world, modes, policy)
                if value <= 0:
                    continue

                score = apply_score_modifiers(
                    value / (send_cap + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "capture",
                    world,
                )

                option = ShotOption(
                    score=score,
                    src_id=src.id,
                    target_id=target.id,
                    angle=angle,
                    turns=turns,
                    needed=needed,
                    send_cap=send_cap,
                    mission="capture",
                )

                if send_cap >= needed:
                    missions.append(
                        Mission(
                            kind="single",
                            score=score,
                            target_id=target.id,
                            turns=turns,
                            options=[option],
                        )
                    )

            snipe = build_snipe_mission(src, target, src_available, world, planned_commitments, modes, policy)
            if snipe is not None:
                missions.append(snipe)

    # Allow small synchronized two-source finishes when one source is not
    # enough on its own.
    for target_id, options in source_options_by_target.items():
        if expired():
            return finalize_moves()
        if len(options) < 2:
            continue

        target = world.planet_by_id[target_id]
        top_options = sorted(options, key=lambda item: -item.score)[:MULTI_SOURCE_TOP_K]
        for i in range(len(top_options)):
            for j in range(i + 1, len(top_options)):
                first = top_options[i]
                second = top_options[j]
                if first.src_id == second.src_id:
                    continue
                pair_tol = swarm_eta_tolerance((first, second), target, world)
                if abs(first.turns - second.turns) > pair_tol:
                    continue

                joint_turn = max(first.turns, second.turns)
                total_cap = first.send_cap + second.send_cap
                need = world.min_ships_to_own_at(
                    target_id,
                    joint_turn,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=total_cap,
                )
                if need <= 0:
                    continue
                if first.send_cap >= need or second.send_cap >= need:
                    continue
                if total_cap < need:
                    continue

                value = target_value(target, joint_turn, "swarm", world, modes, policy)
                if value <= 0:
                    continue

                pair_score = apply_score_modifiers(
                    value / (need + joint_turn * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "swarm",
                    world,
                )
                pair_score *= MULTI_SOURCE_PLAN_PENALTY
                missions.append(
                    Mission(
                        kind="swarm",
                        score=pair_score,
                        target_id=target_id,
                        turns=joint_turn,
                        options=[first, second],
                    )
                )

        if (
            THREE_SOURCE_SWARM_ENABLED
            and allow_heavy_phase()
            and target.owner not in (-1, world.player)
            and int(target.ships) >= THREE_SOURCE_MIN_TARGET_SHIPS
            and len(top_options) >= 3
        ):
            for i in range(len(top_options)):
                for j in range(i + 1, len(top_options)):
                    for k in range(j + 1, len(top_options)):
                        if expired():
                            return finalize_moves()
                        trio = [top_options[i], top_options[j], top_options[k]]
                        if len({option.src_id for option in trio}) < 3:
                            continue
                        trio_tol = swarm_eta_tolerance(tuple(trio), target, world)
                        turns = [option.turns for option in trio]
                        if max(turns) - min(turns) > trio_tol:
                            continue

                        joint_turn = max(turns)
                        total_cap = sum(option.send_cap for option in trio)
                        need = world.min_ships_to_own_at(
                            target_id,
                            joint_turn,
                            world.player,
                            planned_commitments=planned_commitments,
                            upper_bound=total_cap,
                        )
                        if need <= 0 or total_cap < need:
                            continue
                        if any(
                            trio[a].send_cap + trio[b].send_cap >= need
                            for a in range(3)
                            for b in range(a + 1, 3)
                        ):
                            continue

                        value = target_value(target, joint_turn, "swarm", world, modes, policy)
                        if value <= 0:
                            continue

                        trio_score = apply_score_modifiers(
                            value / (need + joint_turn * ATTACK_COST_TURN_WEIGHT + 1.0),
                            target,
                            "swarm",
                            world,
                        )
                        trio_score *= THREE_SOURCE_PLAN_PENALTY
                        missions.append(
                            Mission(
                                kind="swarm",
                                score=trio_score,
                                target_id=target_id,
                                turns=joint_turn,
                                options=trio,
                            )
                        )

    if allow_heavy_phase():
        missions.extend(build_crash_exploit_missions(world, policy, planned_commitments, modes))

    missions.sort(key=lambda item: -item.score)

    # Update commitments after every accepted launch so later plans see the
    # timing that is already spoken for.
    for mission in missions:
        if expired():
            return finalize_moves()
        target = world.planet_by_id[mission.target_id]

        if mission.kind in ("single", "snipe", "rescue", "recapture", "reinforce", "crash_exploit"):
            option = mission.options[0]
            src = world.planet_by_id[option.src_id]
            if mission.kind == "reinforce":
                left = min(
                    source_inventory_left(option.src_id),
                    int(src.ships * REINFORCE_MAX_SOURCE_FRACTION),
                )
            else:
                left = source_attack_left(option.src_id)
            if left <= 0:
                continue

            if mission.kind == "reinforce":
                plan = settle_reinforce_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    option.anchor_turn,
                    mission.turns,
                )
            elif mission.kind == "rescue":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="rescue",
                    eval_turn_fn=lambda _turns, hold_turn=mission.turns: hold_turn,
                    anchor_turn=option.anchor_turn,
                )
            elif mission.kind == "snipe":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="snipe",
                    eval_turn_fn=lambda turns, enemy_eta=option.anchor_turn: max(turns, enemy_eta),
                    anchor_turn=option.anchor_turn,
                )
            elif mission.kind == "crash_exploit":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="crash_exploit",
                    eval_turn_fn=lambda turns, desired_arrival=option.anchor_turn: max(turns, desired_arrival),
                    anchor_turn=option.anchor_turn,
                    anchor_tolerance=CRASH_EXPLOIT_ETA_WINDOW,
                )
            else:
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )
            if plan is None:
                continue

            angle, turns, _, need, send = plan
            if send < need or need > left:
                continue

            sent = append_move(option.src_id, angle, send)
            if sent < need:
                continue
            planned_commitments[target.id].append((turns, world.player, int(sent)))
            continue

        limits = []
        for option in mission.options:
            left = source_attack_left(option.src_id)
            limits.append(min(left, option.send_cap))
        if min(limits) <= 0:
            continue

        missing = world.min_ships_to_own_at(
            target.id,
            mission.turns,
            world.player,
            planned_commitments=planned_commitments,
            upper_bound=sum(limits),
        )
        if missing <= 0 or sum(limits) < missing:
            continue

        ordered = sorted(
            zip(mission.options, limits),
            key=lambda item: (item[0].turns, -item[1], item[0].src_id),
        )
        remaining = missing
        sends = {}
        for idx, (option, limit) in enumerate(ordered):
            remaining_other = sum(other_limit for _, other_limit in ordered[idx + 1 :])
            send = min(limit, max(0, remaining - remaining_other))
            sends[option.src_id] = send
            remaining -= send
        if remaining > 0:
            continue

        reaimed = []
        for option, _ in ordered:
            send = sends.get(option.src_id, 0)
            if send <= 0:
                continue
            src = world.planet_by_id[option.src_id]
            fixed_aim = world.plan_shot(src.id, target.id, send)
            if fixed_aim is None:
                reaimed = []
                break
            angle, turns, _, _ = fixed_aim
            reaimed.append((option.src_id, angle, turns, send))
        if not reaimed:
            continue

        turns_only = [item[2] for item in reaimed]
        eta_tol = swarm_eta_tolerance(mission.options, target, world)
        if max(turns_only) - min(turns_only) > eta_tol:
            continue

        actual_joint_turn = max(turns_only)
        owner_after, _ = world.projected_state(
            target.id,
            actual_joint_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=[(turns, world.player, send) for _, _, turns, send in reaimed],
        )
        if owner_after != world.player:
            continue

        committed = []
        for src_id, angle, turns, send in reaimed:
            actual = append_move(src_id, angle, send)
            if actual <= 0:
                continue
            committed.append((turns, world.player, int(actual)))
        if sum(item[2] for item in committed) < missing:
            continue
        planned_commitments[target.id].extend(committed)

    # Use leftover attack budget for one more pass after the first commitment
    # wave is fixed.
    if not world.is_very_late and allow_optional_phase():
        for src in world.my_planets:
            if expired():
                return finalize_moves()
            src_left = source_attack_left(src.id)
            if src_left < FOLLOWUP_MIN_SHIPS:
                continue

            best = None
            for target in world.planets:
                if expired():
                    return finalize_moves()
                if target.id == src.id or target.owner == world.player:
                    continue
                if target.id in world.comet_ids and target.production <= LOW_VALUE_COMET_PRODUCTION:
                    continue

                seeded = world.best_probe_aim(
                    src.id,
                    target.id,
                    src_left,
                    hints=(int(target.ships) + 1,),
                )
                if seeded is None:
                    continue
                rough_ships, rough_aim = seeded

                est_turns = rough_aim[1]
                if world.is_late and est_turns > world.remaining_steps - LATE_CAPTURE_BUFFER:
                    continue

                rough_needed = world.min_ships_to_own_at(
                    target.id,
                    est_turns,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=src_left,
                )
                if rough_needed <= 0 or rough_needed > src_left:
                    continue
                if opening_filter(target, est_turns, rough_needed, src_left, world, policy):
                    continue

                send = preferred_send(target, rough_needed, est_turns, src_left, world, modes, policy)
                if send < rough_needed:
                    continue

                plan = settle_plan(
                    src,
                    target,
                    src_left,
                    send,
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )
                if plan is None:
                    continue

                _, turns, _, need, final_send = plan
                if world.is_late and turns > world.remaining_steps - LATE_CAPTURE_BUFFER:
                    continue
                if final_send < need:
                    continue

                value = target_value(target, turns, "capture", world, modes, policy)
                if value <= 0:
                    continue

                score = apply_score_modifiers(
                    value / (final_send + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "capture",
                    world,
                )
                if best is None or score > best[0]:
                    best = (score, target, plan)

            if best is None:
                continue

            _, target, plan = best
            angle, turns, _, need, send = plan
            src_left = source_attack_left(src.id)
            if need > src_left:
                continue

            plan = settle_plan(
                src,
                target,
                src_left,
                min(src_left, send),
                world,
                planned_commitments,
                modes,
                policy,
                mission="capture",
            )
            if plan is None:
                continue

            angle, turns, _, need, send = plan
            if send < need:
                continue

            actual = append_move(src.id, angle, send)
            if actual < need:
                continue
            planned_commitments[target.id].append((turns, world.player, int(actual)))

    # If a planet cannot hold soon, prefer reinforcement first. For stacks that
    # still look doomed after the main mission pass, prefer a last useful
    # capture; otherwise retreat the stack to a safer ally.
    if expired():
        return finalize_moves()
    live_doomed = compute_live_doomed()
    if live_doomed:
        frontier_targets = (
            world.enemy_planets
            if world.enemy_planets
            else (world.static_neutral_planets or world.neutral_planets)
        )
        if frontier_targets:
            frontier_distance = {
                planet.id: nearest_distance_to_set(planet.x, planet.y, frontier_targets)
                for planet in world.my_planets
            }
        else:
            frontier_distance = {planet.id: 10**9 for planet in world.my_planets}

        for planet in world.my_planets:
            if expired():
                return finalize_moves()
            if planet.id not in live_doomed:
                continue

            available_now = source_inventory_left(planet.id)
            if available_now < policy["reserve"].get(planet.id, 0):
                continue

            best_capture = None
            for target in world.planets:
                if expired():
                    return finalize_moves()
                if target.id == planet.id or target.owner == world.player:
                    continue
                seeded = world.best_probe_aim(
                    planet.id,
                    target.id,
                    available_now,
                    hints=(available_now, int(target.ships) + 1),
                )
                if seeded is None:
                    continue
                _, probe_aim = seeded
                probe_turns = probe_aim[1]
                if probe_turns > world.remaining_steps - 2:
                    continue

                need = world.min_ships_to_own_at(
                    target.id,
                    probe_turns,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=available_now,
                )
                if need <= 0 or need > available_now:
                    continue

                plan = settle_plan(
                    planet,
                    target,
                    available_now,
                    min(available_now, max(need, int(target.ships) + 1)),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )
                if plan is None:
                    continue
                angle, turns, _, plan_need, send = plan
                if send < plan_need:
                    continue
                score = target_value(target, turns, "capture", world, modes, policy) / (send + turns + 1.0)
                if target.owner not in (-1, world.player):
                    score *= 1.05
                if best_capture is None or score > best_capture[0]:
                    best_capture = (score, target.id, angle, turns, send)

            if best_capture is not None:
                _, target_id, angle, turns, need = best_capture
                actual = append_move(planet.id, angle, need)
                if actual >= 1:
                    planned_commitments[target_id].append((turns, world.player, int(actual)))
                continue

            safe_allies = [
                ally
                for ally in world.my_planets
                if ally.id != planet.id and ally.id not in live_doomed
            ]
            if not safe_allies:
                continue

            retreat_target = min(
                safe_allies,
                key=lambda ally: (
                    frontier_distance.get(ally.id, 10**9),
                    planet_distance(planet, ally),
                ),
            )
            aim = world.plan_shot(planet.id, retreat_target.id, available_now)
            if aim is None:
                continue
            angle, _, _, _ = aim
            append_move(planet.id, angle, available_now)

    # Rear planets feed the frontier through staging allies instead of acting
    # as slow solo attackers.
    if (
        (world.enemy_planets or world.neutral_planets)
        and len(world.my_planets) > 1
        and not world.is_late
        and allow_optional_phase()
    ):
        live_doomed = compute_live_doomed()
        frontier_targets = (
            world.enemy_planets
            if world.enemy_planets
            else (world.static_neutral_planets or world.neutral_planets)
        )
        frontier_distance = {
            planet.id: nearest_distance_to_set(planet.x, planet.y, frontier_targets)
            for planet in world.my_planets
        }
        safe_fronts = [
            planet for planet in world.my_planets if planet.id not in live_doomed
        ]
        if safe_fronts:
            front_anchor = min(safe_fronts, key=lambda planet: frontier_distance[planet.id])
            send_ratio = (
                REAR_SEND_RATIO_FOUR_PLAYER if world.is_four_player else REAR_SEND_RATIO_TWO_PLAYER
            )
            if modes["is_finishing"]:
                send_ratio = max(send_ratio, REAR_SEND_RATIO_FOUR_PLAYER)

            for rear in sorted(world.my_planets, key=lambda planet: -frontier_distance[planet.id]):
                if expired():
                    return finalize_moves()
                if rear.id == front_anchor.id or rear.id in live_doomed:
                    continue
                if source_attack_left(rear.id) < REAR_SOURCE_MIN_SHIPS:
                    continue
                if frontier_distance[rear.id] < frontier_distance[front_anchor.id] * REAR_DISTANCE_RATIO:
                    continue

                stage_candidates = [
                    planet
                    for planet in safe_fronts
                    if planet.id != rear.id
                    and frontier_distance[planet.id] < frontier_distance[rear.id] * REAR_STAGE_PROGRESS
                ]
                if stage_candidates:
                    front = min(
                        stage_candidates,
                        key=lambda planet: planet_distance(rear, planet),
                    )
                else:
                    objective = min(
                        frontier_targets,
                        key=lambda target: planet_distance(rear, target),
                    )
                    remaining_fronts = [planet for planet in safe_fronts if planet.id != rear.id]
                    if not remaining_fronts:
                        continue
                    front = min(
                        remaining_fronts,
                        key=lambda planet: planet_distance(planet, objective),
                    )

                if front.id == rear.id:
                    continue

                send = int(source_attack_left(rear.id) * send_ratio)
                if send < REAR_SEND_MIN_SHIPS:
                    continue

                aim = world.plan_shot(rear.id, front.id, send)
                if aim is None:
                    continue

                angle, turns, _, _ = aim
                if turns > REAR_MAX_TRAVEL_TURNS:
                    continue
                append_move(rear.id, angle, send)

    if world.is_total_war and world.enemy_planets and allow_optional_phase():
        def enemy_priority(planet):
            blood_in_water = planet.owner in world.blood_in_water_owners
            strength = world.owner_strength.get(planet.owner, 10**9)
            return (0 if blood_in_water else 1, strength, -int(planet.production))

        priority_targets = sorted(world.enemy_planets, key=enemy_priority)
        for src in world.my_planets:
            if expired():
                return finalize_moves()
            left = source_attack_left(src.id)
            if left < TOTAL_WAR_MIN_SEND:
                continue
            chosen = None
            for target in priority_targets:
                aim = world.plan_shot(src.id, target.id, left)
                if aim is None:
                    continue
                if aim[1] >= world.remaining_steps:
                    continue
                chosen = (target.id, aim)
                break
            if chosen is None:
                for target in sorted(world.enemy_planets, key=lambda planet: planet_distance(src, planet)):
                    aim = world.plan_shot(src.id, target.id, left)
                    if aim is not None and aim[1] < world.remaining_steps:
                        chosen = (target.id, aim)
                        break
            if chosen is None:
                continue
            target_id, (angle, turns, _, _) = chosen
            sent = append_move(src.id, angle, left)
            if sent >= 1:
                planned_commitments[target_id].append((turns, world.player, int(sent)))

    return finalize_moves()

# ============================================================
# Agent Entry Point
# ============================================================

def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def build_world(obs):
    player = _read(obs, "player", 0)
    step = _read(obs, "step", 0) or 0
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    ang_vel = _read(obs, "angular_velocity", 0.0) or 0.0
    raw_init = _read(obs, "initial_planets", []) or []
    comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])

    planets = [Planet(*planet) for planet in raw_planets]
    fleets = [Fleet(*fleet) for fleet in raw_fleets]
    initial_planets = [Planet(*planet) for planet in raw_init]
    initial_by_id = {planet.id: planet for planet in initial_planets}

    return WorldModel(
        player=player,
        step=step,
        planets=planets,
        fleets=fleets,
        initial_by_id=initial_by_id,
        ang_vel=ang_vel,
        comets=comets,
        comet_ids=comet_ids,
    )


def _v4_agent_internal(obs, config=None):
    start_time = time.perf_counter()
    world = build_world(obs)
    if not world.my_planets:
        return []
    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    soft_budget = min(SOFT_ACT_DEADLINE, max(0.55, act_timeout * 0.82))
    deadline = start_time + soft_budget
    return plan_moves(world, deadline=deadline)








# ---- Hybrid entrypoint ----
def agent(obs, config=None):
    moves = _v4_agent_internal(obs, config)
    if not moves or _VALIDATOR is None:
        return moves
    side = int(obs.get("player", 0))
    planets = obs["planets"]
    owner_by_id = {}
    src_xy = {}
    for p in planets:
        pid = int(p[0])
        owner_by_id[pid] = int(p[1])
        src_xy[pid] = (float(p[2]), float(p[3]))
    feats = []
    idxs = []
    for i, mv in enumerate(moves):
        try:
            src_id = int(mv[0]); ang = float(mv[1]); ships = int(mv[2])
        except Exception:
            continue
        if src_id not in src_xy:
            continue
        tgt_id = _find_target_ray(src_xy[src_id], ang, planets)
        if tgt_id < 0 or tgt_id == src_id:
            continue
        if owner_by_id.get(tgt_id, -2) == side:
            continue  # own-planet reinforcement: always keep
        feat = _encode_shot_np(obs, src_id, tgt_id, ships)
        if feat is None:
            continue
        feats.append(feat); idxs.append(i)
    if not feats:
        return moves
    x = _np_hybrid.stack(feats)
    probs = _VALIDATOR.proba(x)
    keep = [True] * len(moves)
    for i, prob in zip(idxs, probs):
        if prob < _VAL_THRESHOLD:
            keep[i] = False
    return [mv for i, mv in enumerate(moves) if keep[i]]

__all__ = ["agent"]

```

    Writing submission.py
    

## 6. Sanity check

Make sure `submission.py` parses cleanly and the agent function is callable.



```python
# Sanity: ensure the submission imports cleanly with weights present
import importlib.util, pathlib
assert pathlib.Path("weights.npz").exists(), "weights.npz must exist before importing"
spec = importlib.util.spec_from_file_location("submission", "submission.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("agent:", m.agent)
print("validator loaded:", m._VALIDATOR is not None)
print("threshold:", m._VAL_THRESHOLD)

```

    agent: <function agent at 0x79d7f5b43740>
    validator loaded: True
    threshold: 0.4
    

## 7. Submit

To submit: download both `submission.py` and `weights.npz` from this notebook, package
them into a single tar.gz with both files at the **root** (no enclosing folder), and
upload via the *Submit Predictions* button on the competition page.

```bash
tar -czf submission.tar.gz submission.py weights.npz
# then upload submission.tar.gz
```

If you fork and re-run this notebook end to end, the two files will be sitting in the
working directory ready to download.


## 8. What did *not* work (so you don't repeat it)

Five separate ML directions before this hybrid hit the same tier3+ wall:

- **PPO from random init** with a structured-policy network — collapsed to no-op around
  update ~80, classic dense-shaping trap.
- **PPO with curriculum** (sniper → tier1 → tier2 → tier3 → mixed) — survived longer but
  still 0% on tier3 in eval.
- **PPO with smoother 5-phase curriculum + lower lr + lower shaping coef** — entropy stayed
  healthy through training, still 0% on tier3.
- **SFT (single teacher, orbitbotnext)** — got to 75% vs sniper but tier3+ still 0%.
- **SFT (multi-teacher: orbitbotnext + v4 + exp004_a)** — val pos_acc went *up* (20% → 34%)
  but in-game wr collapsed to **2% overall** (sniper 12%, everything else 0%). Conflicting
  teacher labels averaged out into a policy that hesitated everywhere.

The lesson that pushed us toward the validator hybrid: when the rule-base ceiling and the
ML floor don't intersect, don't try to make ML stand alone — let it *edit* the rule-base.

A few smaller dead ends:

- **Hand-tuning the rule-base constants** (HOSTILE_REINFORCE, attack-cost, rotating-opening
  thresholds): every change that helped against one opponent hurt against another. The
  rule-base has already absorbed the easy wins from constant tuning.
- **Threshold 0.5 for the validator**: rejects too many shots, hybrid wr drops to 57%.
  More-rejection is not always better.


## 9. Acknowledgements

- [Pilkwang Kim](https://www.kaggle.com/pilkwang) — `Orbit Wars: Structured Baseline` (the common ancestor of the rule-base used here)
- [Roman Tamrazov](https://www.kaggle.com/romantamrazov) — public structured-baseline derivative
- [Yegor Khnykin](https://www.kaggle.com/ykhnkf) — public hybrid lineage
- [Kaggle Simulations](https://www.kaggle.com/c/orbit-wars) — for the competition format that allows quick local benchmarking against forks of public agents

If this notebook saves you a few days of PPO debugging, please consider an upvote 🛰️

