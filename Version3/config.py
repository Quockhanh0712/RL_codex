from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Any

# === Game Constants ===
P_MAX: int = 64
F_MAX: int = 256
BOARD_SIZE: float = 100.0
CENTER: float = 50.0
SUN_RADIUS: float = 10.0
MAX_SHIP_SPEED: float = 6.0
ROT_RADIUS_LIMIT: float = 50.0
OWN: int = 0
ENEMY: int = 1
NEUTRAL: int = 2
DEAD: int = 3
LIBRARY_K_DEFAULT: int = 100000
COMET_EVENTS: int = 5
COMETS_PER_EVENT: int = 4
COMET_PATH_MAX: int = 40
COMET_SPAWN_STEPS: tuple[int, ...] = (50, 150, 250, 350, 450)
COMET_RADIUS: float = 1.0
COMET_PRODUCTION: float = 1.0
EARLY_TERM_MARGIN: float = 2.0
EARLY_TERM_STREAK_2P: int = 5
EARLY_TERM_STREAK_4P: int = 20
EARLY_TERM_PROD_WEIGHT_2P: float = 5.0
EARLY_TERM_SHIP_WEIGHT_2P: float = 1.0
EARLY_TERM_PROD_WEIGHT_4P: float = 1.0
EARLY_TERM_SHIP_WEIGHT_4P: float = 0.0
DEFAULT_EPISODE_STEPS: int = 500

# === Movement/Kinematics Constants ===
DEFAULT_MOVEMENT_HORIZON = 20
DEFAULT_DRIFT_EPSILON = 0.0001
DEFAULT_MAX_TRACKED_FLEETS = 64
_FP_ITERS = 6
LAUNCH_SURFACE_OFFSET: float = 0.1
TARGET_HIT_SURFACE_OFFSET: float = 0.0
KAGGLE_SUN_RADIUS: float = SUN_RADIUS

# === MovementConfig dataclass (L170-176) ===
@dataclass(frozen=True)
class MovementConfig:
    movement_horizon: int = DEFAULT_MOVEMENT_HORIZON
    drift_epsilon: float = DEFAULT_DRIFT_EPSILON
    track_fleets: bool = False
    player_count: int | None = None
    max_tracked_fleets: int = DEFAULT_MAX_TRACKED_FLEETS


# === ProducerLiteConfig dataclass (L2225-2248) ===
@dataclass(frozen=True)
class ProducerLiteConfig:
    horizon: int = 18
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12
    max_defensive_targets: int = 4
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5
    min_ships_to_launch: float = 4.0
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = 'none'
    regroup_time_penalty_weight: float = 0.001
    enable_potential_risk: bool = False
    risk_blend_weight: float = 1.0
    risk_enemy_prod_weight: float = 2.0
    risk_self_prod_weight: float = 2.0
    risk_support_weight: float = 0.5
    enable_focus_fire: bool = True
    max_strike_sources: int = 4



# === CONFIG_4P (L2467) ===
CONFIG_4P = dataclasses.replace(ProducerLiteConfig(), horizon=13, max_sources_per_lane=6, max_defensive_targets=2, max_regroup_time=6.0, max_regroup_targets_per_source=8, risk_blend_weight=0.5, max_strike_sources=3)


# === Config helpers ===
def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    return MovementConfig(movement_horizon=int(config.horizon), track_fleets=True, player_count=int(player_count))

def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()
