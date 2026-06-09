from __future__ import annotations
import torch
from torch import Tensor
from typing import Any
from .config import (
    ProducerLiteConfig, _movement_config, _config_for,
    P_MAX, F_MAX, DEFAULT_EPISODE_STEPS,
    COMET_EVENTS, COMETS_PER_EVENT, COMET_PATH_MAX
)
from .movement import parse_obs, ensure_planet_movement, fleet_speed
from .kinematics import build_distance_cache
from .planner import (
    plan_lite_waves, disambiguate_duplicate_launches, infer_planned_launches_from_entries,
    apply_private_planned_launches, largest_initial_player_count,
    entries_to_sparse_payload
)

# === empty_action_row (L2080-2082) ===
def empty_action_row(device: torch.device) -> dict[str, Tensor]:
    return {'from_planet_id': torch.full((0,), -1, dtype=torch.int32, device=device), 'angle': torch.zeros((0,), dtype=torch.float32, device=device), 'num_ships': torch.zeros((0,), dtype=torch.float32, device=device), 'counts': torch.zeros((), dtype=torch.int32, device=device)}



# === Obs tensor conversion (L2108-2206) ===
def _infer_player_count_from_obs(planets: list[Any], fleets: list[Any], player_id: int) -> int:
    owners: list[int] = [int(player_id)]
    for row in planets:
        if len(row) >= 2 and int(row[0]) >= 0 and (int(row[1]) >= 0):
            owners.append(int(row[1]))
    for row in fleets:
        if len(row) >= 2 and int(row[0]) >= 0 and (int(row[1]) >= 0):
            owners.append(int(row[1]))
    return 4 if max(owners, default=0) >= 2 else 2

def _dict_obs_to_tensor(obs: dict[str, Any], player_id: int, P: int=P_MAX, F: int=F_MAX, device: Any='cpu') -> dict[str, Any]:
    dev = torch.device(device)
    planets_raw = obs.get('planets', [])
    initial_planets_raw = obs.get('initial_planets', planets_raw)
    fleets_raw = obs.get('fleets', [])
    comets_raw = obs.get('comets', [])
    comet_planet_ids_raw = obs.get('comet_planet_ids', [])
    step = int(obs.get('step', 0))
    angvel = float(obs.get('angular_velocity', 0.03))
    max_steps = int(obs.get('episode_steps', DEFAULT_EPISODE_STEPS))
    remaining_overtime = float(obs.get('remainingOverageTime', 2.0))
    next_fleet_id = int(obs.get('next_fleet_id', 0))
    planet_t = torch.zeros(P, 7, dtype=torch.float32, device=dev)
    planet_t[..., 0] = -1.0
    for i, p in enumerate(planets_raw[:P]):
        pid, owner, x, y, r, ships, prod = p[:7]
        planet_t[i, 0] = float(pid)
        planet_t[i, 1] = float(owner)
        planet_t[i, 2] = float(x)
        planet_t[i, 3] = float(y)
        planet_t[i, 4] = float(r)
        planet_t[i, 5] = float(ships)
        planet_t[i, 6] = float(prod)
    initial_planet_t = torch.zeros(P, 7, dtype=torch.float32, device=dev)
    initial_planet_t[..., 0] = -1.0
    for i, p in enumerate(initial_planets_raw[:P]):
        pid, owner, x, y, r, ships, prod = p[:7]
        initial_planet_t[i, 0] = float(pid)
        initial_planet_t[i, 1] = float(owner)
        initial_planet_t[i, 2] = float(x)
        initial_planet_t[i, 3] = float(y)
        initial_planet_t[i, 4] = float(r)
        initial_planet_t[i, 5] = float(ships)
        initial_planet_t[i, 6] = float(prod)
    fleet_t = torch.zeros(F, 7, dtype=torch.float32, device=dev)
    fleet_t[..., 0] = -1.0
    fleet_t[..., 5] = -1.0
    for i, f in enumerate(fleets_raw[:F]):
        fid, owner, x, y, angle, from_id, ships = f[:7]
        fleet_t[i, 0] = float(fid)
        fleet_t[i, 1] = float(owner)
        fleet_t[i, 2] = float(x)
        fleet_t[i, 3] = float(y)
        fleet_t[i, 4] = float(angle)
        fleet_t[i, 5] = float(from_id)
        fleet_t[i, 6] = float(ships)
    comet_ids = torch.full((COMET_EVENTS, COMETS_PER_EVENT), -1, dtype=torch.int32, device=dev)
    comet_paths = torch.full((COMET_EVENTS, COMETS_PER_EVENT, COMET_PATH_MAX, 2), float('nan'), dtype=torch.float32, device=dev)
    comet_path_index = torch.full((COMET_EVENTS,), -1, dtype=torch.int32, device=dev)
    for group_idx, group in enumerate(comets_raw[:COMET_EVENTS]):
        comet_path_index[group_idx] = int(group.get('path_index', -1))
        group_ids = group.get('planet_ids', [])
        group_paths = group.get('paths', [])
        for comet_idx, pid in enumerate(group_ids[:COMETS_PER_EVENT]):
            comet_ids[group_idx, comet_idx] = int(pid)
        for comet_idx, path in enumerate(group_paths[:COMETS_PER_EVENT]):
            for point_idx, point in enumerate(path[:COMET_PATH_MAX]):
                comet_paths[group_idx, comet_idx, point_idx, 0] = float(point[0])
                comet_paths[group_idx, comet_idx, point_idx, 1] = float(point[1])
    comet_planet_ids = torch.full((COMET_EVENTS * COMETS_PER_EVENT,), -1, dtype=torch.int32, device=dev)
    for idx, pid in enumerate(comet_planet_ids_raw[:COMET_EVENTS * COMETS_PER_EVENT]):
        comet_planet_ids[idx] = int(pid)
    return {'planets': planet_t, 'fleets': fleet_t, 'player': torch.tensor(player_id, dtype=torch.int32, device=dev), 'player_count': torch.tensor(_infer_player_count_from_obs(planets_raw, fleets_raw, player_id), dtype=torch.int32, device=dev), 'angular_velocity': torch.tensor(angvel, dtype=torch.float32, device=dev), 'initial_planets': initial_planet_t, 'next_fleet_id': torch.tensor(next_fleet_id, dtype=torch.int32, device=dev), 'comets': {'planet_ids': comet_ids, 'paths': comet_paths, 'path_index': comet_path_index}, 'comet_planet_ids': comet_planet_ids, 'step': torch.tensor(step, dtype=torch.int32, device=dev), 'episode_steps': torch.tensor(max_steps, dtype=torch.int32, device=dev), 'remainingOverageTime': torch.tensor(remaining_overtime, dtype=torch.float32, device=dev)}

def _sparse_actions_to_list(action_payload: dict[str, Any], obs: dict[str, Any], player_id: int) -> list[list[Any]]:
    from_pid_t = action_payload['from_planet_id']
    angle_t = action_payload['angle']
    num_ships_t = action_payload['num_ships']
    counts = int(action_payload['counts'].item())
    planets_by_id = {int(p[0]): p for p in obs.get('planets', []) if len(p) >= 7}
    moves: list[list[Any]] = []
    for launch_idx in range(counts):
        from_pid = int(from_pid_t[launch_idx].item())
        ships = float(num_ships_t[launch_idx].item())
        angle = float(angle_t[launch_idx].item())
        if ships < 1.0:
            continue
        source = planets_by_id.get(from_pid)
        if source is None:
            continue
        owner = int(source[1])
        available = float(source[5])
        if owner != int(player_id):
            continue
        if ships != float(round(ships)) or ships > available:
            raise ValueError(f'Invalid launch ship count in sparse action payload at from_planet_id={from_pid}: requested={ships}, available={available}. Counts must be finite, integer-valued, >= 0, and <= available planet ships.')
        moves.append([from_pid, angle, int(ships)])
    return moves



# === single_obs_to_tensor + sparse_action_row_to_moves (L2207-2224) ===
def single_obs_to_tensor(obs: dict[str, Any], *, player_id: int, P: int=P_MAX, F: int=F_MAX, device: Any='cpu') -> dict[str, Any]:
    return _dict_obs_to_tensor(obs, player_id=player_id, P=P, F=F, device=device)

def sparse_action_row_to_moves(action_payload: dict[str, Any], obs: dict[str, Any], *, player_id: int) -> list[list[Any]]:
    return _sparse_actions_to_list(action_payload, obs, player_id=int(player_id))
import dataclasses
import os
import sys
from dataclasses import dataclass
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import torch
from torch import Tensor



# === run_turn — ORIGINAL, unmodified (L2449-2466) ===
def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    device = obs_tensors['planets'].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)
    movement = ensure_planet_movement(obs_tensors=obs_tensors, expected_cfg=_movement_config(config, player_count=int(player_count)), cached_movement=getattr(memory, 'movement', None))
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[:H + 1]
    entries = plan_lite_waves(movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache, garrison_status=status, prod=movement.planet_prod, alive_by_step=alive_by_step, config=config, player_count=int(player_count))
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id))
    apply_private_planned_launches(movement=movement, launches=launches, owner_id=int(obs.player_id), obs_tensors=obs_tensors)
    planet_ids = obs_tensors['planets'][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# === ProducerLiteMemory + ProducerLiteRuntime (L2472-2503) ===
class ProducerLiteMemory:

    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None

class ProducerLiteRuntime:

    def __init__(self, memory: ProducerLiteMemory | None=None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors['step'] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(obs_tensors, config=config, player_count=int(mem.cached_player_count), memory=mem)
        mem.last_sparse_action_row = row
        return row
_RUNTIME = ProducerLiteRuntime()



# === agent entrypoint (L2504-2511) ===
def agent(obs):
    player = obs.get('player', 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)

