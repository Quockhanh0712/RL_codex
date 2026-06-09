from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import Tensor
from typing import Sequence, Any
from .config import (
    BOARD_SIZE, CENTER, SUN_RADIUS, LAUNCH_SURFACE_OFFSET, TARGET_HIT_SURFACE_OFFSET,
    MovementConfig, ProducerLiteConfig
)
from .movement import PlanetMovement, PlanetGarrisonStatus, parse_obs, fleet_speed
from .garrison import LaunchSet, GarrisonFlowDiff, sparse_launch_flow_delta, safe_drain
from .kinematics import DistanceCache, min_distance_to_targets, intercept_angle

# === PlannedLaunches + LaunchEntries (L1661-1717) ===
@dataclass(frozen=True)
class PlannedLaunches:
    source_slots: Tensor
    angle: Tensor
    ships: Tensor
    target_slots: Tensor
    eta_turns: Tensor
    valid: Tensor
    fleet_ids: Tensor

@dataclass(frozen=True)
class LaunchEntries:
    source_slots: Tensor
    target_slots: Tensor
    ships: Tensor
    angle: Tensor
    eta: Tensor
    valid: Tensor

    @property
    def width(self) -> int:
        return int(self.source_slots.shape[0])

def concat_launch_entries(entries: Sequence[LaunchEntries]) -> LaunchEntries:
    if not entries:
        raise ValueError('concat_launch_entries requires at least one entry table')
    if len(entries) == 1:
        return entries[0]
    return LaunchEntries(source_slots=torch.cat([e.source_slots for e in entries], dim=0), target_slots=torch.cat([e.target_slots for e in entries], dim=0), ships=torch.cat([e.ships for e in entries], dim=0), angle=torch.cat([e.angle for e in entries], dim=0), eta=torch.cat([e.eta for e in entries], dim=0), valid=torch.cat([e.valid for e in entries], dim=0))

def disambiguate_duplicate_launches(entries: LaunchEntries, *, epsilon: float=1e-05) -> LaunchEntries:
    src = entries.source_slots
    ang = entries.angle
    ships = entries.ships
    valid = entries.valid
    L = src.shape[0]
    if L < 2 or not bool(valid.any()):
        return entries
    device = src.device
    src_i = src.unsqueeze(1)
    src_j = src.unsqueeze(0)
    ang_i = ang.unsqueeze(1)
    ang_j = ang.unsqueeze(0)
    ships_i = ships.unsqueeze(1)
    ships_j = ships.unsqueeze(0)
    valid_i = valid.unsqueeze(1)
    valid_j = valid.unsqueeze(0)
    j_indices = torch.arange(L, device=device).view(1, L)
    i_indices = torch.arange(L, device=device).view(L, 1)
    earlier = j_indices < i_indices
    match = valid_i & valid_j & (src_i == src_j) & (ang_i == ang_j) & (ships_i == ships_j) & earlier
    if not bool(match.any()):
        return entries
    dup_count = match.sum(dim=1).to(ang.dtype)
    new_angle = ang + dup_count * float(epsilon)
    return LaunchEntries(source_slots=entries.source_slots, target_slots=entries.target_slots, ships=entries.ships, angle=new_angle, eta=entries.eta, valid=entries.valid)



# === Launch helpers (L1724-1798) ===
def _resolve_player_next_fleet_id(obs_tensors: dict, *, device: torch.device) -> Tensor:
    next_fleet_id = obs_tensors.get('player_next_fleet_id', obs_tensors.get('next_fleet_id'))
    if next_fleet_id is None:
        return torch.zeros((), dtype=torch.long, device=device)
    return next_fleet_id.to(device=device, dtype=torch.long)

def infer_planned_launches_from_entries(*, obs_tensors: dict, movement: PlanetMovement, entries: LaunchEntries, player_id: int) -> PlannedLaunches:
    source_slots = entries.source_slots
    angle = entries.angle
    ships = entries.ships
    launch_valid = entries.valid
    L = source_slots.shape[0]
    device = source_slots.device
    P = max(int(movement.P), 1)
    next_fleet_id = _resolve_player_next_fleet_id(obs_tensors, device=device)
    launch_long = launch_valid.to(torch.long)
    launch_rank = launch_long.cumsum(0) - launch_long
    fleet_ids = next_fleet_id + launch_rank
    src_safe = source_slots.clamp(min=0, max=P - 1)
    launch_x, launch_y = movement.position_at_slots(src_safe, 0)
    source_r = movement.radii[src_safe]
    start_x = launch_x + torch.cos(angle) * (source_r + 0.1)
    start_y = launch_y + torch.sin(angle) * (source_r + 0.1)
    source_planet_ids = movement.planet_ids[src_safe]
    rows = torch.full((L, 7), -1.0, dtype=movement.dtype, device=device)
    rows[..., 0] = fleet_ids.to(dtype=movement.dtype)
    rows[..., 1] = float(player_id)
    rows[..., 2] = start_x.to(dtype=movement.dtype)
    rows[..., 3] = start_y.to(dtype=movement.dtype)
    rows[..., 4] = angle.to(dtype=movement.dtype)
    rows[..., 5] = source_planet_ids.to(dtype=movement.dtype)
    rows[..., 6] = ships.to(dtype=movement.dtype)
    rows[..., 0] = torch.where(launch_valid, rows[..., 0], torch.full_like(rows[..., 0], -1.0))
    target_slots = torch.zeros(L, dtype=torch.long, device=device)
    eta_turns = torch.zeros(L, dtype=torch.float32, device=device)
    intent_valid = torch.zeros(L, dtype=torch.bool, device=device)
    fleet_slot = torch.where(launch_valid)[0]
    if int(fleet_slot.numel()) > 0:
        estimate = _estimate_new_fleet_arrivals(movement=movement, obs_fleets=rows, fleet_slot=fleet_slot)
        valid_hit = estimate['has_hit']
        if bool(valid_hit.any()):
            src = fleet_slot[valid_hit]
            target_slots[src] = estimate['target_slot'][valid_hit]
            eta_turns[src] = estimate['eta_index'][valid_hit].to(dtype=torch.float32) + 1.0
            intent_valid[src] = True
    return PlannedLaunches(source_slots=source_slots, angle=angle, ships=ships, target_slots=target_slots, eta_turns=eta_turns, valid=intent_valid, fleet_ids=fleet_ids)

def apply_private_planned_launches(*, movement: PlanetMovement, launches: PlannedLaunches, owner_id: int, obs_tensors: dict) -> None:
    if not movement.track_fleets:
        return
    movement.record_fleet_arrivals(target_slots=launches.target_slots, owner_ids=int(owner_id), ships=launches.ships, eta=launches.eta_turns, valid=launches.valid)
    nfid = obs_tensors.get('next_fleet_id')
    if nfid is None:
        raise ValueError("obs_tensors is missing 'next_fleet_id'")
    movement.stash_pending_own_launches(owner_id=int(owner_id), source_slots=launches.source_slots, ships=launches.ships, angle=launches.angle, target_slots=launches.target_slots, eta=launches.eta_turns, valid=launches.valid, prev_next_fleet_id=nfid)
'Flow-diff scored planner core: candidate scoring, shortlists, aim, selection.\n\n\n\nPure, tensor-only planning helpers for one game: the competitive net-ship-delta\n\nscorer, target/source shortlists, capture-floor sizing, the strict-superset\n\nreachability gate, the device-stable greedy selector, the hold-reserve cap\n\n``safe_drain``, and the pressure-gradient regrouper.\n\n'
import torch
from torch import Tensor

def largest_initial_player_count(obs_tensors: dict) -> int:
    metadata_count = obs_tensors.get('player_count')
    if metadata_count is not None:
        count = int(metadata_count.flatten()[0].item()) if isinstance(metadata_count, Tensor) else int(metadata_count)
        if count in (2, 4):
            return count
    initial = obs_tensors['initial_planets']
    pid = initial[:, 0]
    owner = initial[:, 1]
    mask = (pid >= 0) & (owner >= 0)
    owners = owner[mask]
    n_max = 2
    if owners.numel() > 0:
        n_max = max(n_max, int(torch.unique(owners.long()).numel()))
    return n_max



# === Make launch set + scoring (L1799-1831) ===
def make_launch_set(*, source_slots: Tensor, target_slots: Tensor, ships: Tensor, eta: Tensor, valid: Tensor, player_id: int) -> LaunchSet:
    owner = torch.full_like(source_slots, int(player_id), dtype=torch.long)
    return LaunchSet(source_slots=source_slots.to(torch.long), target_slots=target_slots.to(torch.long), ships=ships, eta=eta, owner=owner, valid=valid.to(torch.bool))

def competitive_score(diff: GarrisonFlowDiff, *, player_id: int) -> Tensor:
    net = diff.net_ship_delta
    me = net[..., int(player_id)]
    opp = net.sum(dim=-1) - me
    return me - opp

def score_candidates(status: PlanetGarrisonStatus, *, prod: Tensor, alive_by_step: Tensor, player_count: int, launches: LaunchSet, player_id: int) -> Tensor:
    diff = sparse_launch_flow_delta(status, prod=prod, alive_by_step=alive_by_step, player_count=int(player_count), launches=launches, player_id=int(player_id))
    return competitive_score(diff, player_id=int(player_id))

def _stable_topk_indices(ranked: Tensor, k: int) -> Tensor:
    order = torch.argsort(ranked, dim=-1, descending=True, stable=True)
    return order[..., :max(1, int(k))]

def _stable_argmax(scores: Tensor) -> Tensor:
    C = int(scores.shape[-1])
    is_max = scores == scores.max(dim=-1, keepdim=True).values
    idx = torch.arange(C, device=scores.device).expand_as(scores)
    return torch.where(is_max, idx, torch.full_like(idx, C)).argmin(dim=-1)

def _candidate_indices(values: Tensor, mask: Tensor, cap: int) -> tuple[Tensor, Tensor]:
    p_count = values.shape[0]
    k = p_count if cap <= 0 else min(int(cap), p_count)
    neg_inf = torch.full_like(values, float('-inf'))
    ranked = torch.where(mask, values, neg_inf)
    top_idx = _stable_topk_indices(ranked, max(1, k))
    top_vals = ranked[top_idx]
    return (top_idx, top_vals > float('-inf'))



# === Target selection + greedy + regroup (L1832-2052) ===
def is_comet_planet(obs_tensors: dict, P: int, device: torch.device) -> Tensor | None:
    comet_ids = obs_tensors.get('comet_planet_ids')
    planets = obs_tensors.get('planets')
    if comet_ids is None or planets is None:
        return None
    planet_ids = planets[..., 0].long()
    comet_ids = comet_ids.to(device=device)
    mask = torch.zeros(P, dtype=torch.bool, device=device)
    for c in range(int(comet_ids.shape[-1])):
        cid = comet_ids[c]
        mask = mask | (planet_ids == cid) & (cid >= 0)
    return mask

def reinforcement_timing_factor(eta: Tensor, *, eta_free: float, eta_scale: float) -> Tensor:
    scale = max(float(eta_scale), 1e-06)
    return ((eta - float(eta_free)) / scale).clamp(0.0, 1.0)

def capture_floor(garrison_status: PlanetGarrisonStatus, *, target_idx: Tensor, k_max: int, capture_overhead: float, player_id: int, reinforcement: Tensor | None=None) -> Tensor:
    ships = garrison_status.ships
    owner = garrison_status.owner
    dtype = ships.dtype if ships.is_floating_point() else torch.float32
    T = target_idx.shape[0]
    H_axis = int(ships.shape[-1])
    P = int(ships.shape[0])
    K = max(0, min(int(k_max), H_axis - 1))
    if K == 0:
        return torch.empty(T, 0, dtype=dtype, device=ships.device)
    tgt = target_idx.clamp(min=0, max=max(P - 1, 0))
    gathered = ships[tgt].to(dtype=dtype)
    owner_g = owner[tgt]
    k_idx = torch.arange(1, K + 1, device=ships.device).view(1, K).expand(T, K)
    defenders = gathered.gather(-1, k_idx)
    mine_at_k = owner_g.gather(-1, k_idx) == int(player_id)
    if reinforcement is not None:
        assert reinforcement.shape[-1] >= K, f'reinforcement last dim {reinforcement.shape[-1]} < capture_floor K={K}'
        extra = reinforcement[..., :K].to(dtype=dtype, device=ships.device)
    else:
        extra = 0.0
    cap = (defenders + float(capture_overhead) + extra).clamp(min=1.0).ceil()
    return torch.where(mine_at_k, torch.ones_like(cap), cap)

def attack_target_mask(obs, obs_tensors: dict) -> Tensor:
    mask = (obs.is_enemy | obs.is_neutral) & obs.alive
    comet = is_comet_planet(obs_tensors, obs.P, obs.device)
    if comet is not None:
        mask = mask & ~comet
    return mask

def friendly_flip_targets(obs, garrison_status: PlanetGarrisonStatus, *, H: int, prod: Tensor) -> tuple[Tensor, Tensor]:
    P = obs.P
    device = obs.device
    pid = int(obs.player_id)
    if H <= 0:
        z = torch.zeros(P, device=device)
        return (torch.zeros(P, dtype=torch.bool, device=device), z)
    owner_h = garrison_status.owner[..., 1:]
    flips = obs.owned.unsqueeze(-1) & (owner_h != pid)
    any_flip = flips.any(dim=-1)
    flip_turn = _stable_argmax(flips.to(torch.int64)) + 1
    remaining = (float(H) - flip_turn.to(prod.dtype)).clamp(min=0.0)
    urgency = prod * remaining + obs.ships
    urgency = torch.where(any_flip, urgency, torch.full_like(urgency, float('-inf')))
    return (any_flip, urgency)

def build_target_shortlist(obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask):
    P = obs.P
    device = obs.device
    n_attack = max(1, min(int(config.max_offensive_targets), P))
    R = max(0, min(int(config.max_defensive_targets), P))
    attack_mask = attack_target_mask(obs, obs_tensors)
    proximity = min_distance_to_targets(cache, source_mask, attack_mask, max_k=K_eta)
    attack_pref = torch.where(attack_mask, -proximity, torch.full_like(proximity, float('-inf')))
    atk_idx, atk_exists = _candidate_indices(attack_pref, attack_mask, n_attack)
    if R > 0:
        flip_mask, urgency = friendly_flip_targets(obs, garrison_status, H=H, prod=prod)
        def_idx, def_exists = _candidate_indices(urgency, flip_mask, R)
        target_idx = torch.cat([atk_idx, def_idx], dim=0)
        target_exists = torch.cat([atk_exists, def_exists], dim=0)
    else:
        target_idx, target_exists = (atk_idx, atk_exists)
    return (target_idx, target_exists)

def reachable_mask(movement: PlanetMovement, *, source_idx: Tensor, target_idx: Tensor, fleet_sizes: Tensor, eta_cap: Tensor, eps: float=0.0001) -> Tensor:
    S, T, G = fleet_sizes.shape
    P = int(movement.P)
    dt = movement.dtype
    K = max(1, min(int(movement.movement_horizon), int(torch.ceil(eta_cap.max()).item())))
    src = source_idx.clamp(0, P - 1)
    tgt = target_idx.clamp(0, P - 1)
    sx = movement.x[0][src].view(S, 1, 1)
    sy = movement.y[0][src].view(S, 1, 1)
    tx = movement.x[:K + 1].gather(1, tgt.view(1, T).expand(K + 1, T))
    ty = movement.y[:K + 1].gather(1, tgt.view(1, T).expand(K + 1, T))
    ax = tx[:K, :].view(1, K, T)
    ay = ty[:K, :].view(1, K, T)
    bx = tx[1:, :].view(1, K, T)
    by = ty[1:, :].view(1, K, T)
    abx = bx - ax
    aby = by - ay
    apx = sx - ax
    apy = sy - ay
    denom = (abx * abx + aby * aby).clamp(min=1e-12)
    u = ((apx * abx + apy * aby) / denom).clamp(0.0, 1.0)
    cx = ax + u * abx
    cy = ay + u * aby
    seg_dist = torch.sqrt(((sx - cx) ** 2 + (sy - cy) ** 2).clamp(min=0.0))
    src_r = movement.radii[src].view(S, 1, 1)
    tgt_r = movement.radii[tgt].view(1, 1, T)
    gap = src_r + tgt_r + (LAUNCH_SURFACE_OFFSET + TARGET_HIT_SURFACE_OFFSET)
    surf = (seg_dist - gap).clamp(min=0.0)
    kv = torch.arange(1, K + 1, device=movement.device, dtype=dt).view(1, K, 1)
    ratio = surf / kv
    within = kv <= eta_cap.view(1, 1, T)
    ratio = torch.where(within, ratio, torch.full_like(ratio, float('inf')))
    min_ratio = ratio.amin(dim=1)
    speed = fleet_speed(fleet_sizes.clamp(min=1.0))
    reachable = min_ratio.unsqueeze(-1) <= speed * (1.0 + float(eps))
    distinct = (src.view(S, 1) != tgt.view(1, T)).unsqueeze(-1)
    return reachable & distinct

def _greedy_select(*, P, W, device, dtype, score, cand_src, cand_send, cand_angle, cand_eta, cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def, source_budget, target_exists, roi_threshold) -> LaunchEntries:
    C, L = (int(cand_src.shape[0]), int(cand_src.shape[1]))
    target_taken = ~target_exists.clone()
    defended = torch.zeros(P, dtype=torch.bool, device=device)
    used_src = torch.zeros(P, dtype=torch.bool, device=device)
    w_src = torch.zeros(W, L, dtype=torch.long, device=device)
    w_send = torch.zeros(W, L, dtype=dtype, device=device)
    w_angle = torch.zeros(W, L, dtype=dtype, device=device)
    w_eta = torch.ones(W, L, dtype=dtype, device=device)
    w_tgt = torch.zeros(W, L, dtype=torch.long, device=device)
    w_active = torch.zeros(W, L, dtype=torch.bool, device=device)
    for w in range(W):
        taken_cand = target_taken[cand_tgt_short]
        budget_at = source_budget[cand_src]
        can_fund = ((cand_send <= budget_at) | ~cand_active).all(dim=-1)
        tgt_used_as_src = used_src[cand_tgt_slot]
        contrib_defended = (defended[cand_src] & cand_active).any(dim=-1)
        mask = torch.isfinite(score) & ~taken_cand & can_fund & ~tgt_used_as_src & ~contrib_defended
        masked = torch.where(mask, score, torch.full_like(score, float('-inf')))
        best_c = _stable_argmax(masked)
        best_score = masked[best_c]
        fired = bool(torch.isfinite(best_score) & (best_score > roi_threshold))
        if not fired:
            break
        sel_src = cand_src[best_c]
        sel_send = cand_send[best_c]
        sel_active = cand_active[best_c]
        w_src[w] = sel_src
        w_send[w] = torch.where(sel_active, sel_send, torch.zeros_like(sel_send))
        w_angle[w] = cand_angle[best_c]
        w_eta[w] = cand_eta[best_c]
        w_tgt[w] = cand_tgt_slot[best_c]
        w_active[w] = sel_active
        debit = torch.zeros_like(source_budget)
        debit.scatter_add_(0, sel_src, torch.where(sel_active, sel_send, torch.zeros_like(sel_send)))
        source_budget = (source_budget - debit).clamp(min=0.0)
        target_taken[cand_tgt_short[best_c]] = True
        src_mark = torch.zeros(P, dtype=torch.long, device=device)
        src_mark.scatter_add_(0, sel_src, sel_active.to(torch.long))
        used_src = used_src | (src_mark > 0)
        sel_tgt = cand_tgt_slot[best_c]
        sel_is_def = bool(cand_is_def[best_c])
        defended[sel_tgt] = defended[sel_tgt] | sel_is_def
    WL = W * L
    entries = LaunchEntries(source_slots=w_src.reshape(WL), target_slots=w_tgt.reshape(WL), ships=torch.where(w_active, w_send, torch.zeros_like(w_send)).reshape(WL), angle=torch.where(w_active, w_angle, torch.zeros_like(w_angle)).reshape(WL), eta=torch.where(w_active, w_eta, torch.ones_like(w_eta)).reshape(WL), valid=w_active.reshape(WL))
    return (entries, source_budget)

def _plan_regroup(*, movement, obs, obs_tensors, garrison_status, leftover, original_ships, pressure, config, H) -> LaunchEntries:
    P = obs.P
    device = obs.device
    dtype = original_ships.dtype
    pid = int(obs.player_id)
    min_send = float(config.min_ships_to_launch)
    src_mask = obs.owned & obs.alive & (leftover >= min_send)
    if not bool(src_mask.any()):
        return _empty_entries(device, dtype)
    S_cap = max(1, min(int(config.max_regroup_sources_per_lane), P))
    src_idx, src_exists = _candidate_indices(leftover, src_mask, S_cap)
    S = int(src_idx.shape[0])
    leftover_s = leftover[src_idx.clamp(0, P - 1)]
    orig_s = original_ships[src_idx.clamp(0, P - 1)]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain_s = safe_drain(garrison_status, source_idx=src_idx, source_ships=orig_s, H_eff=H_eff, player_id=pid)
    committed_s = (orig_s - leftover_s).clamp(min=0.0)
    regroup_cap = torch.minimum(leftover_s, (drain_s - committed_s).clamp(min=0.0)).floor()
    can_send = src_exists & (regroup_cap >= min_send)
    if not bool(can_send.any()):
        return _empty_entries(device, dtype)
    dst_mask = obs.owned & obs.alive
    comet = is_comet_planet(obs_tensors, P, device)
    if comet is not None:
        dst_mask = dst_mask & ~comet
    T_cap = max(1, min(int(config.max_regroup_targets_per_source), P))
    dst_idx, dst_exists = _candidate_indices(pressure, dst_mask, T_cap)
    T = int(dst_idx.shape[0])
    regroup_active = reachable_mask(movement, source_idx=src_idx, target_idx=dst_idx, fleet_sizes=regroup_cap.view(S, 1, 1).expand(S, T, 1), eta_cap=torch.full((T,), float(config.max_regroup_time), device=device)).squeeze(-1)
    aim = intercept_angle(movement, src_idx.unsqueeze(1), dst_idx.unsqueeze(0), regroup_cap.unsqueeze(1), active=regroup_active)
    angle = aim['angle']
    eta = aim['eta']
    viable = aim['viable']
    src_pres = pressure[src_idx.clamp(0, P - 1)].view(S, 1)
    dst_pres = pressure[dst_idx.clamp(0, P - 1)].view(1, T)
    gap = dst_pres - src_pres
    owner = garrison_status.owner
    H_axis = int(owner.shape[-1])
    dst_owner = owner[dst_idx.clamp(0, P - 1)]
    k = torch.ceil(eta).clamp(min=0, max=H_axis - 1).to(torch.long)
    owner_at_k = dst_owner.unsqueeze(0).expand(S, T, H_axis).gather(-1, k.unsqueeze(-1)).squeeze(-1)
    still_mine = owner_at_k == pid
    src_neq_dst = src_idx.view(S, 1) != dst_idx.view(1, T)
    valid = viable & still_mine & src_neq_dst & (gap > float(config.regroup_pressure_delta_min)) & (eta <= float(config.max_regroup_time)) & can_send.view(S, 1) & dst_exists.view(1, T)
    sc = torch.where(valid, gap - float(config.regroup_time_penalty_weight) * eta, torch.full_like(gap, float('-inf')))
    best_t = _stable_argmax(sc)
    best_score = sc.gather(-1, best_t.unsqueeze(-1)).squeeze(-1)
    best_valid = torch.isfinite(best_score)
    s_ar = torch.arange(S, device=device)
    best_dst = dst_idx[best_t]
    best_angle = angle[s_ar, best_t]
    best_eta = eta[s_ar, best_t]
    return LaunchEntries(source_slots=src_idx, target_slots=best_dst, ships=torch.where(best_valid, regroup_cap, torch.zeros_like(regroup_cap)), angle=torch.where(best_valid, best_angle, torch.zeros_like(best_angle)), eta=torch.where(best_valid, best_eta, torch.ones_like(best_eta)), valid=best_valid)



# === Empty entries + payload conversion (L2053-2079) ===
def _empty_entries(device: torch.device, dtype: torch.dtype) -> LaunchEntries:
    z = torch.zeros(0, dtype=dtype, device=device)
    zl = torch.zeros(0, dtype=torch.long, device=device)
    return LaunchEntries(source_slots=zl, target_slots=zl, ships=z, angle=z, eta=z, valid=torch.zeros(0, dtype=torch.bool, device=device))

def entries_to_sparse_payload(entries: LaunchEntries, *, planet_ids: Tensor) -> dict[str, Tensor]:
    L = entries.source_slots.shape[0]
    device = entries.source_slots.device
    P = int(planet_ids.shape[0])
    valid_long = entries.valid.to(torch.int64)
    counts = valid_long.sum().to(torch.int32)
    max_count = int(counts.item())
    out_from = torch.full((max_count,), -1, dtype=torch.int32, device=device)
    out_angle = torch.zeros((max_count,), dtype=torch.float32, device=device)
    out_ships = torch.zeros((max_count,), dtype=torch.float32, device=device)
    if max_count == 0:
        return {'from_planet_id': out_from, 'angle': out_angle, 'num_ships': out_ships, 'counts': counts}
    safe_src = entries.source_slots.clamp(min=0, max=max(P - 1, 0))
    from_pid_full = planet_ids[safe_src].to(torch.int32)
    launch_rank = valid_long.cumsum(0) - valid_long
    l_idx = torch.where(entries.valid)[0]
    pos = launch_rank[l_idx]
    out_from[pos] = from_pid_full[l_idx]
    out_angle[pos] = entries.angle[l_idx].to(torch.float32)
    out_ships[pos] = entries.ships[l_idx].to(torch.float32)
    return {'from_planet_id': out_from, 'angle': out_angle, 'num_ships': out_ships, 'counts': counts}



# === Enemy pressure + risk (L2252-2307) ===
def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-06))
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-06)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)

def potential_attack_risk(obs, cache, *, horizon: float, player_id: int, config) -> Tensor:
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    H = max(float(horizon), 1e-06)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    prod = obs.prod.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-06))
    reach = (speeds.view(P, 1) * H).clamp(min=1e-06)
    decay = (1.0 - d0 / reach).clamp(min=0.0)
    eye = torch.eye(P, device=device, dtype=torch.bool)
    x = obs.x.to(dtype)
    y = obs.y.to(dtype)
    ax = x.view(P, 1)
    ay = y.view(P, 1)
    bx = x.view(1, P)
    by = y.view(1, P)
    abx = bx - ax
    aby = by - ay
    denom = (abx * abx + aby * aby).clamp(min=1e-12)
    u = (((CENTER - ax) * abx + (CENTER - ay) * aby) / denom).clamp(0.0, 1.0)
    cx = ax + u * abx
    cy = ay + u * aby
    sun_dist = torch.sqrt(((cx - CENTER) ** 2 + (cy - CENTER) ** 2).clamp(min=0.0))
    los_clear = (sun_dist >= float(SUN_RADIUS)).to(dtype)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    strength = ships + float(config.risk_enemy_prod_weight) * prod
    valid_e = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    threat = torch.where(valid_e, strength.view(P, 1) * decay * los_clear, torch.zeros_like(decay))
    enemy_threat = threat.sum(dim=0)
    own = obs.owned & obs.alive
    valid_o = own.view(P, 1) & obs.alive.view(1, P) & ~eye
    support = torch.where(valid_o, (1.0 + ships).view(P, 1) * decay, torch.zeros_like(decay)).sum(dim=0)
    value = 1.0 + float(config.risk_self_prod_weight) * prod
    return value * enemy_threat / (1.0 + float(config.risk_support_weight) * support)



# === plan_lite_waves (L2308-2447) — ORIGINAL, unmodified ===
def plan_lite_waves(*, movement: PlanetMovement, obs, obs_tensors: dict, cache, garrison_status, prod: Tensor, alive_by_step: Tensor, config: ProducerLiteConfig, player_count: int):
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))
    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)
    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(obs, obs_tensors, garrison_status, cache, config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask)
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]
    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(garrison_status, source_idx=source_idx, source_ships=source_ships, H_eff=H_eff, player_id=pid)
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)
    floor = capture_floor(garrison_status, target_idx=target_idx, k_max=K_eta, capture_overhead=1.0, player_id=pid)
    K = int(floor.shape[-1])
    sizes = drain.view(S, 1).expand(S, T).floor()
    active = reachable_mask(movement, source_idx=source_idx, target_idx=target_idx, fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap).squeeze(-1)
    aim = intercept_angle(movement, source_idx.unsqueeze(1), target_idx.unsqueeze(0), sizes, active=active)
    angle = aim['angle']
    eta = aim['eta']
    viable = aim['viable'] & (eta <= eta_cap.view(1, T))
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr
    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = viable & clears_floor & (sizes >= 1.0) & src_neq_tgt & source_exists.view(S, 1) & target_exists.view(1, T)
    if not bool(config.enable_focus_fire):
        L = 1
        C = S * T
        cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
        cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
        cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
        cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
        cand_angle = angle.reshape(C, L)
        cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
        cand_active = valid.reshape(C, L)
        cand_valid = valid.reshape(C)
    else:
        L = max(1, int(config.max_strike_sources))
        ST = S * T
        ss_src = torch.zeros(ST, L, dtype=torch.long, device=device)
        ss_src[:, 0] = source_idx.view(S, 1).expand(S, T).reshape(-1)
        ss_send = torch.zeros(ST, L, dtype=dtype, device=device)
        ss_send[:, 0] = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(-1)
        ss_angle = torch.zeros(ST, L, dtype=dtype, device=device)
        ss_angle[:, 0] = angle.reshape(-1)
        ss_eta = torch.ones(ST, L, dtype=dtype, device=device)
        ss_eta[:, 0] = torch.where(valid, eta, torch.ones_like(eta)).reshape(-1)
        ss_active = torch.zeros(ST, L, dtype=torch.bool, device=device)
        ss_active[:, 0] = valid.reshape(-1)
        ss_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(-1)
        ss_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(-1)
        ss_valid = valid.reshape(-1)
        eligible = viable & (sizes >= 1.0) & src_neq_tgt & source_exists.view(S, 1) & target_exists.view(1, T)
        step_arr = eta.clamp(min=1.0, max=float(K_eta)).ceil().long()
        pooled = []
        if L >= 2 and K > 0:
            for t in range(T):
                if bool(target_is_mine[t]):
                    continue
                rows = torch.nonzero(eligible[:, t], as_tuple=False).flatten()
                if int(rows.numel()) < 2:
                    continue
                steps_t = step_arr[rows, t]
                for k in torch.unique(steps_t).tolist():
                    k = int(k)
                    if k < 1 or k - 1 >= K:
                        continue
                    grp = rows[steps_t == k]
                    if int(grp.numel()) < 2:
                        continue
                    gd = sizes[grp, t]
                    order = torch.argsort(gd, descending=True, stable=True)
                    grp = grp[order]
                    csum = torch.cumsum(gd[order], dim=0)
                    need = floor[t, k - 1]
                    hit = torch.nonzero(csum >= need, as_tuple=False)
                    if int(hit.numel()) == 0:
                        continue
                    j = int(hit[0].item()) + 1
                    if j < 2 or j > L:
                        continue
                    pooled.append((t, grp[:j]))
        if pooled:
            C2 = len(pooled)
            p_src = torch.zeros(C2, L, dtype=torch.long, device=device)
            p_send = torch.zeros(C2, L, dtype=dtype, device=device)
            p_angle = torch.zeros(C2, L, dtype=dtype, device=device)
            p_eta = torch.ones(C2, L, dtype=dtype, device=device)
            p_active = torch.zeros(C2, L, dtype=torch.bool, device=device)
            p_tgt_slot = torch.zeros(C2, dtype=torch.long, device=device)
            p_tgt_short = torch.zeros(C2, dtype=torch.long, device=device)
            for i, (t, grp) in enumerate(pooled):
                j = int(grp.numel())
                p_src[i, :j] = source_idx[grp]
                p_send[i, :j] = sizes[grp, t]
                p_angle[i, :j] = angle[grp, t]
                p_eta[i, :j] = eta[grp, t]
                p_active[i, :j] = True
                p_tgt_slot[i] = target_idx[t]
                p_tgt_short[i] = t
            cand_src = torch.cat([ss_src, p_src], dim=0)
            cand_send = torch.cat([ss_send, p_send], dim=0)
            cand_angle = torch.cat([ss_angle, p_angle], dim=0)
            cand_eta = torch.cat([ss_eta, p_eta], dim=0)
            cand_active = torch.cat([ss_active, p_active], dim=0)
            cand_tgt_slot = torch.cat([ss_tgt_slot, p_tgt_slot], dim=0)
            cand_tgt_short = torch.cat([ss_tgt_short, p_tgt_short], dim=0)
            cand_valid = torch.cat([ss_valid, torch.ones(C2, dtype=torch.bool, device=device)], dim=0)
        else:
            cand_src, cand_send, cand_angle = (ss_src, ss_send, ss_angle)
            cand_eta, cand_active = (ss_eta, ss_active)
            cand_tgt_slot, cand_tgt_short, cand_valid = (ss_tgt_slot, ss_tgt_short, ss_valid)
        C = int(cand_src.shape[0])
    cand_is_def = target_is_mine[cand_tgt_short]
    launches = make_launch_set(source_slots=cand_src, target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L), ships=cand_send, eta=cand_eta, valid=cand_active & cand_valid.unsqueeze(-1), player_id=pid)
    score = score_candidates(garrison_status, prod=prod, alive_by_step=alive_by_step, player_count=int(player_count), launches=launches, player_id=pid)
    score = torch.where(cand_valid, score, torch.full_like(score, float('-inf')))
    wave_entries, leftover = _greedy_select(P=P, W=W, device=device, dtype=dtype, score=score, cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta, cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short, cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(), target_exists=target_exists, roi_threshold=float(config.roi_threshold))
    if not bool(config.enable_regroup):
        return wave_entries
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    if bool(config.enable_potential_risk):
        enemy_mass = enemy_mass + float(config.risk_blend_weight) * potential_attack_risk(obs, cache, horizon=float(K_eta), player_id=pid, config=config)
    regroup_entries = _plan_regroup(movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status, leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass, config=config, H=H)
    return concat_launch_entries([wave_entries, regroup_entries])

