from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .config import OWN, ENEMY, NEUTRAL, DEAD
from .movement import PlanetGarrisonStatus

# === LaunchSet + garrison simulation (L1263-1459) ===
@dataclass(frozen=True)
class LaunchSet:
    source_slots: Tensor
    target_slots: Tensor
    ships: Tensor
    eta: Tensor
    owner: Tensor
    valid: Tensor

    @property
    def has_candidate_axis(self) -> bool:
        return self.source_slots.dim() >= 2

def _per_step_survivor(arrivals: Tensor) -> tuple[Tensor, Tensor]:
    A = int(arrivals.shape[-1])
    if A >= 2:
        top2 = arrivals.topk(k=2, dim=-1)
        top_ships = top2.values[..., 0]
        second_ships = top2.values[..., 1]
        top_owner = top2.indices[..., 0].to(dtype=torch.long)
    else:
        top_ships, top_owner = arrivals.max(dim=-1)
        second_ships = torch.zeros_like(top_ships)
        top_owner = top_owner.to(dtype=torch.long)
    tied = top_ships == second_ships
    survivor_ships = torch.where(tied, torch.zeros_like(top_ships), (top_ships - second_ships).clamp(min=0.0))
    return (top_owner, survivor_ships)

def _run_exact_recurrence(*, init_owner: Tensor, init_ships: Tensor, prod: Tensor, alive: Tensor, arrivals: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    N, P = init_owner.shape
    H = int(arrivals.shape[2])
    device = init_ships.device
    owner_out = torch.empty(N, P, H + 1, dtype=init_owner.dtype, device=device)
    ships_out = torch.empty(N, P, H + 1, dtype=init_ships.dtype, device=device)
    pre_owner_out = torch.empty_like(owner_out)
    pre_ships_out = torch.empty_like(ships_out)
    owner_out[..., 0] = init_owner
    ships_out[..., 0] = init_ships
    pre_owner_out[..., 0] = init_owner
    pre_ships_out[..., 0] = init_ships
    survivor_owner, survivor_ships = _per_step_survivor(arrivals)
    state_owner = init_owner.clone()
    state_ships = init_ships.clone()
    zero_ships = torch.zeros((), dtype=state_ships.dtype, device=device)
    neg_one = torch.full((), -1, dtype=state_owner.dtype, device=device)
    zero_prod = torch.zeros((), dtype=prod.dtype, device=device)
    for k in range(1, H + 1):
        a_before = alive[..., k - 1]
        a_now = alive[..., k]
        s_owner = survivor_owner[..., k - 1]
        s_ships = survivor_ships[..., k - 1]
        produces = a_before & (state_owner >= 0)
        state_ships = state_ships + torch.where(produces, prod, zero_prod)
        pre_owner_out[..., k] = torch.where(a_now, state_owner, neg_one)
        pre_ships_out[..., k] = torch.where(a_now, state_ships, zero_ships)
        has_combat = (s_ships > 0.0) & a_now
        same = state_owner == s_owner
        diff = state_ships - s_ships
        attacker_wins = ~same & (diff < 0.0)
        combat_ships = torch.where(same, state_ships + s_ships, diff.abs())
        combat_owner = torch.where(attacker_wins, s_owner, state_owner)
        state_ships = torch.where(has_combat, combat_ships, state_ships)
        state_owner = torch.where(has_combat, combat_owner, state_owner)
        state_owner = torch.where(a_now, state_owner, neg_one)
        state_ships = torch.where(a_now, state_ships, zero_ships)
        owner_out[..., k] = state_owner
        ships_out[..., k] = state_ships
    return (owner_out, ships_out, pre_owner_out, pre_ships_out)

def _validate_inputs(status: PlanetGarrisonStatus, prod: Tensor, alive_by_step: Tensor, player_count: int) -> tuple[int, int, int, int]:
    if status.arrivals_by_owner is None:
        raise ValueError('garrison status must carry arrivals_by_owner (build it from a PlanetMovement with track_fleets=True)')
    if status.pre_combat_owner is None or status.pre_combat_ships is None:
        raise ValueError('garrison status must carry pre_combat_owner/ships')
    if status.owner.dim() != 2:
        raise ValueError(f'expected a full-board status with owner shaped [P, H+1]; got {tuple(status.owner.shape)}')
    P, H1 = status.owner.shape
    H = H1 - 1
    A = int(status.arrivals_by_owner.shape[-1])
    if int(player_count) != A:
        raise ValueError(f'player_count={player_count} disagrees with arrivals owner axis A={A}')
    if tuple(prod.shape) != (P,):
        raise ValueError(f'prod must be [P]=({P},); got {tuple(prod.shape)}')
    if tuple(alive_by_step.shape) != (H1, P):
        raise ValueError(f'alive_by_step must be [H+1, P]=({H1}, {P}); got {tuple(alive_by_step.shape)}')
    return (P, H, A)

@dataclass(frozen=True)
class GarrisonFlowDiff:
    player_id: int
    ships_produced_current: Tensor
    ships_produced_hypothetical: Tensor
    ships_produced_delta: Tensor
    ships_lost_combat_current: Tensor
    ships_lost_combat_hypothetical: Tensor
    ships_lost_combat_delta: Tensor
    net_ship_delta: Tensor

    @property
    def player_count(self) -> int:
        return int(self.ships_produced_delta.shape[-1])

def _flow_terms_per_planet(*, owner: Tensor, pre_owner: Tensor, pre_ships: Tensor, arr_full: Tensor, prod: Tensor, alive_pmajor: Tensor) -> tuple[Tensor, Tensor]:
    A = int(arr_full.shape[-1])
    H = int(owner.shape[-1]) - 1
    fdtype = pre_ships.dtype
    a_idx = torch.arange(A, device=owner.device)
    producing_owner = owner[..., :H]
    amount = prod.unsqueeze(-1) * alive_pmajor[..., :H].to(fdtype)
    prod_owner_oh = producing_owner.unsqueeze(-1) == a_idx
    produced = (amount.unsqueeze(-1) * prod_owner_oh.to(fdtype)).sum(dim=-2)
    arr_k = arr_full[..., 1:, :]
    survivor_owner, survivor_ships = _per_step_survivor(arr_k)
    survived = torch.where(a_idx == survivor_owner.unsqueeze(-1), survivor_ships.unsqueeze(-1), torch.zeros_like(survivor_ships).unsqueeze(-1))
    attacker_lost = (arr_k - survived).clamp(min=0.0)
    prior_owner = pre_owner[..., 1:]
    prior_ships = pre_ships[..., 1:]
    fights_garrison = (survivor_ships > 0.0) & (survivor_owner != prior_owner) & (survivor_owner >= 0)
    garrison_loss = torch.where(fights_garrison, torch.minimum(prior_ships, survivor_ships), torch.zeros_like(prior_ships))
    is_survivor = (a_idx == survivor_owner.unsqueeze(-1)) & fights_garrison.unsqueeze(-1)
    is_prior = (a_idx == prior_owner.unsqueeze(-1)) & fights_garrison.unsqueeze(-1) & (prior_owner >= 0).unsqueeze(-1)
    garrison_lost = garrison_loss.unsqueeze(-1) * (is_survivor.to(fdtype) + is_prior.to(fdtype))
    combat_lost = (attacker_lost + garrison_lost).sum(dim=-2)
    return (produced, combat_lost)

def _normalize_launches_bcl(launches: LaunchSet) -> tuple[Tensor, ...]:
    fields = (launches.source_slots, launches.target_slots, launches.ships, launches.eta, launches.owner, launches.valid)
    if launches.has_candidate_axis:
        return fields
    return tuple((f.unsqueeze(0) for f in fields))

def sparse_launch_flow_delta(status: PlanetGarrisonStatus, *, prod: Tensor, alive_by_step: Tensor, player_count: int, launches: LaunchSet, player_id: int=0) -> GarrisonFlowDiff:
    P, H, A = _validate_inputs(status, prod, alive_by_step, player_count)
    device = status.owner.device
    fdtype = status.ships.dtype
    assert status.pre_combat_owner is not None and status.pre_combat_ships is not None
    assert status.arrivals_by_owner is not None
    src, tgt, ships, eta, owner, valid = _normalize_launches_bcl(launches)
    C = int(src.shape[0])
    L = int(src.shape[-1])
    src = src.to(device=device, dtype=torch.long)
    tgt = tgt.to(device=device, dtype=torch.long)
    ships = ships.to(device=device, dtype=fdtype)
    owner = owner.to(device=device, dtype=torch.long)
    valid = valid.to(device=device, dtype=torch.bool)
    h_idx = torch.ceil(eta.to(device=device, dtype=fdtype)).to(torch.long) - 1
    valid_t = valid & (ships > 0) & (tgt >= 0) & (tgt < P) & (owner >= 0) & (owner < A) & (h_idx >= 0) & (h_idx < H)
    valid_s = valid & (ships > 0) & (src >= 0) & (src < P)
    src_safe = src.clamp(0, max(P - 1, 0))
    tgt_safe = tgt.clamp(0, max(P - 1, 0))
    affected = torch.zeros(C, P, dtype=fdtype, device=device)
    affected.scatter_add_(1, src_safe, valid_s.to(fdtype))
    affected.scatter_add_(1, tgt_safe, valid_t.to(fdtype))
    affected_mask = affected > 0
    base_prod_pp, base_combat_pp = _flow_terms_per_planet(owner=status.owner, pre_owner=status.pre_combat_owner, pre_ships=status.pre_combat_ships, arr_full=status.arrivals_by_owner, prod=prod, alive_pmajor=alive_by_step.permute(1, 0))
    base_prod = base_prod_pp.sum(dim=0)
    base_combat = base_combat_pp.sum(dim=0)
    produced_delta = torch.zeros(C, A, dtype=fdtype, device=device)
    combat_delta = torch.zeros(C, A, dtype=fdtype, device=device)
    if bool(affected_mask.any()):
        c_aff, p_aff = affected_mask.nonzero(as_tuple=True)
        N = int(c_aff.numel())
        cell_id = torch.full((C, P), -1, dtype=torch.long, device=device)
        cell_id[c_aff, p_aff] = torch.arange(N, device=device)
        debit_cp = torch.zeros(C, P, dtype=fdtype, device=device)
        debit_cp.scatter_add_(1, src_safe, torch.where(valid_s, ships, torch.zeros_like(ships)))
        debit_aff = debit_cp[c_aff, p_aff]
        arr_aff = torch.zeros(N, H, A, dtype=fdtype, device=device)
        launch_cell = cell_id.gather(1, tgt_safe)
        m = valid_t
        cells, hh, oo, ss = (launch_cell[m], h_idx[m], owner[m], ships[m])
        ok = cells >= 0
        arr_aff.index_put_((cells[ok], hh[ok], oo[ok]), ss[ok], accumulate=True)
        base_arr_k = status.arrivals_by_owner[..., 1:, :]
        arrivals_cell = base_arr_k[p_aff] + arr_aff
        init_owner = status.owner[p_aff, 0]
        init_ships = (status.ships[p_aff, 0] - debit_aff).clamp(min=0.0)
        prod_aff = prod[p_aff]
        alive_aff = alive_by_step[:, p_aff].transpose(0, 1)
        o_t, _s_t, po_t, ps_t = _run_exact_recurrence(init_owner=init_owner.unsqueeze(1), init_ships=init_ships.unsqueeze(1), prod=prod_aff.unsqueeze(1), alive=alive_aff.unsqueeze(1), arrivals=arrivals_cell.unsqueeze(1))
        zero_frame = torch.zeros(N, 1, 1, A, dtype=fdtype, device=device)
        arr_full_cell = torch.cat([zero_frame, arrivals_cell.unsqueeze(1)], dim=-2)
        hyp_prod_pp, hyp_combat_pp = _flow_terms_per_planet(owner=o_t, pre_owner=po_t, pre_ships=ps_t, arr_full=arr_full_cell, prod=prod_aff.unsqueeze(1), alive_pmajor=alive_aff.unsqueeze(1))
        dprod = hyp_prod_pp.squeeze(1) - base_prod_pp[p_aff]
        dcombat = hyp_combat_pp.squeeze(1) - base_combat_pp[p_aff]
        produced_delta.index_put_((c_aff,), dprod, accumulate=True)
        combat_delta.index_put_((c_aff,), dcombat, accumulate=True)
    produced_current = base_prod.unsqueeze(0)
    combat_current = base_combat.unsqueeze(0)
    diff = GarrisonFlowDiff(player_id=int(player_id), ships_produced_current=produced_current, ships_produced_hypothetical=produced_current + produced_delta, ships_produced_delta=produced_delta, ships_lost_combat_current=combat_current, ships_lost_combat_hypothetical=combat_current + combat_delta, ships_lost_combat_delta=combat_delta, net_ship_delta=produced_delta - combat_delta)
    if not launches.has_candidate_axis:

        def _sq(t: Tensor) -> Tensor:
            return t.squeeze(0)
        diff = GarrisonFlowDiff(player_id=diff.player_id, ships_produced_current=base_prod, ships_produced_hypothetical=_sq(diff.ships_produced_hypothetical), ships_produced_delta=_sq(diff.ships_produced_delta), ships_lost_combat_current=base_combat, ships_lost_combat_hypothetical=_sq(diff.ships_lost_combat_hypothetical), ships_lost_combat_delta=_sq(diff.ships_lost_combat_delta), net_ship_delta=_sq(diff.net_ship_delta))
    return diff
'Cross-k distance cache for the movement-backed planner.\n\n\n\nEntry ``cross_dist[k, s, t]`` is the Euclidean distance from planet ``s`` at step\n\n0 to planet ``t`` at step ``k`` — the *cross-time* distance a fleet must travel if\n\nit launches now from ``s`` to intercept ``t`` at time ``k``. For static planets\n\nthis equals same-step pairwise distance; for orbiting sources the cross-time form\n\nis the geometrically correct quantity for fleet-intercept feasibility. A\n\nprecomputed ``[K+1, P, P]`` window gives exact per-step lookups for free.\n\n'


# === safe_drain (L2083-2107) ===
def safe_drain(garrison_status: PlanetGarrisonStatus, *, source_idx: Tensor, source_ships: Tensor, H_eff: Tensor, player_id: int=0) -> Tensor:
    S = source_idx.shape[0]
    ships_cache = garrison_status.ships
    dtype = ships_cache.dtype if ships_cache.is_floating_point() else torch.float32
    device = ships_cache.device
    H_axis = int(ships_cache.shape[-1])
    H = max(H_axis - 1, 0)
    P = int(ships_cache.shape[0])
    if H == 0:
        return torch.zeros(S, dtype=dtype, device=device)
    src_idx_safe = source_idx.clamp(min=0, max=max(P - 1, 0))
    src_ships_traj = ships_cache[src_idx_safe][..., 1:].to(dtype=dtype)
    src_owner_traj = garrison_status.owner[src_idx_safe][..., 1:]
    me_owned = src_owner_traj == int(player_id)
    turn_grid = torch.arange(1, H + 1, device=device, dtype=dtype).view(1, H)
    within_horizon = turn_grid <= H_eff
    held = me_owned & within_horizon & (src_ships_traj > 0.0)
    inf_fill = torch.full_like(src_ships_traj, float('inf'))
    cap_traj = torch.where(held, src_ships_traj, inf_fill)
    min_slack = cap_traj.min(dim=-1).values
    return torch.minimum(min_slack, source_ships.to(dtype)).clamp(min=0.0)
'Observation/action adapter between the move-list format and tensors.\n\n\n\nConverts an observation dict (``{"planets": [...], "fleets": [...], ...}``) into\n\nthe named tensor observation the planner consumes, and converts the planner\'s\n\nsparse launch payload\n\n(``{"from_planet_id": [L], "angle": [L], "num_ships": [L], "counts": scalar}``)\n\nback into a move list (``[[from_planet_id, angle, ships], ...]``).\n\n'
from typing import Any
import torch


