from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import Tensor
from .config import CENTER, LAUNCH_SURFACE_OFFSET, TARGET_HIT_SURFACE_OFFSET, BOARD_SIZE, SUN_RADIUS, _FP_ITERS
from .movement import PlanetMovement, fleet_speed, _swept_pair_hit_mask

_BIG = 1000000.0

# === Point-to-segment distance (L1121-1131) ===
def _point_to_segment_distance_sq(px: Tensor, py: Tensor, x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor) -> Tensor:
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    t = ((px - x1) * dx + (py - y1) * dy) / safe_denom
    t = t.clamp(0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return (px - proj_x) ** 2 + (py - proj_y) ** 2



# === DistanceCache + intercept + analytic first contact (L1464-1655) ===
@dataclass
class DistanceCache:
    cross_dist: Tensor
    alive_by_step: Tensor
    K: int

    @property
    def P(self) -> int:
        return int(self.cross_dist.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.cross_dist.device

    @property
    def dtype(self) -> torch.dtype:
        return self.cross_dist.dtype

def build_distance_cache(movement: PlanetMovement, *, max_k: int) -> DistanceCache:
    K = max(0, min(int(max_k), int(movement.movement_horizon)))
    P = int(movement.P)
    src_x0 = movement.x[0]
    src_y0 = movement.y[0]
    tgt_x = movement.x[:K + 1]
    tgt_y = movement.y[:K + 1]
    dx = src_x0.view(1, P, 1) - tgt_x.unsqueeze(1)
    dy = src_y0.view(1, P, 1) - tgt_y.unsqueeze(1)
    cross_dist = torch.sqrt((dx * dx + dy * dy).clamp(min=0.0))
    alive_by_step = movement.alive_by_step[:K + 1]
    return DistanceCache(cross_dist=cross_dist, alive_by_step=alive_by_step, K=K)

def min_distance_to_targets(cache: DistanceCache, source_mask: Tensor, target_mask: Tensor, *, max_k: int) -> Tensor:
    if source_mask.shape[-1] != cache.P or target_mask.shape[-1] != cache.P:
        raise ValueError('source_mask and target_mask must have shape [P]')
    K = max(0, min(int(max_k), int(cache.K)))
    if K <= 0:
        return torch.zeros(cache.P, dtype=cache.dtype, device=cache.device)
    cross = cache.cross_dist[1:K + 1].clone()
    alive_steps = cache.alive_by_step[1:K + 1]
    src_mask = source_mask.to(device=cache.device, dtype=torch.bool)
    tgt_mask = target_mask.to(device=cache.device, dtype=torch.bool)
    inf_v = float('inf')
    cross.masked_fill_(~alive_steps.unsqueeze(1), inf_v)
    cross.masked_fill_(~src_mask.view(1, cache.P, 1), inf_v)
    cross.masked_fill_(~tgt_mask.view(1, 1, cache.P), inf_v)
    best_per_target = cross.amin(dim=(0, 1))
    return torch.where(torch.isfinite(best_per_target), best_per_target, torch.zeros_like(best_per_target))
"Fixed-fleet intercept aim — sub-turn-accurate angle for an orbiting target.\n\n\n\nSolves the **continuous** intercept time ``t*`` (root of\n\n``v·t = dist(target_pos(t), source) − gap`` with the target on its analytic\n\norbit), aims at ``target_pos(t*)``, and verifies that angle with a\n\nfully-vectorized analytic first-contact check.\n\n\n\n* **Root** — a continuous fixed-point iteration (no grid scan / argmax /\n\n  bisection), free of grid-resolution artifacts.\n\n* **Verify** — :func:`_analytic_first_contact` reproduces the engine's\n\n  first-contact verdict exactly (swept-pair vs every planet, sun, bounds,\n\n  lowest-slot same-step tie-break) with no engine state and no per-step loop.\n\n  A shot is viable iff it contacts the target first.\n\n\n\nReturns ``angle`` / ``eta`` / ``viable``.\n\n"
import torch
from torch import Tensor
_FP_ITERS = 6
_BIG = 1000000.0

def intercept_angle(movement: PlanetMovement, source_slots: Tensor, target_slots: Tensor, fleet_sizes: Tensor, *, fp_iters: int=_FP_ITERS, active: Tensor | None=None) -> dict[str, Tensor]:
    dev = movement.device
    dt = movement.dtype
    H = int(movement.movement_horizon)
    src, tgt, ships = torch.broadcast_tensors(source_slots.to(device=dev), target_slots.to(device=dev), fleet_sizes.to(device=dev, dtype=dt))
    shape = src.shape
    src = src.long().clamp(0, max(movement.P - 1, 0)).reshape(-1)
    tgt = tgt.long().clamp(0, max(movement.P - 1, 0)).reshape(-1)
    ships = ships.to(dt).clamp(min=1.0).reshape(-1)
    M = src.shape[0]
    sx, sy = movement.position_at_slots(src, 0)
    src_r = movement.radii[src]
    tgt_r = movement.radii[tgt]
    speed = fleet_speed(ships).clamp(min=1e-06)
    t0x, t0y = movement.position_at_slots(tgt, 0)
    t1x, t1y = movement.position_at_slots(tgt, 1)
    R = torch.sqrt(((t0x - CENTER) ** 2 + (t0y - CENTER) ** 2).clamp(min=0.0))
    a0 = torch.atan2(t0y - CENTER, t0x - CENTER)
    a1 = torch.atan2(t1y - CENTER, t1x - CENTER)
    omega = torch.atan2(torch.sin(a1 - a0), torch.cos(a1 - a0))
    gap = src_r + LAUNCH_SURFACE_OFFSET + tgt_r + TARGET_HIT_SURFACE_OFFSET

    def target_pos(t: Tensor):
        ang = a0 + omega * t
        return (CENTER + R * torch.cos(ang), CENTER + R * torch.sin(ang))
    d0 = torch.sqrt(((t0x - sx) ** 2 + (t0y - sy) ** 2).clamp(min=0.0))
    t_star = ((d0 - gap) / speed).clamp(min=0.0, max=float(H))
    for _ in range(int(fp_iters)):
        tx, ty = target_pos(t_star)
        d = torch.sqrt(((tx - sx) ** 2 + (ty - sy) ** 2).clamp(min=0.0))
        t_star = ((d - gap) / speed).clamp(min=0.0, max=float(H))
    tx, ty = target_pos(t_star)
    angle = torch.atan2(ty - sy, tx - sx)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    launch_x = sx + cos_a * (src_r + LAUNCH_SURFACE_OFFSET)
    launch_y = sy + sin_a * (src_r + LAUNCH_SURFACE_OFFSET)
    eta_cap = (t_star + 2.0).clamp(max=float(H))
    seg_len = speed * eta_cap + tgt_r + 2.0
    px = movement.x[:H + 1, :]
    py = movement.y[:H + 1, :]
    radii_p = movement.radii
    alive0 = movement.alive_at(0)
    if active is None:
        contact, eta_c = _analytic_first_contact(launch_x=launch_x, launch_y=launch_y, cos_a=cos_a, sin_a=sin_a, speed=speed, px=px, py=py, p_alive0=alive0, radii=radii_p, H=H, seg_len=seg_len)
    else:
        act = active.broadcast_to(shape).reshape(M).to(torch.bool)
        n_max = max(1, int(act.sum().item()))
        order = (~act).to(torch.int8).argsort(stable=True)
        midx = order[:n_max]
        keep = act[midx]
        contact_m, eta_cm = _analytic_first_contact(launch_x=launch_x[midx], launch_y=launch_y[midx], cos_a=cos_a[midx], sin_a=sin_a[midx], speed=speed[midx], px=px, py=py, p_alive0=alive0, radii=radii_p, H=H, seg_len=seg_len[midx])
        contact = torch.full((M,), -1, dtype=contact_m.dtype, device=dev)
        eta_c = torch.full((M,), float(H), dtype=eta_cm.dtype, device=dev)
        contact[midx] = torch.where(keep, contact_m, torch.full_like(contact_m, -1))
        eta_c[midx] = torch.where(keep, eta_cm, torch.full_like(eta_cm, float(H)))
    viable = contact == tgt
    eta_out = torch.where(viable, eta_c.to(dt), torch.full_like(eta_c.to(dt), float('inf')))
    return {'angle': angle.reshape(shape), 'eta': eta_out.reshape(shape), 'viable': viable.reshape(shape)}

def _analytic_first_contact(*, launch_x: Tensor, launch_y: Tensor, cos_a: Tensor, sin_a: Tensor, speed: Tensor, px: Tensor, py: Tensor, p_alive0: Tensor, radii: Tensor, H: int, seg_len: Tensor | None=None, max_bytes: int=256 * 1024 * 1024):
    M = cos_a.shape[0]
    P = px.shape[-1]
    dev = cos_a.device
    dt = launch_x.dtype
    N = M
    big = _BIG
    lx = launch_x.reshape(N)
    ly = launch_y.reshape(N)
    ca = cos_a.reshape(N)
    sa = sin_a.reshape(N)
    sp = speed.reshape(N)
    slen = sp * float(H) if seg_len is None else seg_len.reshape(N)
    end_x = lx + ca * slen
    end_y = ly + sa * slen
    seg_xmin = torch.minimum(lx, end_x)
    seg_xmax = torch.maximum(lx, end_x)
    seg_ymin = torch.minimum(ly, end_y)
    seg_ymax = torch.maximum(ly, end_y)
    bb_xmin = px.amin(0) - radii
    bb_xmax = px.amax(0) + radii
    bb_ymin = py.amin(0) - radii
    bb_ymax = py.amax(0) + radii
    keep = ~((seg_xmax.unsqueeze(1) < bb_xmin) | (seg_xmin.unsqueeze(1) > bb_xmax) | (seg_ymax.unsqueeze(1) < bb_ymin) | (seg_ymin.unsqueeze(1) > bb_ymax))
    K = max(1, int(keep.sum(1).amax().item()))
    order = (~keep).to(torch.int8).argsort(dim=1, stable=True)
    shortlist = order[:, :K]
    valid = keep.gather(1, shortlist)
    k = torch.arange(H + 1, device=dev, dtype=dt)
    t_ax = torch.arange(H + 1, device=dev).view(1, H + 1, 1)
    step_h = torch.arange(1, H + 1, device=dev, dtype=dt).view(1, H, 1)
    bytes_per = max(1, 16 * H * K * 4)
    chunk = max(4096, max_bytes // bytes_per)
    chunk = min(chunk, max(N, 1))
    contacts: list[Tensor] = []
    etas: list[Tensor] = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sl = shortlist[s:e]
        fx = lx[s:e].view(-1, 1) + ca[s:e].view(-1, 1) * sp[s:e].view(-1, 1) * k
        fy = ly[s:e].view(-1, 1) + sa[s:e].view(-1, 1) * sp[s:e].view(-1, 1) * k
        sl_e = sl.view(-1, 1, K)
        pxc = px[t_ax, sl_e]
        pyc = py[t_ax, sl_e]
        radc = radii[sl]
        alivec = p_alive0[sl] & valid[s:e]
        real_slot = sl.to(dt)
        fx0 = fx[:, :-1].unsqueeze(-1)
        fy0 = fy[:, :-1].unsqueeze(-1)
        fx1 = fx[:, 1:].unsqueeze(-1)
        fy1 = fy[:, 1:].unsqueeze(-1)
        hit = _swept_pair_hit_mask(fx0, fy0, fx1, fy1, pxc[:, :-1, :], pyc[:, :-1, :], pxc[:, 1:, :], pyc[:, 1:, :], radc.unsqueeze(1))
        hit = hit & alivec.unsqueeze(1)
        planet_hit_step = torch.where(hit, step_h, torch.full_like(step_h, big)).amin(1)
        first_planet_step = planet_hit_step.amin(1)
        is_first = planet_hit_step == first_planet_step.unsqueeze(-1)
        contact_planet = torch.where(is_first, real_slot, torch.full_like(real_slot, big)).amin(1)
        nfx = fx[:, 1:]
        nfy = fy[:, 1:]
        ofx = fx[:, :-1]
        ofy = fy[:, :-1]
        oob = (nfx < 0) | (nfx > BOARD_SIZE) | (nfy < 0) | (nfy > BOARD_SIZE)
        vx = nfx - ofx
        vy = nfy - ofy
        wx = CENTER - ofx
        wy = CENTER - ofy
        vv = (vx * vx + vy * vy).clamp(min=1e-12)
        t = ((wx * vx + wy * vy) / vv).clamp(0.0, 1.0)
        cxp = ofx + t * vx
        cyp = ofy + t * vy
        sun = (cxp - CENTER) ** 2 + (cyp - CENTER) ** 2 < SUN_RADIUS * SUN_RADIUS
        env = oob | sun
        death_step = torch.where(env, step_h.squeeze(-1), torch.full_like(env, big, dtype=dt)).amin(1)
        ht = (first_planet_step <= death_step) & (first_planet_step < big)
        contacts.append(torch.where(ht, contact_planet, torch.full_like(contact_planet, -1.0)).long())
        etas.append(torch.where(ht, first_planet_step, torch.full_like(first_planet_step, float(H))))
    contact = (contacts[0] if len(contacts) == 1 else torch.cat(contacts)).view(M)
    eta = (etas[0] if len(etas) == 1 else torch.cat(etas)).view(M)
    return (contact, eta)

