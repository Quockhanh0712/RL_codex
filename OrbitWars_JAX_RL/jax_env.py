import jax
import jax.numpy as jnp
from typing import NamedTuple

class EnvConfig(NamedTuple):
    board_size: float = 100.0
    center: float = 50.0
    sun_radius: float = 10.0
    max_ship_speed: float = 6.0
    rot_radius_limit: float = 50.0
    comet_events: int = 5
    comets_per_event: int = 4
    comet_spawn_steps: tuple = (50, 150, 250, 350, 450)
    comet_radius: float = 1.0
    comet_production: float = 1.0
    max_planets: int = 64
    max_fleets: int = 256
    episode_steps: int = 500

class GameState(NamedTuple):
    step: jnp.int32
    p_active: jnp.ndarray
    p_id: jnp.ndarray
    p_owner: jnp.ndarray       # 0, 1, 2, 3. -1 is Neutral.
    p_x: jnp.ndarray
    p_y: jnp.ndarray
    p_r: jnp.ndarray
    p_ships: jnp.ndarray
    p_prod: jnp.ndarray
    
    f_active: jnp.ndarray
    f_id: jnp.ndarray
    f_owner: jnp.ndarray       # 0, 1, 2, 3
    f_x: jnp.ndarray
    f_y: jnp.ndarray
    f_angle: jnp.ndarray
    f_speed: jnp.ndarray
    f_ships: jnp.ndarray
    f_target_id: jnp.ndarray
    
    player_count: jnp.int32
    next_fleet_id: jnp.int32
    angular_velocity: jnp.float32

def create_empty_state(config: EnvConfig) -> GameState:
    P, F = config.max_planets, config.max_fleets
    return GameState(
        step=jnp.int32(0),
        p_active=jnp.zeros(P, dtype=bool),
        p_id=jnp.full(P, -1, dtype=jnp.int32),
        p_owner=jnp.full(P, -1, dtype=jnp.int32),
        p_x=jnp.zeros(P, dtype=jnp.float32),
        p_y=jnp.zeros(P, dtype=jnp.float32),
        p_r=jnp.zeros(P, dtype=jnp.float32),
        p_ships=jnp.zeros(P, dtype=jnp.float32),
        p_prod=jnp.zeros(P, dtype=jnp.float32),
        f_active=jnp.zeros(F, dtype=bool),
        f_id=jnp.full(F, -1, dtype=jnp.int32),
        f_owner=jnp.full(F, -1, dtype=jnp.int32),
        f_x=jnp.zeros(F, dtype=jnp.float32),
        f_y=jnp.zeros(F, dtype=jnp.float32),
        f_angle=jnp.zeros(F, dtype=jnp.float32),
        f_speed=jnp.zeros(F, dtype=jnp.float32),
        f_ships=jnp.zeros(F, dtype=jnp.float32),
        f_target_id=jnp.full(F, -1, dtype=jnp.int32),
        player_count=jnp.int32(2),
        next_fleet_id=jnp.int32(0),
        angular_velocity=jnp.float32(0.0)
    )

@jax.jit
def move_fleets(state: GameState, config: EnvConfig) -> tuple[GameState, jax.Array, jax.Array]:
    """Di chuyển hạm đội, trả về state mới và (old_f_x, old_f_y) để dùng cho Combat."""
    old_f_x, old_f_y = state.f_x, state.f_y
    
    dx = state.f_speed * jnp.cos(state.f_angle)
    dy = state.f_speed * jnp.sin(state.f_angle)
    
    new_f_x = state.f_x + dx
    new_f_y = state.f_y + dy
    
    dist_to_sun_sq = (new_f_x - config.center)**2 + (new_f_y - config.center)**2
    sun_radius_sq = config.sun_radius**2
    margin = 20.0
    max_radius_sq = (config.rot_radius_limit + margin)**2
    
    destroyed = (dist_to_sun_sq <= sun_radius_sq) | (dist_to_sun_sq > max_radius_sq)
    new_f_active = state.f_active & (~destroyed)
    
    new_state = state._replace(
        f_x=jnp.where(new_f_active, new_f_x, state.f_x),
        f_y=jnp.where(new_f_active, new_f_y, state.f_y),
        f_active=new_f_active
    )
    return new_state, old_f_x, old_f_y

@jax.jit
def resolve_landing_and_combat(state: GameState, config: EnvConfig, old_f_x: jax.Array, old_f_y: jax.Array) -> GameState:
    """Xử lý Hạm đội đáp xuống Hành tinh và Giao tranh."""
    P = state.p_x.shape[0]
    F = state.f_x.shape[0]
    
    # 1. Tính toán va chạm bằng khoảng cách từ điểm (Planet) tới đoạn thẳng (Fleet Path)
    ax, ay = old_f_x[:, None], old_f_y[:, None]       # [F, 1]
    bx, by = state.f_x[:, None], state.f_y[:, None]   # [F, 1]
    px, py = state.p_x[None, :], state.p_y[None, :]   # [1, P]
    
    l2 = jnp.maximum((bx - ax)**2 + (by - ay)**2, 1e-6)
    t = jnp.clip(((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2, 0.0, 1.0)
    proj_x = ax + t * (bx - ax)
    proj_y = ay + t * (by - ay)
    
    dist_sq = (px - proj_x)**2 + (py - proj_y)**2
    r_sq = state.p_r[None, :] ** 2
    
    # Hit mask: [F, P]
    hit_mask = (dist_sq <= r_sq) & state.f_active[:, None] & state.p_active[None, :]
    
    # Chọn planet gần điểm xuất phát A nhất nếu hit nhiều planet
    dist_A_to_hit_sq = (proj_x - ax)**2 + (proj_y - ay)**2
    hit_dist = jnp.where(hit_mask, dist_A_to_hit_sq, jnp.inf)
    
    first_hit_p = jnp.argmin(hit_dist, axis=1) # [F]
    valid_hit = jnp.min(hit_dist, axis=1) != jnp.inf # [F]
    
    new_f_active = state.f_active & (~valid_hit)
    fleet_hit_mask = jax.nn.one_hot(first_hit_p, P, dtype=bool) & valid_hit[:, None] # [F, P]
    
    # 2. Tính tổng quân đến từng hành tinh theo Player ID
    # Player IDs: 0, 1, 2, 3 và Neutral (-1). Ta map -1 -> 4 để dùng index 0..4
    num_players = 5 
    
    def calc_arr_for_player(idx):
        player_id = jnp.where(idx == 4, -1, idx)
        is_player = (state.f_owner == player_id)[:, None] # [F, 1]
        player_hit = fleet_hit_mask & is_player # [F, P]
        return jnp.sum(jnp.where(player_hit, state.f_ships[:, None], 0.0), axis=0) # [P]
        
    arr_by_player = jax.vmap(calc_arr_for_player)(jnp.arange(num_players)) # [5, P]
    
    # 3. Áp dụng Combat Logic
    def resolve_planet(p_idx):
        pre_owner = state.p_owner[p_idx]
        pre_ships = state.p_ships[p_idx]
        
        # JAX array index: Nếu pre_owner = -1, nó tự động lấy index cuối cùng (4) -> Rất tuyệt!
        my_arr = arr_by_player[pre_owner, p_idx]
        def_ships = pre_ships + my_arr
        
        enemy_arr = arr_by_player[:, p_idx].at[pre_owner].set(0.0)
        
        sum_enemy = jnp.sum(enemy_arr)
        max_enemy_idx = jnp.argmax(enemy_arr)
        # Chuyển đổi ngược: Nếu max_enemy_idx == 4 thì đó là Neutral (-1)
        max_enemy_player = jnp.where(max_enemy_idx == 4, -1, max_enemy_idx)
        
        is_capture = sum_enemy > def_ships
        
        new_owner = jnp.where(is_capture, max_enemy_player, pre_owner)
        new_ships = jnp.where(is_capture, sum_enemy - def_ships, def_ships - sum_enemy)
        
        # Đảm bảo các hành tinh inactive không bị thay đổi
        return new_owner, new_ships

    new_p_owner, new_p_ships = jax.vmap(resolve_planet)(jnp.arange(P))
    
    return state._replace(
        f_active=new_f_active,
        p_owner=jnp.where(state.p_active, new_p_owner, state.p_owner),
        p_ships=jnp.where(state.p_active, new_p_ships, state.p_ships)
    )

@jax.jit
def env_step(state: GameState, config: EnvConfig) -> GameState:
    """Step hoàn chỉnh: Move -> Combat -> Production."""
    # 1. Di chuyển hạm đội
    state, old_f_x, old_f_y = move_fleets(state, config)
    
    # 2. Hạ cánh & Giao tranh
    state = resolve_landing_and_combat(state, config, old_f_x, old_f_y)
    
    # 3. Sản xuất quân (Chỉ áp dụng cho các hành tinh không phải Neutral/Dead)
    # Trong Orbit Wars, Neutral cũng có thể sản xuất nếu là Comet, nhưng logic chung là p_prod
    # Chỉ active planets mới sản xuất
    new_ships = state.p_ships + state.p_prod
    state = state._replace(
        p_ships=jnp.where(state.p_active, new_ships, state.p_ships),
        step=state.step + 1
    )
    
    return state

if __name__ == "__main__":
    cfg = EnvConfig()
    state = create_empty_state(cfg)
    
    # Setup test:
    # Planet 0: Neutral (-1), X=80, Y=50, R=4, 10 ships
    # Fleet 0: P1 (0), X=70, Y=50, Speed=6, 15 ships, Angle=0 (bay sang phải)
    
    state = state._replace(
        p_active=state.p_active.at[0].set(True),
        p_id=state.p_id.at[0].set(0),
        p_owner=state.p_owner.at[0].set(-1),
        p_x=state.p_x.at[0].set(80.0),
        p_y=state.p_y.at[0].set(50.0),
        p_r=state.p_r.at[0].set(4.0),
        p_ships=state.p_ships.at[0].set(10.0),
        p_prod=state.p_prod.at[0].set(0.0), # No prod for simpler test
        
        f_active=state.f_active.at[0].set(True),
        f_owner=state.f_owner.at[0].set(0),
        f_x=state.f_x.at[0].set(70.0),
        f_y=state.f_y.at[0].set(50.0),
        f_angle=state.f_angle.at[0].set(0.0), # 0 radian = sang phải
        f_speed=state.f_speed.at[0].set(6.0),
        f_ships=state.f_ships.at[0].set(15.0),
    )
    
    # Bước 0 -> Fleet bay từ X=50 đến X=56. Vẫn cách Planet (X=60) khoảng X=4, vừa đúng bằng Radius=4
    # Nên step 1 sẽ HIT!
    state = env_step(state, cfg)
    
    print(f"Step 1: Fleet active={state.f_active[0]}")
    print(f"Step 1: Planet owner={state.p_owner[0]} (Kỳ vọng: 0)")
    print(f"Step 1: Planet ships={state.p_ships[0]} (Kỳ vọng: 15 - 10 = 5)")


@jax.jit
def apply_actions(state: GameState, config: EnvConfig, player_id: int, target_actions: jax.Array, fraction_actions: jax.Array) -> GameState:
    P = state.p_x.shape[0]
    F = state.f_x.shape[0]
    
    is_mine = (state.p_owner == player_id) & state.p_active
    do_launch = is_mine & (target_actions != jnp.arange(P))
    
    fraction_vals = (fraction_actions + 1.0) / 10.0
    send_ships = jnp.floor(state.p_ships * fraction_vals)
    do_launch = do_launch & (send_ships >= 1.0)
    new_p_ships = jnp.where(do_launch, state.p_ships - send_ships, state.p_ships)
    
    num_launches = jnp.sum(do_launch)
    sort_idx = jnp.argsort(do_launch)[::-1]
    
    comp_s_idx = jnp.arange(P)[sort_idx]
    comp_s_owner = state.p_owner[sort_idx]
    comp_s_target = target_actions[sort_idx]
    comp_s_ships = send_ships[sort_idx]
    
    slot_sort_idx = jnp.argsort(~state.f_active)[::-1]
    update_indices = slot_sort_idx[:P]
    
    valid_update = jnp.arange(P) < jnp.minimum(num_launches, jnp.sum(~state.f_active))
    
    new_f_owner = state.f_owner.at[update_indices].set(jnp.where(valid_update, comp_s_owner, state.f_owner[update_indices]))
    new_f_ships = state.f_ships.at[update_indices].set(jnp.where(valid_update, comp_s_ships, state.f_ships[update_indices]))
    new_f_target_id = state.f_target_id.at[update_indices].set(jnp.where(valid_update, comp_s_target, state.f_target_id[update_indices]))
    
    comp_s_x = state.p_x[comp_s_idx]
    comp_s_y = state.p_y[comp_s_idx]
    new_f_x = state.f_x.at[update_indices].set(jnp.where(valid_update, comp_s_x, state.f_x[update_indices]))
    new_f_y = state.f_y.at[update_indices].set(jnp.where(valid_update, comp_s_y, state.f_y[update_indices]))
    
    new_f_speed = state.f_speed.at[update_indices].set(jnp.where(valid_update, 0.1, state.f_speed[update_indices]))
    new_f_active = state.f_active.at[update_indices].set(jnp.where(valid_update, True, state.f_active[update_indices]))
    
    return state._replace(
        p_ships=new_p_ships,
        f_owner=new_f_owner,
        f_ships=new_f_ships,
        f_target_id=new_f_target_id,
        f_x=new_f_x,
        f_y=new_f_y,
        f_speed=new_f_speed,
        f_active=new_f_active
    )
