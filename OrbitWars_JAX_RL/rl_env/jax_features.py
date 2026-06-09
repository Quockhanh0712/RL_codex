import jax
import jax.numpy as jnp
from jax_env import GameState, EnvConfig, env_step

import functools

@functools.partial(jax.jit, static_argnums=(2,))
def rollout_future(state: GameState, config: EnvConfig, horizon: int = 30):
    """
    Dự phóng tương lai H bước bằng cách gọi `env_step` liên tục.
    (Chỉ mô phỏng các hạm đội hiện có, không có hành động mới).
    """
    def step_fn(carry_state, _):
        next_state = env_step(carry_state, config)
        return next_state, (next_state.p_owner, next_state.p_ships)
    
    final_state, (future_owner, future_ships) = jax.lax.scan(step_fn, state, jnp.arange(horizon))
    # future_owner: [H, P]
    # future_ships: [H, P]
    return future_owner, future_ships

@jax.jit
def extract_planet_features(state: GameState, config: EnvConfig, player_id: int) -> jax.Array:
    """
    Trích xuất ma trận đặc trưng [P, Feature_Dim] cho Entity Transformer.
    Tính năng cốt lõi được kế thừa từ tư duy của `smb.py`.
    """
    P = state.p_x.shape[0]
    
    # 1. Các thuộc tính tĩnh & hiện tại
    norm_x = (state.p_x - config.center) / config.rot_radius_limit
    norm_y = (state.p_y - config.center) / config.rot_radius_limit
    norm_r = state.p_r / 5.0  # Max radius thường là ~4-5
    
    norm_ships = state.p_ships / 100.0
    norm_prod = state.p_prod / 5.0
    
    is_mine_bool = (state.p_owner == player_id)
    is_neutral_bool = (state.p_owner == -1)
    is_enemy_bool = (~is_mine_bool) & (~is_neutral_bool) & (state.p_owner != 3)
    
    is_mine = is_mine_bool.astype(jnp.float32)
    is_neutral = is_neutral_bool.astype(jnp.float32)
    is_enemy = is_enemy_bool.astype(jnp.float32)
    
    # 2. Đặc trưng dự phóng tương lai (The smb.py advantage)
    future_owner, future_ships = rollout_future(state, config, horizon=30)
    
    # Trạng thái ở step 10 và 20
    ships_at_10 = future_ships[9] / 100.0
    ships_at_30 = future_ships[29] / 100.0
    
    # 3. Tính Safe Drain (Thủ) và Capture Floor (Công)
    # Giả sử mình sở hữu hành tinh, số quân thấp nhất trong tương lai là bao nhiêu?
    # Nếu enemy đánh, future_ships sẽ tụt, hoặc đổi chủ.
    # Ta tính `min_ships_my` = giá trị nhỏ nhất của ships trong tương lai NẾU vẫn thuộc về mình.
    # Nếu đổi chủ sang địch, xem như ships rơi xuống số âm (để an toàn ta gán 0).
    is_mine_future = (future_owner == player_id)
    my_ships_future = jnp.where(is_mine_future, future_ships, 0.0)
    # minimum over time (axis 0)
    safe_drain = jnp.min(my_ships_future, axis=0) / 100.0
    
    # Tính Capture Floor (đối với hành tinh địch/neutral): 
    # Là số lượng lớn nhất quân địch có trên đó trước khi mình đến.
    capture_floor = jnp.max(future_ships, axis=0) / 100.0
    
    # 4. Gộp toàn bộ lại thành Feature Matrix
    # Shape of each array is [P], we stack them to [P, D]
    features = jnp.stack([
        state.p_active.astype(jnp.float32), # Mask xem planet có tồn tại không
        norm_x,
        norm_y,
        norm_r,
        norm_ships,
        norm_prod,
        is_mine,
        is_enemy,
        is_neutral,
        ships_at_10,
        ships_at_30,
        safe_drain,
        capture_floor
    ], axis=1) # [P, 13]
    
    # 5. Mask out inactive planets
    features = features * state.p_active[:, None]
    
    return features

@jax.jit
def extract_fleet_features(state: GameState, config: EnvConfig, player_id: int) -> jax.Array:
    """
    Trích xuất đặc trưng cho hạm đội [F, D_fleet]
    """
    norm_x = (state.f_x - config.center) / config.rot_radius_limit
    norm_y = (state.f_y - config.center) / config.rot_radius_limit
    
    norm_ships = state.f_ships / 100.0
    
    is_mine = (state.f_owner == player_id).astype(jnp.float32)
    is_enemy = (state.f_owner != player_id).astype(jnp.float32)
    
    # Vector hướng vận tốc
    vx = jnp.cos(state.f_angle)
    vy = jnp.sin(state.f_angle)
    
    features = jnp.stack([
        state.f_active.astype(jnp.float32),
        norm_x,
        norm_y,
        vx,
        vy,
        norm_ships,
        is_mine,
        is_enemy
    ], axis=1) # [F, 8]
    
    features = features * state.f_active[:, None]
    
    return features

if __name__ == "__main__":
    from jax_env import EnvConfig, create_empty_state
    
    cfg = EnvConfig()
    state = create_empty_state(cfg)
    
    # Setup 1 planet, 1 fleet để test
    state = state._replace(
        p_active=state.p_active.at[0].set(True),
        p_owner=state.p_owner.at[0].set(1), # Địch
        p_ships=state.p_ships.at[0].set(10.0),
        p_prod=state.p_prod.at[0].set(1.0),
        
        f_active=state.f_active.at[0].set(True),
        f_owner=state.f_owner.at[0].set(0), # Mình
        f_x=state.f_x.at[0].set(50.0),
        f_y=state.f_y.at[0].set(50.0),
        f_angle=state.f_angle.at[0].set(0.0),
        f_speed=state.f_speed.at[0].set(6.0),
        f_ships=state.f_ships.at[0].set(20.0),
    )
    
    p_feat = extract_planet_features(state, cfg, player_id=0)
    f_feat = extract_fleet_features(state, cfg, player_id=0)
    
    print("Planet Features Shape:", p_feat.shape)
    print("Fleet Features Shape:", f_feat.shape)
    print("Planet 0 features:", p_feat[0])
    
    # Kiểm tra Rollout:
    future_owner, future_ships = rollout_future(state, cfg, horizon=5)
    print("Future Ships (5 steps) of Planet 0:", future_ships[:, 0])
