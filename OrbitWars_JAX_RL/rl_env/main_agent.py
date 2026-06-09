import sys
import os
import traceback
import math

# --- FALLBACK AN TOÀN TUYỆT ĐỐI ---
try:
    import smb
except ImportError:
    smb = None

# Chúng ta thử import JAX và Flax, nếu bị lỗi (do Kaggle chưa cài) thì tịt JAX luôn
JAX_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp
    import flax.serialization
    import numpy as np
    from jax_env import EnvConfig, GameState
    from jax_features import extract_planet_features, extract_fleet_features
    from jax_model import EntityTransformer
    JAX_AVAILABLE = True
except Exception as e:
    print(f"Không thể tải JAX: {e}")

# --- BIẾN TOÀN CỤC JAX ---
_MODEL_INIT = False
_NETWORK = None
_PARAMS = None
_ENV_CFG = None

def obs_to_jax_state(obs: dict, config) -> 'GameState':
    P = config.max_planets
    F = config.max_fleets
    
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    
    p_active = np.zeros(P, dtype=bool)
    p_id = np.full(P, -1, dtype=np.int32)
    p_owner = np.full(P, -1, dtype=np.int32)
    p_x = np.zeros(P, dtype=np.float32)
    p_y = np.zeros(P, dtype=np.float32)
    p_r = np.zeros(P, dtype=np.float32)
    p_ships = np.zeros(P, dtype=np.float32)
    p_prod = np.zeros(P, dtype=np.float32)
    
    for i, p in enumerate(planets[:P]):
        pid, owner, x, y, r, ships, prod = p[:7]
        p_active[i] = True
        p_id[i] = int(pid)
        p_owner[i] = int(owner)
        p_x[i] = float(x)
        p_y[i] = float(y)
        p_r[i] = float(r)
        p_ships[i] = float(ships)
        p_prod[i] = float(prod)
        
    f_active = np.zeros(F, dtype=bool)
    f_id = np.full(F, -1, dtype=np.int32)
    f_owner = np.full(F, -1, dtype=np.int32)
    f_x = np.zeros(F, dtype=np.float32)
    f_y = np.zeros(F, dtype=np.float32)
    f_angle = np.zeros(F, dtype=np.float32)
    f_speed = np.zeros(F, dtype=np.float32)
    f_ships = np.zeros(F, dtype=np.float32)
    f_target_id = np.full(F, -1, dtype=np.int32)
    
    for i, f in enumerate(fleets[:F]):
        fid, owner, x, y, angle, from_id, ships = f[:7]
        f_active[i] = True
        f_id[i] = int(fid)
        f_owner[i] = int(owner)
        f_x[i] = float(x)
        f_y[i] = float(y)
        f_angle[i] = float(angle)
        f_speed[i] = config.max_ship_speed
        f_ships[i] = float(ships)
        f_target_id[i] = -1
        
    return GameState(
        step=jnp.int32(obs.get("step", 0)),
        p_active=jnp.array(p_active),
        p_id=jnp.array(p_id),
        p_owner=jnp.array(p_owner),
        p_x=jnp.array(p_x),
        p_y=jnp.array(p_y),
        p_r=jnp.array(p_r),
        p_ships=jnp.array(p_ships),
        p_prod=jnp.array(p_prod),
        f_active=jnp.array(f_active),
        f_id=jnp.array(f_id),
        f_owner=jnp.array(f_owner),
        f_x=jnp.array(f_x),
        f_y=jnp.array(f_y),
        f_angle=jnp.array(f_angle),
        f_speed=jnp.array(f_speed),
        f_ships=jnp.array(f_ships),
        f_target_id=jnp.array(f_target_id),
        player_count=jnp.int32(2),
        next_fleet_id=jnp.int32(obs.get("next_fleet_id", 0)),
        angular_velocity=jnp.float32(obs.get("angular_velocity", 0.03))
    )

def init_model():
    global _MODEL_INIT, _NETWORK, _PARAMS, _ENV_CFG
    if _MODEL_INIT:
        return
    
    _ENV_CFG = EnvConfig()
    _NETWORK = EntityTransformer()
    
    rng = jax.random.PRNGKey(0)
    B, P, F = 1, _ENV_CFG.max_planets, _ENV_CFG.max_fleets
    dummy_p_feat = jnp.zeros((B, P, 13))
    dummy_f_feat = jnp.zeros((B, F, 8))
    dummy_p_act = jnp.zeros((B, P), dtype=bool)
    dummy_f_act = jnp.zeros((B, F), dtype=bool)
    
    _PARAMS = _NETWORK.init(rng, dummy_p_feat, dummy_f_feat, dummy_p_act, dummy_f_act)
    
    weight_path_1 = os.path.join(os.path.dirname(__file__), "weights_super_ai.msgpack")
    weight_path_2 = os.path.join(os.path.dirname(__file__), "weights.msgpack")
    
    weight_path = weight_path_1 if os.path.exists(weight_path_1) else weight_path_2
    
    if os.path.exists(weight_path):
        with open(weight_path, "rb") as f:
            bytes_data = f.read()
        _PARAMS = flax.serialization.from_bytes(_PARAMS, bytes_data)
        print(f"Đã tải trọng số JAX từ {weight_path} thành công!")
    else:
        print(f"CẢNH BÁO: Không tìm thấy {weight_path_1} hoặc {weight_path_2}. Dùng trọng số ngẫu nhiên!")
        
    _MODEL_INIT = True

if JAX_AVAILABLE:
    @jax.jit
    def jax_predict(params, p_feat, f_feat, p_act, f_act):
        target_logits, fraction_logits, _ = _NETWORK.apply(params, p_feat, f_feat, p_act, f_act)
        best_targets = jnp.argmax(target_logits, axis=-1)
        best_fractions = jnp.argmax(fraction_logits, axis=-1)
        return best_targets[0], best_fractions[0]

def jax_agent(obs, config):
    init_model()
    state = obs_to_jax_state(obs, _ENV_CFG)
    
    player = obs.get('player', 0) if isinstance(obs, dict) else getattr(obs, 'player', 0)
    player_id = int(player)
    
    p_feat = extract_planet_features(state, _ENV_CFG, player_id)
    f_feat = extract_fleet_features(state, _ENV_CFG, player_id)
    
    p_feat_b = jnp.expand_dims(p_feat, 0)
    f_feat_b = jnp.expand_dims(f_feat, 0)
    p_act_b = jnp.expand_dims(state.p_active, 0)
    f_act_b = jnp.expand_dims(state.f_active, 0)
    
    best_targets, best_fractions = jax_predict(_PARAMS, p_feat_b, f_feat_b, p_act_b, f_act_b)
    
    actions = []
    targets_np = np.array(best_targets)
    fractions_np = np.array(best_fractions)
    p_owner_np = np.array(state.p_owner)
    p_active_np = np.array(state.p_active)
    p_ships_np = np.array(state.p_ships)
    p_x_np = np.array(state.p_x)
    p_y_np = np.array(state.p_y)
    
    planets_list = obs.get("planets", [])
    P_count = len(planets_list)
    for i in range(P_count):
        if not p_active_np[i]: continue
        if p_owner_np[i] != player_id: continue
        
        target_id = int(targets_np[i])
        fraction_bin = int(fractions_np[i])
        
        if target_id == i or not p_active_np[target_id]:
            continue
            
        fraction_val = (fraction_bin + 1.0) / 10.0
        ships_to_send = int(p_ships_np[i] * fraction_val)
        
        if ships_to_send > 0:
            dx = float(p_x_np[target_id] - p_x_np[i])
            dy = float(p_y_np[target_id] - p_y_np[i])
            angle = math.atan2(dy, dx)
            # Kaggle Orbit Wars action format: [from_planet_id, angle, ships]
            actions.append([int(i), float(angle), int(ships_to_send)])
            
    return actions

def agent(obs, config=None):
    # Chuẩn hoá nếu obs là object (trong Kaggle) thay vì dictionary
    if not isinstance(obs, dict):
        obs_dict = {
            "step": getattr(obs, "step", 0),
            "player": getattr(obs, "player", 0),
            "planets": getattr(obs, "planets", []),
            "fleets": getattr(obs, "fleets", []),
            "next_fleet_id": getattr(obs, "next_fleet_id", 0),
            "angular_velocity": getattr(obs, "angular_velocity", 0.03),
            "episode_steps": getattr(obs, "episode_steps", 500),
            "remainingOverageTime": getattr(obs, "remainingOverageTime", 2.0)
        }
    else:
        obs_dict = obs

    try:
        if JAX_AVAILABLE:
            return jax_agent(obs_dict, config)
        else:
            raise Exception("JAX không khả dụng")
    except Exception as e:
        print("====== LỖI JAX RL - KÍCH HOẠT FALLBACK ======")
        traceback.print_exc()
        
        if smb is not None:
            print("Đang gọi lại smb.py heuristic agent...")
            return smb.agent(obs)
        return []
