import sys
import os
import jax
import jax.numpy as jnp
import flax.serialization
import traceback

# Cần import smb từ thư mục gốc hoặc nơi chứa nó
# Trong môi trường thi đấu, ta nén cả rl_env và smb.py thành tar.gz
try:
    import smb
except ImportError:
    # Fallback nếu không tìm thấy
    smb = None

# Import JAX components
from jax_env import EnvConfig, GameState, obs_to_jax_state
from jax_features import extract_planet_features, extract_fleet_features
from jax_model import EntityTransformer

# --- BIẾN TOÀN CỤC ---
_MODEL_INIT = False
_NETWORK = None
_PARAMS = None
_ENV_CFG = None

def init_model():
    """Tải trọng số mô hình vào RAM (Chỉ chạy 1 lần ở Step 0)"""
    global _MODEL_INIT, _NETWORK, _PARAMS, _ENV_CFG
    if _MODEL_INIT:
        return
    
    _ENV_CFG = EnvConfig()
    _NETWORK = EntityTransformer()
    
    # Tạo dummy inputs để lấy cấu trúc tham số
    rng = jax.random.PRNGKey(0)
    B, P, F = 1, _ENV_CFG.max_planets, _ENV_CFG.max_fleets
    dummy_p_feat = jnp.zeros((B, P, 13))
    dummy_f_feat = jnp.zeros((B, F, 8))
    dummy_p_act = jnp.zeros((B, P), dtype=bool)
    dummy_f_act = jnp.zeros((B, F), dtype=bool)
    
    _PARAMS = _NETWORK.init(rng, dummy_p_feat, dummy_f_feat, dummy_p_act, dummy_f_act)
    
    # Load trọng số từ file weights.msgpack
    weight_path = os.path.join(os.path.dirname(__file__), "weights.msgpack")
    if os.path.exists(weight_path):
        with open(weight_path, "rb") as f:
            bytes_data = f.read()
        _PARAMS = flax.serialization.from_bytes(_PARAMS, bytes_data)
        print("Đã tải trọng số JAX thành công!")
    else:
        print(f"CẢNH BÁO: Không tìm thấy {weight_path}. Dùng trọng số ngẫu nhiên!")
        
    _MODEL_INIT = True

@jax.jit
def jax_predict(params, p_feat, f_feat, p_act, f_act):
    """JIT Compiled hàm dự đoán hành động"""
    target_logits, fraction_logits, _ = _NETWORK.apply(params, p_feat, f_feat, p_act, f_act)
    
    # Lấy argmax cho target và fraction
    best_targets = jnp.argmax(target_logits, axis=-1) # [B, P]
    best_fractions = jnp.argmax(fraction_logits, axis=-1) # [B, P]
    
    return best_targets[0], best_fractions[0] # Lấy phần tử Batch 0

def jax_agent(obs, config):
    """Agent JAX chính"""
    init_model()
    
    # 1. Chuyển đổi Obs sang State tĩnh của JAX
    state = obs_to_jax_state(obs, _ENV_CFG)
    
    # 2. Trích xuất đặc trưng
    player_id = obs["player"]
    p_feat = extract_planet_features(state, _ENV_CFG, player_id)
    f_feat = extract_fleet_features(state, _ENV_CFG, player_id)
    
    # Thêm chiều Batch (B=1)
    p_feat_b = jnp.expand_dims(p_feat, 0)
    f_feat_b = jnp.expand_dims(f_feat, 0)
    p_act_b = jnp.expand_dims(state.p_active, 0)
    f_act_b = jnp.expand_dims(state.f_active, 0)
    
    # 3. Chạy Mô hình nơ-ron (JIT)
    best_targets, best_fractions = jax_predict(_PARAMS, p_feat_b, f_feat_b, p_act_b, f_act_b)
    
    # 4. Giải mã hành động
    # fraction_bins: 0 -> 10%, 9 -> 100%
    actions = []
    
    import numpy as np
    targets_np = np.array(best_targets)
    fractions_np = np.array(best_fractions)
    p_owner_np = np.array(state.p_owner)
    p_active_np = np.array(state.p_active)
    p_ships_np = np.array(state.p_ships)
    
    P_count = len(obs["planets"])
    for i in range(P_count):
        if not p_active_np[i]: continue
        if p_owner_np[i] != player_id: continue # Chỉ điều khiển quân mình
        
        target_id = int(targets_np[i])
        fraction_bin = int(fractions_np[i])
        
        # Nếu mô hình khuyên đánh chính mình -> Bỏ qua (Do Nothing)
        if target_id == i or not p_active_np[target_id]:
            continue
            
        # Tính số quân
        fraction_val = (fraction_bin + 1.0) / 10.0
        ships_to_send = int(p_ships_np[i] * fraction_val)
        
        if ships_to_send > 0:
            actions.append((i, target_id, ships_to_send))
            
    return actions

def agent(obs, config):
    """
    Hàm Giao diện chính với cơ chế Fallback an toàn tuyệt đối.
    Thử dùng JAX RL trước, nếu Crash thì gọi smb.py ra đỡ đòn.
    """
    try:
        # Bước đầu tiên JAX sẽ mất ~3s để biên dịch (JIT), các bước sau < 0.01s
        action = jax_agent(obs, config)
        return action
    except Exception as e:
        print("====== LỖI JAX RL - KÍCH HOẠT FALLBACK ======")
        traceback.print_exc()
        
        if smb is not None:
            print("Đang gọi lại smb.py heuristic agent...")
            return smb.agent(obs, config)
        else:
            print("Không có smb.py, đành không làm gì cả.")
            return []
