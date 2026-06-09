!pip install -q flax optax

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import optax
import flax.serialization
import functools

# KẾT NỐI VỚI DATASET
dataset_path = Path("/kaggle/input/datasets/quockhanhh05/ob0077")
sys.path.insert(0, str(dataset_path))

from jax_env import EnvConfig, env_step, GameState, apply_actions
from jax_features import extract_planet_features, extract_fleet_features
from jax_model import EntityTransformer

class PPOConfig:
    LR: float = 3e-5
    NUM_ENVS: int = 64     
    NUM_STEPS: int = 16    
    TOTAL_TIMESTEPS: int = 1_000_000_000
    GAMMA: float = 0.99
    MAX_GRAD_NORM: float = 0.5
    UPDATES_PER_CHUNK: int = 1000  

# --- KHỞI TẠO BẢN ĐỒ NGẪU NHIÊN ---
def reset_env(rng, config: EnvConfig) -> GameState:
    P, F = config.max_planets, config.max_fleets
    rng_p, rng_owner, rng_x, rng_y, rng_r, rng_ships, rng_prod = jax.random.split(rng, 7)
    
    num_planets = jax.random.randint(rng_p, (), 15, 40) # 15 đến 40 hành tinh
    indices = jnp.arange(P)
    p_active = indices < num_planets
    p_id = jnp.where(p_active, indices, -1)
    
    owner_rands = jax.random.uniform(rng_owner, (P,))
    # 15% là của Player 0, 15% của Player 1, 70% Neutral (-1)
    p_owner = jnp.where(owner_rands < 0.15, 0,
                jnp.where(owner_rands < 0.30, 1, -1))
    p_owner = jnp.where(p_active, p_owner, -1)
    
    # Đảm bảo P0 và P1 luôn có ít nhất 1 hành tinh
    p_owner = p_owner.at[0].set(0)
    p_owner = p_owner.at[1].set(1)
    p_active = p_active.at[0].set(True)
    p_active = p_active.at[1].set(True)
    
    p_x = jax.random.uniform(rng_x, (P,), minval=10.0, maxval=config.board_size - 10.0)
    p_y = jax.random.uniform(rng_y, (P,), minval=10.0, maxval=config.board_size - 10.0)
    p_r = jax.random.uniform(rng_r, (P,), minval=2.0, maxval=8.0)
    
    p_ships = jax.random.uniform(rng_ships, (P,), minval=20.0, maxval=100.0)
    p_prod = jax.random.uniform(rng_prod, (P,), minval=0.5, maxval=2.0)
    
    f_active=jnp.zeros(F, dtype=bool)
    f_id=jnp.full(F, -1, dtype=jnp.int32)
    f_owner=jnp.full(F, -1, dtype=jnp.int32)
    f_x=jnp.zeros(F, dtype=jnp.float32)
    f_y=jnp.zeros(F, dtype=jnp.float32)
    f_angle=jnp.zeros(F, dtype=jnp.float32)
    f_speed=jnp.zeros(F, dtype=jnp.float32)
    f_ships=jnp.zeros(F, dtype=jnp.float32)
    f_target_id=jnp.full(F, -1, dtype=jnp.int32)
    
    return GameState(
        step=jnp.int32(0),
        p_active=p_active, p_id=p_id, p_owner=p_owner, p_x=p_x, p_y=p_y, p_r=p_r, p_ships=p_ships, p_prod=p_prod,
        f_active=f_active, f_id=f_id, f_owner=f_owner, f_x=f_x, f_y=f_y, f_angle=f_angle, f_speed=f_speed, f_ships=f_ships, f_target_id=f_target_id,
        player_count=jnp.int32(2), next_fleet_id=jnp.int32(0), angular_velocity=jnp.float32(0.0)
    )

def check_done_and_reset(state: GameState, rng, config: EnvConfig):
    # Trận đấu kết thúc sau 500 bước
    done = state.step >= config.episode_steps
    new_state = reset_env(rng, config)
    return jax.tree_util.tree_map(lambda n_x, x: jnp.where(done, n_x, x), new_state, state)

def get_reward(state: GameState, player_id: int):
    my_ships = jnp.sum(state.p_ships * (state.p_owner == player_id) * state.p_active) + \
               jnp.sum(state.f_ships * (state.f_owner == player_id) * state.f_active)
    enemy_ships = jnp.sum(state.p_ships * (state.p_owner != player_id) * (state.p_owner != -1) * state.p_active) + \
                  jnp.sum(state.f_ships * (state.f_owner != player_id) * state.f_active)
    return my_ships - enemy_ships

env_cfg = EnvConfig()
network = EntityTransformer()
cfg = PPOConfig()

num_devices = jax.local_device_count()
total_updates = cfg.TOTAL_TIMESTEPS // (cfg.NUM_ENVS * cfg.NUM_STEPS * num_devices)
schedule = optax.linear_schedule(init_value=cfg.LR, end_value=1e-6, transition_steps=total_updates)
tx = optax.chain(optax.clip_by_global_norm(cfg.MAX_GRAD_NORM), optax.adam(learning_rate=schedule, eps=1e-5))

# 1. HÀM KHỞI TẠO BỘ NÃO (Chỉ chạy 1 lần)
def init_runner(rng):
    rng, _rng = jax.random.split(rng)
    dummy_state = reset_env(_rng, env_cfg)
    p_feat = extract_planet_features(dummy_state, env_cfg, player_id=0)
    f_feat = extract_fleet_features(dummy_state, env_cfg, player_id=0)
    
    p_feat_b, f_feat_b = jnp.expand_dims(p_feat, 0), jnp.expand_dims(f_feat, 0)
    p_act_b, f_act_b = jnp.expand_dims(dummy_state.p_active, 0), jnp.expand_dims(dummy_state.f_active, 0)
    
    network_params = network.init(_rng, p_feat_b, f_feat_b, p_act_b, f_act_b)
    tx_state = tx.init(network_params)
    
    rngs_envs = jax.random.split(rng, cfg.NUM_ENVS)
    env_states = jax.vmap(reset_env, in_axes=(0, None))(rngs_envs, env_cfg)
    prev_rewards = jax.vmap(get_reward, in_axes=(0, None))(env_states, 0)
    
    return (network_params, tx_state, env_states, prev_rewards, rng)

# 2. HÀM CHẠY LẶP THEO TỪNG CỤC (Chunk)
def train_chunk(runner_state):
    def _update_step(runner_state, unused):
        network_params, tx_state, env_states, prev_rewards, rng = runner_state
        
        def _env_step(runner_state_inner, unused_2):
            network_params_inner, env_states_inner, prev_rewards_inner, rng_inner = runner_state_inner
            
            vmap_extract_p = jax.vmap(extract_planet_features, in_axes=(0, None, None))
            vmap_extract_f = jax.vmap(extract_fleet_features, in_axes=(0, None, None))
            p_feat_batch = vmap_extract_p(env_states_inner, env_cfg, 0)
            f_feat_batch = vmap_extract_f(env_states_inner, env_cfg, 0)
            p_act_batch, f_act_batch = env_states_inner.p_active, env_states_inner.f_active
            
            t_logits, f_logits, value = network.apply(network_params_inner, p_feat_batch, f_feat_batch, p_act_batch, f_act_batch)
            
            rng_inner, rng_t, rng_f = jax.random.split(rng_inner, 3)
            target_actions = jax.random.categorical(rng_t, t_logits)
            fraction_actions = jax.random.categorical(rng_f, f_logits)
            
            vmap_apply = jax.vmap(apply_actions, in_axes=(0, None, None, 0, 0))
            env_states_inner = vmap_apply(env_states_inner, env_cfg, 0, target_actions, fraction_actions)
            vmap_env_step = jax.vmap(env_step, in_axes=(0, None))
            env_states_inner = vmap_env_step(env_states_inner, env_cfg)
            
            # --- RESET NẾU TRẬN ĐẤU KẾT THÚC (STEP >= 500) ---
            rng_inner, rng_reset = jax.random.split(rng_inner)
            rngs_reset = jax.random.split(rng_reset, cfg.NUM_ENVS)
            vmap_reset = jax.vmap(check_done_and_reset, in_axes=(0, 0, None))
            env_states_inner = vmap_reset(env_states_inner, rngs_reset, env_cfg)
            
            curr_rewards = jax.vmap(get_reward, in_axes=(0, None))(env_states_inner, 0)
            step_reward = curr_rewards - prev_rewards_inner
            transition = (p_feat_batch, step_reward, value, target_actions, fraction_actions, p_act_batch)
            return (network_params_inner, env_states_inner, curr_rewards, rng_inner), transition

        inner_runner = (network_params, env_states, prev_rewards, rng)
        inner_runner, transitions = jax.lax.scan(_env_step, inner_runner, None, cfg.NUM_STEPS)
        network_params, env_states, prev_rewards, rng = inner_runner
        
        p_feat_batch, step_rewards, values, target_actions, fraction_actions, p_act_batch = transitions
        advantages = step_rewards - values.squeeze(-1)
        
        def _loss_fn(params):
            flat_p_feat = jnp.reshape(p_feat_batch, (-1, *p_feat_batch.shape[2:]))
            dummy_f_feat = jnp.zeros((flat_p_feat.shape[0], env_cfg.max_fleets, 8), dtype=jnp.float32)
            dummy_p_act = jnp.reshape(p_act_batch, (-1, p_act_batch.shape[2]))
            dummy_f_act = jnp.zeros((flat_p_feat.shape[0], env_cfg.max_fleets), dtype=bool)
            
            P = dummy_p_act.shape[1]
            t_logits, f_logits, v = network.apply(params, flat_p_feat, dummy_f_feat, dummy_p_act, dummy_f_act)
            t_logits = jnp.reshape(t_logits, (cfg.NUM_STEPS, cfg.NUM_ENVS, P, P))
            f_logits = jnp.reshape(f_logits, (cfg.NUM_STEPS, cfg.NUM_ENVS, P, -1))
            v = jnp.reshape(v, (cfg.NUM_STEPS, cfg.NUM_ENVS))
            
            t_log_probs = jax.nn.log_softmax(t_logits, axis=-1)
            f_log_probs = jax.nn.log_softmax(f_logits, axis=-1)
            t_log_prob = jnp.take_along_axis(t_log_probs, jnp.expand_dims(target_actions, -1), axis=-1).squeeze(-1)
            f_log_prob = jnp.take_along_axis(f_log_probs, jnp.expand_dims(fraction_actions, -1), axis=-1).squeeze(-1)
            
            active_mask = p_act_batch
            log_prob = jnp.sum((t_log_prob + f_log_prob) * active_mask, axis=-1)
            
            actor_loss = -jnp.mean(log_prob * advantages)
            value_loss = jnp.mean(jnp.square(v - (step_rewards + values.squeeze(-1))))
            return actor_loss + 0.5 * value_loss, actor_loss + 0.5 * value_loss

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (total_loss, raw_loss), grads = grad_fn(network_params)
        
        grads = jax.lax.pmean(grads, axis_name='batch')
        total_loss = jax.lax.pmean(total_loss, axis_name='batch')
        
        updates, tx_state = tx.update(grads, tx_state, network_params)
        network_params = optax.apply_updates(network_params, updates)
        
        metrics = {"avg_reward": jnp.mean(step_rewards), "total_loss": total_loss}
        return (network_params, tx_state, env_states, prev_rewards, rng), metrics
        
    runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, length=cfg.UPDATES_PER_CHUNK)
    return runner_state, metrics

# --- KHỞI CHẠY MULTI-GPU (TỰ ĐỘNG LƯU) ---
print(f"🔥 BẮT ĐẦU HUẤN LUYỆN MULTI-GPU TRÊN {num_devices} THIẾT BỊ T4 🔥")

pmapped_init = jax.pmap(init_runner)
pmapped_train = jax.pmap(train_chunk, axis_name='batch')

rngs = jax.random.split(jax.random.PRNGKey(42), num_devices)
runner_state = pmapped_init(rngs)

total_chunks = total_updates // cfg.UPDATES_PER_CHUNK

for chunk_idx in range(total_chunks):
    print(f"Đang chạy Cục (Chunk) {chunk_idx + 1}/{total_chunks}...")
    
    runner_state, metrics = pmapped_train(runner_state)
    
    loss_val = metrics["total_loss"][0][-1] 
    reward_val = metrics["avg_reward"][0][-1]
    print(f"   -> Xong chunk {chunk_idx + 1}! Loss: {loss_val:.4f} | Reward: {reward_val:.4f}")
    
    network_params_gpu0 = jax.tree_util.tree_map(lambda x: x[0], runner_state[0])
    with open("weights_super_ai.msgpack", "wb") as f_out:
        f_out.write(flax.serialization.to_bytes(network_params_gpu0))
    print(f"   -> Đã lưu tự động an toàn vào 'weights_super_ai.msgpack'!")

print("🏆 CHÚC MỪNG! Đã hoàn thành toàn bộ quá trình Train 1 tỷ bước!")
