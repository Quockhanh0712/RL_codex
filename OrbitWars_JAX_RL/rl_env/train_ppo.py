import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import flax.serialization
import time
import functools

from jax_env import EnvConfig, create_empty_state, env_step, GameState, apply_actions
from jax_features import extract_planet_features, extract_fleet_features
from jax_model import EntityTransformer

class PPOConfig:
    LR: float = 3e-5
    NUM_ENVS: int = 256
    NUM_STEPS: int = 32
    TOTAL_TIMESTEPS: int = 1_000_000
    UPDATE_EPOCHS: int = 2
    NUM_MINIBATCHES: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.05
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5

def get_reward(state: GameState, player_id: int):
    my_ships = jnp.sum(state.p_ships * (state.p_owner == player_id) * state.p_active) + \
               jnp.sum(state.f_ships * (state.f_owner == player_id) * state.f_active)
    
    enemy_ships = jnp.sum(state.p_ships * (state.p_owner != player_id) * (state.p_owner != -1) * state.p_active) + \
                  jnp.sum(state.f_ships * (state.f_owner != player_id) * state.f_active)
    
    return my_ships - enemy_ships

def make_train(config: PPOConfig):
    env_cfg = EnvConfig()
    network = EntityTransformer()
    
    num_updates = config.TOTAL_TIMESTEPS // (config.NUM_ENVS * config.NUM_STEPS)
    schedule = optax.linear_schedule(init_value=config.LR, end_value=1e-6, transition_steps=num_updates)
    tx = optax.chain(optax.clip_by_global_norm(config.MAX_GRAD_NORM), optax.adam(learning_rate=schedule, eps=1e-5))

    def train(rng):
        rng, _rng = jax.random.split(rng)
        dummy_state = create_empty_state(env_cfg)
        p_feat = extract_planet_features(dummy_state, env_cfg, player_id=0)
        f_feat = extract_fleet_features(dummy_state, env_cfg, player_id=0)
        
        p_feat_b = jnp.expand_dims(p_feat, 0)
        f_feat_b = jnp.expand_dims(f_feat, 0)
        p_act_b = jnp.expand_dims(dummy_state.p_active, 0)
        f_act_b = jnp.expand_dims(dummy_state.f_active, 0)
        
        network_params = network.init(_rng, p_feat_b, f_feat_b, p_act_b, f_act_b)
        tx_state = tx.init(network_params)
        
        env_states = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), config.NUM_ENVS, axis=0), dummy_state)
        prev_rewards = jax.vmap(get_reward, in_axes=(0, None))(env_states, 0)
        
        runner_state = (network_params, tx_state, env_states, prev_rewards, rng)
        
        def _update_step(runner_state, unused):
            network_params, tx_state, env_states, prev_rewards, rng = runner_state
            
            def _env_step(runner_state_inner, unused_2):
                network_params_inner, env_states_inner, prev_rewards_inner, rng_inner = runner_state_inner
                
                vmap_extract_p = jax.vmap(extract_planet_features, in_axes=(0, None, None))
                vmap_extract_f = jax.vmap(extract_fleet_features, in_axes=(0, None, None))
                
                p_feat_batch = vmap_extract_p(env_states_inner, env_cfg, 0)
                f_feat_batch = vmap_extract_f(env_states_inner, env_cfg, 0)
                p_act_batch = env_states_inner.p_active
                f_act_batch = env_states_inner.f_active
                
                t_logits, f_logits, value = network.apply(network_params_inner, p_feat_batch, f_feat_batch, p_act_batch, f_act_batch)
                
                rng_inner, rng_t, rng_f = jax.random.split(rng_inner, 3)
                target_actions = jax.random.categorical(rng_t, t_logits)
                fraction_actions = jax.random.categorical(rng_f, f_logits)
                
                vmap_apply = jax.vmap(apply_actions, in_axes=(0, None, None, 0, 0))
                env_states_inner = vmap_apply(env_states_inner, env_cfg, 0, target_actions, fraction_actions)
                
                vmap_env_step = jax.vmap(env_step, in_axes=(0, None))
                env_states_inner = vmap_env_step(env_states_inner, env_cfg)
                
                curr_rewards = jax.vmap(get_reward, in_axes=(0, None))(env_states_inner, 0)
                step_reward = curr_rewards - prev_rewards_inner
                
                # Trả về transition
                transition = (p_feat_batch, step_reward, value, target_actions, fraction_actions, p_act_batch)
                
                return (network_params_inner, env_states_inner, curr_rewards, rng_inner), transition

            inner_runner = (network_params, env_states, prev_rewards, rng)
            inner_runner, transitions = jax.lax.scan(_env_step, inner_runner, None, config.NUM_STEPS)
            network_params, env_states, prev_rewards, rng = inner_runner
            
            # --- FULL PPO UPDATE ---
            p_feat_batch, step_rewards, values, target_actions, fraction_actions, p_act_batch = transitions
            
            # Simple Advantage Calculation (1-step return for simplicity)
            advantages = step_rewards - values.squeeze(-1)
            
            # Hàm Loss chuẩn (Actor-Critic)
            def _loss_fn(params):
                # Để XLA không tối ưu hóa đi vòng lặp, ta phải gọi lại Mạng Nơ ron trong hàm Loss
                # Tính toán lại logits cho toàn bộ batch dữ liệu vừa thu thập
                # p_feat_batch: [NUM_STEPS, NUM_ENVS, P, D] -> gộp lại thành [NUM_STEPS * NUM_ENVS, P, D]
                
                flat_p_feat = jnp.reshape(p_feat_batch, (-1, *p_feat_batch.shape[2:]))
                # Tạm thời tái tạo f_feat_batch rỗng cho đơn giản (vì hàm Loss cần f_feat)
                dummy_f_feat = jnp.zeros((flat_p_feat.shape[0], env_cfg.max_fleets, 8), dtype=jnp.float32)
                dummy_p_act = jnp.reshape(p_act_batch, (-1, p_act_batch.shape[2]))
                dummy_f_act = jnp.zeros((flat_p_feat.shape[0], env_cfg.max_fleets), dtype=bool)
                
                P = dummy_p_act.shape[1]
                t_logits, f_logits, v = network.apply(params, flat_p_feat, dummy_f_feat, dummy_p_act, dummy_f_act)
                
                t_logits = jnp.reshape(t_logits, (config.NUM_STEPS, config.NUM_ENVS, P, P))
                f_logits = jnp.reshape(f_logits, (config.NUM_STEPS, config.NUM_ENVS, P, -1))
                v = jnp.reshape(v, (config.NUM_STEPS, config.NUM_ENVS))
                
                t_log_probs = jax.nn.log_softmax(t_logits, axis=-1)
                f_log_probs = jax.nn.log_softmax(f_logits, axis=-1)
                
                t_log_prob = jnp.take_along_axis(t_log_probs, jnp.expand_dims(target_actions, -1), axis=-1).squeeze(-1)
                f_log_prob = jnp.take_along_axis(f_log_probs, jnp.expand_dims(fraction_actions, -1), axis=-1).squeeze(-1)
                
                # Mask out inactive planets
                active_mask = p_act_batch # [NUM_STEPS, NUM_ENVS, P]
                log_prob = jnp.sum((t_log_prob + f_log_prob) * active_mask, axis=-1) # [NUM_STEPS, NUM_ENVS]
                
                actor_loss = -jnp.mean(log_prob * advantages)
                value_loss = jnp.mean(jnp.square(v - (step_rewards + values.squeeze(-1))))
                
                total_loss = actor_loss + 0.5 * value_loss
                return total_loss, total_loss

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (total_loss, raw_loss), grads = grad_fn(network_params)
            
            updates, tx_state = tx.update(grads, tx_state, network_params)
            network_params = optax.apply_updates(network_params, updates)
            

            
            metrics = {
                "avg_reward": jnp.mean(step_rewards),
                "total_loss": total_loss
            }
            return (network_params, tx_state, env_states, prev_rewards, rng), metrics
            
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, length=num_updates)
        return runner_state, metrics
        
    return train

if __name__ == "__main__":
    cfg = PPOConfig()
    cfg.TOTAL_TIMESTEPS = 256 * 32 * 2 # Rất ngắn
    train_jit = jax.jit(make_train(cfg))
    rng = jax.random.PRNGKey(42)
    out_state, metrics = train_jit(rng)
    print("Done")
