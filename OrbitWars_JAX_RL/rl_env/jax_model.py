import jax
import jax.numpy as jnp
import flax.linen as nn

class EntityTransformer(nn.Module):
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    num_fraction_bins: int = 10
    
    @nn.compact
    def __call__(self, p_features, f_features, p_active, f_active):
        """
        p_features: [Batch, P, 13]
        f_features: [Batch, F, 8]
        p_active:   [Batch, P] boolean
        f_active:   [Batch, F] boolean
        """
        # 1. Embeddings
        p_emb = nn.Dense(self.d_model, name='planet_encoder')(p_features)
        f_emb = nn.Dense(self.d_model, name='fleet_encoder')(f_features)
        
        # Thêm type embedding để mạng phân biệt được đâu là planet, đâu là fleet
        type_emb = self.param('type_emb', nn.initializers.normal(0.02), (2, self.d_model))
        p_emb = p_emb + type_emb[0]
        f_emb = f_emb + type_emb[1]
        
        # Nối lại thành 1 chuỗi sequence: [Batch, P + F, d_model]
        x = jnp.concatenate([p_emb, f_emb], axis=1)
        
        # 2. Attention Mask
        # Không cho phép attention vào các padded planets/fleets
        # active_mask: [Batch, P + F]
        active_mask = jnp.concatenate([p_active, f_active], axis=1)
        # padding_mask shape cho SelfAttention: [Batch, 1, 1, P+F] hoặc attention_mask [Batch, num_heads, P+F, P+F]
        # Hàm nn.make_attention_mask giúp tạo mask chuẩn
        attn_mask = nn.make_attention_mask(active_mask, active_mask, dtype=jnp.float32)
        
        # 3. Transformer Blocks
        for i in range(self.num_layers):
            # Self Attention
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.SelfAttention(num_heads=self.num_heads, name=f'attn_{i}')(x, mask=attn_mask)
            x = residual + x
            
            # FFN
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.d_model * 4)(x)
            x = nn.relu(x)
            x = nn.Dense(self.d_model)(x)
            x = residual + x
            
        # 4. Trích xuất lại Planet Embeddings
        P = p_features.shape[1]
        p_out = x[:, :P, :] # [Batch, P, d_model]
        
        # 5. Actor Heads
        # a. Target Head: Tính xác suất đánh từ hành tinh i sang hành tinh j
        # Dùng Dot-product giữa query(i) và key(j)
        query = nn.Dense(self.d_model, name='target_q')(p_out) # [B, P, d_model]
        key = nn.Dense(self.d_model, name='target_k')(p_out)   # [B, P, d_model]
        
        # target_logits: [B, P, P] (Batch, Source_Planet, Target_Planet)
        target_logits = jnp.einsum('bpd,bqd->bpq', query, key)
        
        # Mask out self-targeting (không đánh chính mình)
        diag_mask = jnp.eye(P, dtype=bool)[None, :, :] # [1, P, P]
        target_logits = jnp.where(diag_mask, -1e9, target_logits)
        
        # Mask out inactive targets (không đánh hành tinh ảo)
        valid_target_mask = p_active[:, None, :] # [B, 1, P]
        target_logits = jnp.where(valid_target_mask, target_logits, -1e9)
        
        # b. Fraction Head: Chọn % quân sẽ bắn (10 bins)
        fraction_logits = nn.Dense(self.num_fraction_bins, name='fraction_head')(p_out) # [Batch, P, 10]
        
        # 6. Value Head (Critic)
        # Pooling tất cả active planet embeddings để tính Value của cả bàn cờ
        p_out_masked = p_out * p_active[:, :, None]
        global_repr = jnp.sum(p_out_masked, axis=1) / (jnp.sum(p_active, axis=1, keepdims=True) + 1e-6)
        
        v = nn.Dense(self.d_model, name='v_hidden')(global_repr)
        v = nn.relu(v)
        value = nn.Dense(1, name='v_out')(v) # [Batch, 1]
        
        return target_logits, fraction_logits, value

if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    
    # Giả lập Batch_size = 2, P_MAX = 64, F_MAX = 256
    B, P, F = 2, 64, 256
    p_feat = jax.random.normal(rng, (B, P, 13))
    f_feat = jax.random.normal(rng, (B, F, 8))
    p_act = jnp.ones((B, P), dtype=bool)
    f_act = jnp.ones((B, F), dtype=bool)
    
    model = EntityTransformer(d_model=64, num_heads=2, num_layers=2)
    variables = model.init(rng, p_feat, f_feat, p_act, f_act)
    
    # Inference
    t_logits, f_logits, value = model.apply(variables, p_feat, f_feat, p_act, f_act)
    
    print("Transformer khởi tạo thành công!")
    print("Target Logits Shape:", t_logits.shape) # Kỳ vọng: (2, 64, 64)
    print("Fraction Logits Shape:", f_logits.shape) # Kỳ vọng: (2, 64, 10)
    print("Value Shape:", value.shape) # Kỳ vọng: (2, 1)
