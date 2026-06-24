import torch
import torch.nn as nn
import torch.nn.functional as F
from src.components.rope import RotaryEmbedding
from src.entities.config_entity import MLAConfig, RoPEConfig
from src.entities.artifact_entity import MLAArtifact

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head_nope = cfg.d_head_nope
        self.d_head_rope = cfg.d_head_rope
        self.d_head = cfg.d_head_nope + cfg.d_head_rope
        self.scale = self.d_head ** -0.5
        self.rope = RotaryEmbedding(RoPEConfig(d_rope=cfg.d_head_rope))
        self.W_DKV = nn.Linear(cfg.d_model, cfg.d_c_kv, bias=False)
        self.W_UK = nn.Linear(cfg.d_c_kv, cfg.n_heads * cfg.d_head_nope, bias=False)
        self.W_UV = nn.Linear(cfg.d_c_kv, cfg.n_heads * cfg.d_head_nope, bias=False)
        self.W_DQ = nn.Linear(cfg.d_model, cfg.d_c_q, bias=False)
        self.W_UQ = nn.Linear(cfg.d_c_q, cfg.n_heads * cfg.d_head_nope, bias=False)
        self.W_QR = nn.Linear(cfg.d_c_q, cfg.n_heads * cfg.d_head_rope, bias=False)
        self.W_KR = nn.Linear(cfg.d_model, cfg.d_head_rope, bias=False)  # head-shared
        self.W_O = nn.Linear(cfg.n_heads * cfg.d_head_nope, cfg.d_model,  bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor, start_pos: int = 0, kv_cache: dict = None):
        B, T, _ = h.shape

        def to_heads(x, d_h, s):
            return x.view(B, s, self.n_heads, d_h).transpose(1, 2)

        c_q  = self.W_DQ(h)
        c_kv = self.W_DKV(h)                                             # (B, T, d_c_kv)
        k_rope_new = self.rope(self.W_KR(h).unsqueeze(1), start_pos)    # (B, 1, T, d_head_rope)

        if kv_cache is not None:
            c_kv_full   = torch.cat([kv_cache["c_kv"],   c_kv],         dim=1)   # (B, S+T, d_c_kv)
            k_rope_full = torch.cat([kv_cache["k_rope"], k_rope_new],   dim=2)   # (B, 1, S+T, d_head_rope)
        else:
            c_kv_full   = c_kv
            k_rope_full = k_rope_new

        S = c_kv_full.shape[1] # Total Sequence length of the Keys (K) and Values (V) that the current Queries (Q) can attend to.

        K_nope = to_heads(self.W_UK(c_kv_full), self.d_head_nope, S)
        V      = to_heads(self.W_UV(c_kv_full), self.d_head_nope, S)
        K_rope = k_rope_full.expand(-1, self.n_heads, -1, -1)           # (B, n_heads, S, d_head_rope)
        K      = torch.cat([K_nope, K_rope], dim=-1)

        Q_nope = to_heads(self.W_UQ(c_q), self.d_head_nope, T)
        Q_rope = self.rope(to_heads(self.W_QR(c_q), self.d_head_rope, T), start_pos)
        Q      = torch.cat([Q_nope, Q_rope], dim=-1)

        attn = (Q @ K.transpose(-2, -1)) * self.scale                   # (B, n_heads, T, S)
        q_idx = torch.arange(start_pos, start_pos + T, device=h.device).unsqueeze(1)
        k_idx = torch.arange(S, device=h.device).unsqueeze(0)
        mask  = q_idx >= k_idx                                           # (T, S) causal mask
        attn  = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn  = self.drop(F.softmax(attn, dim=-1))

        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head_nope)
        new_cache = {"c_kv": c_kv_full, "k_rope": k_rope_full}
        return self.W_O(out), new_cache

    def get_artifact(self) -> MLAArtifact:
        return MLAArtifact(output_dim = self.cfg.d_model, kv_cache_per_token = self.cfg.d_c_kv + self.cfg.d_head_rope, 
                           n_params = sum(p.numel() for p in self.parameters()))

if __name__ == "__main__":
    cfg = MLAConfig()
    model = MultiHeadLatentAttention(cfg)
    x = torch.randn(2, 4, cfg.d_model)
    out, cache = model(x)
    print(out.shape)  # Expected output: torch.Size([2, 4, cfg.d_model])