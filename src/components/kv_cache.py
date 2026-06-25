import torch
from src.entities.config_entity import KVCacheConfig


class KVCache:
    """Pre-allocated, write-in-place KV cache with sliding-window eviction.
    When the buffer is full (filled == max_seq_len) the oldest tokens are
    shifted out to make room for new ones.  `offset` tracks how many tokens
    have been evicted so the attention mask can use correct absolute positions.
    """

    def __init__(self, cfg: KVCacheConfig, device="cpu"):
        self.max_seq_len = cfg.max_seq_len
        self.filled  = 0
        self.offset  = 0  # absolute position of slot 0 in the buffer
        self.c_kv    = torch.zeros(cfg.B, cfg.max_seq_len, cfg.d_c_kv, device=device)
        self.k_rope  = torch.zeros(cfg.B, 1, cfg.max_seq_len, cfg.d_head_rope, device=device)

    def update(self, c_kv_new: torch.Tensor, k_rope_new: torch.Tensor, start_pos: int):
        T  = c_kv_new.shape[1]
        end = start_pos + T

        if end > self.max_seq_len:
            # sliding window: evict oldest tokens to make room
            shift = end - self.max_seq_len
            keep  = self.max_seq_len - shift
            if keep > 0:
                self.c_kv[:, :keep, :]      = self.c_kv[:, shift:, :].clone()
                self.k_rope[:, :, :keep, :] = self.k_rope[:, :, shift:, :].clone()
            tail = min(T, self.max_seq_len)                    # clamp if T itself > window
            self.c_kv[:, self.max_seq_len - tail:, :]      = c_kv_new[:, T - tail:, :]
            self.k_rope[:, :, self.max_seq_len - tail:, :] = k_rope_new[:, :, T - tail:, :]
            self.offset = end - self.max_seq_len
            self.filled = self.max_seq_len
        else:
            self.c_kv[:, start_pos:end, :]      = c_kv_new
            self.k_rope[:, :, start_pos:end, :] = k_rope_new
            self.filled = end

    def get(self):
        return self.c_kv[:, :self.filled, :], self.k_rope[:, :, :self.filled, :]

    def clone(self) -> "KVCache":
        new             = KVCache.__new__(KVCache)
        new.max_seq_len = self.max_seq_len
        new.filled      = self.filled
        new.offset      = self.offset
        new.c_kv        = self.c_kv.clone()
        new.k_rope      = self.k_rope.clone()
        return new
