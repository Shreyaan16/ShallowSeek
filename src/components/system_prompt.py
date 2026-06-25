import torch
from src.components.kv_cache import KVCache
from src.entities.config_entity import KVCacheConfig


class SystemPromptCache:
    """
    Runs a single prefill pass over fixed system-prompt tokens and caches the result.
    Clone the cache for each new user query instead of re-processing the system prefix.
    """

    def __init__(self, model, sys_ids: list, max_seq_len: int, device):
        sys_tensor = torch.tensor([sys_ids], dtype=torch.long, device=device)
        kv_cfg = KVCacheConfig(
            B=1,
            max_seq_len=max_seq_len,
            d_c_kv=model.cfg.block.mla.d_c_kv,
            d_head_rope=model.cfg.block.mla.d_head_rope,
        )
        self._caches = [KVCache(kv_cfg, device) for _ in range(model.cfg.n_layers)]

        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(sys_tensor, start_pos=0, kv_caches=self._caches)
        if was_training:
            model.train()

        self.sys_len = self._caches[0].filled

    def clone_caches(self) -> list:
        return [c.clone() for c in self._caches]
