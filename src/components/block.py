import torch
import torch.nn as nn
from src.components.norm import RMSNorm
from src.components.attention import MultiHeadLatentAttention
from src.components.moe import ShallowSeekMoE
from src.entities.config_entity import TransformerBlockConfig, RMSNormConfig
from src.entities.artifact_entity import TransformerBlockArtifact

class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerBlockConfig):
        super().__init__()
        self.cfg = cfg
        norm_cfg = RMSNormConfig(d_model=cfg.d_model)
        self.norm1 = RMSNorm(norm_cfg)
        self.attn = MultiHeadLatentAttention(cfg.mla)
        self.norm2 = RMSNorm(norm_cfg)
        self.moe = ShallowSeekMoE(cfg.moe)

    def forward(self, x: torch.Tensor, start_pos: int = 0, kv_cache: dict = None):
        attn_out, new_cache = self.attn(self.norm1(x), start_pos=start_pos, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.moe(self.norm2(x))
        return x, new_cache

    def get_artifact(self) -> TransformerBlockArtifact:
        return TransformerBlockArtifact(d_model  = self.cfg.d_model, n_params = sum(p.numel() for p in self.parameters()))

if __name__ == "__main__":
    cfg = TransformerBlockConfig()
    block = TransformerBlock(cfg)
    x = torch.randn(2, 4, cfg.d_model)
    out, cache = block(x)
    print(out.shape)  # Expected output: torch.Size([2, 4, cfg.d_model])