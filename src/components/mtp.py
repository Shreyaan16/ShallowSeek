import torch
import torch.nn as nn
from src.components.norm import RMSNorm
from src.components.block import TransformerBlock
from src.entities.config_entity import MTPConfig, TransformerBlockConfig, RMSNormConfig
from src.entities.artifact_entity import MTPArtifact

class MTPModule(nn.Module):
    def __init__(self, cfg: MTPConfig, block_cfg: TransformerBlockConfig, shared_embed: nn.Embedding, shared_lm_head: nn.Linear):
        super().__init__()
        self.cfg = cfg
        norm_cfg = RMSNormConfig(d_model=cfg.d_model)
        self.norm_h = RMSNorm(norm_cfg)
        self.norm_e = RMSNorm(norm_cfg)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.block = TransformerBlock(block_cfg)
        self.norm_out = RMSNorm(norm_cfg)
        self.shared_embed = shared_embed
        self.shared_lm_head = shared_lm_head

    def forward(self, h_prev: torch.Tensor, token_ids_k: torch.Tensor):
        e_k = self.shared_embed(token_ids_k)
        fused = torch.cat([self.norm_h(h_prev), self.norm_e(e_k)], dim=-1)
        h_in = self.proj(fused)
        h_out, _ = self.block(h_in)
        return h_out, self.shared_lm_head(self.norm_out(h_out))

    def get_artifact(self) -> MTPArtifact:
        own = sum(p.numel() for p in self.parameters(recurse=False))
        block_p = sum(p.numel() for p in self.block.parameters())
        return MTPArtifact(n_mtp = self.cfg.n_mtp, d_model = self.cfg.d_model, n_params = own + block_p)

if __name__ == "__main__":
    cfg = MTPConfig()
    tfblock_cfg = TransformerBlockConfig()
    mtp_module = MTPModule(cfg, tfblock_cfg, nn.Embedding(1000, cfg.d_model), nn.Linear(cfg.d_model, 1000))
    print(mtp_module.get_artifact())