import torch
import torch.nn as nn
import torch.nn.functional as F
from src.components.norm import RMSNorm
from src.components.block import TransformerBlock
from src.components.mtp import MTPModule
from src.entities.config_entity import ModelConfig, RMSNormConfig
from src.entities.artifact_entity import ModelArtifact

class ShallowSeek(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.block.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg.block) for _ in range(cfg.n_layers)])
        self.norm_out = RMSNorm(RMSNormConfig(d_model=cfg.block.d_model))
        self.lm_head = nn.Linear(cfg.block.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.mtp_modules = nn.ModuleList([MTPModule(cfg.mtp, cfg.block, self.embed, self.lm_head)for _ in range(cfg.mtp.n_mtp)])

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None,
                start_pos: int = 0, kv_caches: list = None):
        h = self.embed(x)

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            h, new_cache = block(h, start_pos=start_pos, kv_cache=cache_i)
            new_kv_caches.append(new_cache)

        main_logits = self.lm_head(self.norm_out(h))

        if targets is None:
            return main_logits, None, new_kv_caches

        n_mtp  = self.cfg.mtp.n_mtp
        T  = x.shape[1]
        T_eff  = T - n_mtp - 1
        main_loss = F.cross_entropy(main_logits[:, :T_eff].reshape(-1, self.cfg.vocab_size), targets[:, :T_eff].reshape(-1))

        h_prev = h[:, :T_eff]
        mtp_losses = []
        for k, module in enumerate(self.mtp_modules):
            token_ids_k = x[:, k + 1 : T_eff + k + 1]
            target_k    = x[:, k + 2 : T_eff + k + 2]
            h_prev, logits_k = module(h_prev, token_ids_k)
            mtp_losses.append(F.cross_entropy(logits_k.reshape(-1, self.cfg.vocab_size), target_k.reshape(-1)))
        mtp_loss   = sum(mtp_losses) / n_mtp
        total_loss = main_loss + self.cfg.mtp.mtp_lambda * mtp_loss
        return main_logits, total_loss, new_kv_caches

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None, eos_id: int = None):
        self.eval()
        # prefill: process full prompt, build kv cache
        logits, _, kv_caches = self.forward(prompt_ids, start_pos=0)
        start_pos = prompt_ids.shape[1]

        next_token = self._sample(logits[:, -1], temperature, top_k)    # (B,)
        generated  = [next_token]

        for _ in range(max_new_tokens - 1):
            logits, _, kv_caches = self.forward(
                next_token.unsqueeze(1), start_pos=start_pos, kv_caches=kv_caches)
            next_token = self._sample(logits[:, -1], temperature, top_k)
            generated.append(next_token)
            start_pos += 1
            if eos_id is not None and (next_token == eos_id).all():
                break

        return torch.stack(generated, dim=1)                             # (B, n_generated)

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = None):
        if temperature == 0.0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)

    def get_artifact(self) -> ModelArtifact:
        total  = sum(p.numel() for p in self.parameters())
        unique = sum(p.numel() for p in set(self.parameters()))
        return ModelArtifact(vocab_size = self.cfg.vocab_size, d_model = self.cfg.block.d_model, 
                             n_layers = self.cfg.n_layers, total_params = total, unique_params = unique)


if __name__ == "__main__":
    cfg   = ModelConfig()
    model = ShallowSeek(cfg)
    art   = model.get_artifact()

    print(f"vocab_size    : {art.vocab_size:,}")
    print(f"d_model       : {art.d_model}")
    print(f"n_layers      : {art.n_layers}")
    print(f"total params  : {art.total_params:,}")
    print(f"unique params : {art.unique_params:,}")

    B, T = 4, cfg.rope.max_seq_len
    x    = torch.randint(0, cfg.vocab_size, (B, T))
    y    = torch.randint(0, cfg.vocab_size, (B, T))

    model.train()
    logits, loss, _ = model(x, y)
    print(f"\nlogits : {tuple(logits.shape)}")
    print(f"loss   : {loss.item():.4f}  (main + {cfg.mtp.mtp_lambda} × avg_mtp)")

    model.eval()
    with torch.no_grad():
        logits_inf, _, _ = model(x[:1, :8])
    print(f"inference logits : {tuple(logits_inf.shape)}")

    generated = model.generate(x[:1, :8], max_new_tokens=10, temperature=1.0, top_k=50)
    print(f"generated tokens : {tuple(generated.shape)}")
