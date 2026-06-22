import torch
import torch.nn as nn
import torch.nn.functional as F
from src.entities.config_entity import ExpertConfig, MoEConfig
from src.entities.artifact_entity import ExpertArtifact, MoEArtifact


class Expert(nn.Module):
    def __init__(self, cfg: ExpertConfig):
        super().__init__()
        self.cfg = cfg
        self.w1  = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2  = nn.Linear(cfg.d_ff,    cfg.d_model, bias=False)
        self.w3  = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def get_artifact(self) -> ExpertArtifact:
        n = sum(p.numel() for p in self.parameters())
        return ExpertArtifact(d_model=self.cfg.d_model, d_ff=self.cfg.d_ff, n_params=n)


class ShallowSeekMoE(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.n_routed_experts = cfg.n_routed_experts
        self.top_k = cfg.top_k
        shared_cfg = ExpertConfig(d_model=cfg.d_model, d_ff=cfg.d_ff_shared)
        routed_cfg = ExpertConfig(d_model=cfg.d_model, d_ff=cfg.d_ff_routed)
        self.shared_experts = nn.ModuleList([Expert(shared_cfg) for _ in range(cfg.n_shared_experts)])
        self.routed_experts = nn.ModuleList([Expert(routed_cfg) for _ in range(cfg.n_routed_experts)])
        self.gate = nn.Linear(cfg.d_model, cfg.n_routed_experts, bias=False)
        self.register_buffer("expert_biases", torch.zeros(cfg.n_routed_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C    = x.shape
        flat       = x.view(-1, C)
        num_tokens = flat.size(0)

        shared_out = torch.zeros_like(flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(flat)
        logits         = self.gate(flat) + self.expert_biases.unsqueeze(0)
        scores         = F.softmax(logits, dim=-1)
        topk_w, topk_i = torch.topk(scores, self.top_k, dim=-1)
        topk_w         = topk_w / topk_w.sum(dim=-1, keepdim=True)
        routed_out    = torch.zeros_like(flat)
        actual_counts = torch.zeros(self.n_routed_experts, device=x.device)

        for i, expert in enumerate(self.routed_experts):
            mask = (topk_i == i)
            if mask.any():
                tok_idx, k_idx = torch.where(mask)
                weights = topk_w[tok_idx, k_idx].unsqueeze(-1)
                routed_out[tok_idx] = routed_out[tok_idx] + weights * expert(flat[tok_idx])
                actual_counts[i] += tok_idx.size(0)

        if self.training:
            target = (num_tokens * self.top_k) / self.n_routed_experts
            self.expert_biases = self.expert_biases + self.cfg.bias_update_rate * (target - actual_counts)

        return (shared_out + routed_out).view(B, T, C)

    def get_artifact(self) -> MoEArtifact:
        shared_p = sum(p.numel() for e in self.shared_experts for p in e.parameters())
        routed_p = sum(p.numel() for e in self.routed_experts for p in e.parameters())
        active_p = shared_p + (routed_p // self.cfg.n_routed_experts) * self.cfg.top_k
        return MoEArtifact(n_shared_experts = self.cfg.n_shared_experts, n_routed_experts = self.cfg.n_routed_experts, top_k = self.cfg.top_k,
                          active_params = active_p, total_params = shared_p + routed_p + self.gate.weight.numel())

if __name__ == "__main__":
    moe_cfg = MoEConfig(d_model=512, d_ff_shared=2048, d_ff_routed=4096, n_shared_experts=2, n_routed_experts=4, top_k=2)
    moe = ShallowSeekMoE(moe_cfg)
    x = torch.randn(8, 16, 512)  # Batch of 8 sequences, each of length 16, with model dimension 512
    output = moe(x)
    print(output.shape)  # Should be (8, 16, 512)