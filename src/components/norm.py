import torch
import torch.nn as nn
from src.entities.config_entity import RMSNormConfig
from src.entities.artifact_entity import RMSNormArtifact

class RMSNorm(nn.Module):
    def __init__(self, cfg: RMSNormConfig):
        super().__init__()
        self.cfg = cfg
        self.eps = cfg.eps
        self.weight = nn.Parameter(torch.ones(cfg.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

    def get_artifact(self) -> RMSNormArtifact:
        return RMSNormArtifact(d_model = self.cfg.d_model, eps = self.cfg.eps, n_params = self.weight.numel())

if __name__ == "__main__":
    cfg = RMSNormConfig(d_model=128, eps=1e-8)
    model = RMSNorm(cfg)
    x = torch.randn(2, 3, 128)
    output = model(x)
    print("Output shape:", output.shape)
    artifact = model.get_artifact()
    print("Artifact:", artifact)