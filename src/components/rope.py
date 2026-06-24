import torch
import torch.nn as nn
from src.entities.config_entity import RoPEConfig
from src.entities.artifact_entity import RoPEArtifact

class RotaryEmbedding(nn.Module):
    def __init__(self, cfg: RoPEConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.d_rope % 2 == 0
        inv_freq = 1.0 / (cfg.theta ** (torch.arange(0, cfg.d_rope, 2).float() / cfg.d_rope))
        t  = torch.arange(cfg.max_seq_len, dtype=torch.float32)
        angles = torch.outer(t, inv_freq)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)  # (T, d_rope//2) complex64
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Rotate x using pre-computed complex frequencies.
        x: (B, H, T, d_rope)  ->  (B, H, T, d_rope) rotated
        start_pos: position offset for cached decoding
        """
        B, H, T, d = x.shape
        x_c = torch.view_as_complex(x.float().reshape(B, H, T, d // 2, 2))
        f  = self.freqs_cis[start_pos : start_pos + T].unsqueeze(0).unsqueeze(0)
        out = torch.view_as_real(x_c * f).reshape(B, H, T, d)
        return out.to(x.dtype)

    def get_artifact(self) -> RoPEArtifact:
        return RoPEArtifact(d_rope = self.cfg.d_rope, max_seq_len = self.cfg.max_seq_len, shape = tuple(self.freqs_cis.shape))

if __name__ == "__main__":
    cfg = RoPEConfig(d_rope=16, max_seq_len=128, theta=10000.0)
    model = RotaryEmbedding(cfg)
    x = torch.randn(2, 4, 128, 16)
    output = model(x)
    print("Output shape:", output.shape)
    artifact = model.get_artifact()
    print("Artifact:", artifact)