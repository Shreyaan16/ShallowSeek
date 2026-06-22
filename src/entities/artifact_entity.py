from dataclasses import dataclass
from pathlib import Path

@dataclass
class TokenizerArtifact:
    tokenizer_dir: Path
    vocab_size: int
    sos_id: int
    eos_id: int
    pad_id: int
    system_id: int
    user_id: int
    assistant_id: int

@dataclass
class DataEncoderArtifact:
    output_path: Path
    num_tokens: int

@dataclass
class DatasetArtifact:
    num_samples: int
    split: str
    block_size: int

@dataclass
class RMSNormArtifact:
    d_model: int
    eps: float
    n_params: int

@dataclass
class RoPEArtifact:
    d_rope: int
    max_seq_len: int
    shape: tuple

@dataclass
class ExpertArtifact:
    d_model: int
    d_ff: int
    n_params: int

@dataclass
class MoEArtifact:
    n_shared_experts: int
    n_routed_experts: int
    top_k: int
    active_params: int
    total_params: int

@dataclass
class MLAArtifact:
    output_dim: int
    kv_cache_per_token: int
    n_params: int

@dataclass
class MTPArtifact:
    n_mtp: int
    d_model: int
    n_params: int

@dataclass
class TransformerBlockArtifact:
    d_model: int
    n_params: int

@dataclass
class ModelArtifact:
    vocab_size: int
    d_model: int
    n_layers: int
    total_params: int
    unique_params: int
