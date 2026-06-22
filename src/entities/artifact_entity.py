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