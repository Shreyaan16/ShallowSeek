from dataclasses import dataclass, field
from pathlib import Path
from src.constants import * 

@dataclass
class TokenizerConfig:
    data_path:  Path = field(default_factory=lambda: Path("data/demo.txt"))
    tokenizer_path: Path = field(default_factory=lambda: Path("models/tokenizer"))
    vocab_size: int  = VOCAB_SIZE

@dataclass
class DataEncoderConfig:
    data_path: Path = field(default_factory=lambda: Path("data/demo.txt"))
    tokenizer_path: Path = field(default_factory=lambda: Path("models/tokenizer"))
    output_path: Path = field(default_factory=lambda: Path("data/encoded.bin"))

@dataclass
class DatasetConfig:
    encoded_path: Path = field(default_factory=lambda: Path("data/encoded.bin"))
    block_size: int = BLOCK_SIZE
    split: str = "train"
    val_split: float = VAL_SPLIT

@dataclass
class RMSNormConfig:
    d_model: int = D_MODEL
    eps: float = 1e-8

@dataclass
class RoPEConfig:
    d_rope: int = D_HEAD_ROPE
    max_seq_len: int = BLOCK_SIZE
    theta: float = 10000.0

@dataclass
class ExpertConfig:
    d_model: int = D_MODEL
    d_ff: int = D_FF_ROUTED