from dataclasses import dataclass, field
from pathlib import Path
from data import VOCAB_SIZE

@dataclass
class TokenizerConfig:
    data_path:  Path = field(default_factory=lambda: Path("data/demo.txt"))
    tokenizer_path: Path = field(default_factory=lambda: Path("models/tokenizer"))
    vocab_size: int  = VOCAB_SIZE
