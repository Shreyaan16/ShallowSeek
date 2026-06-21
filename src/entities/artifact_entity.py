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

