import numpy as np
from pathlib import Path
from src.entities.config_entity import DataEncoderConfig, TokenizerConfig
from src.entities.artifact_entity import DataEncoderArtifact
from src.components.tokenizer import ShallowSeekTokenizer
from src.utils import decode

class DataEncoder:
    def __init__(self, config: DataEncoderConfig):
        self.data_path: Path = config.data_path
        self.output_path: Path = config.output_path
        tok_config = TokenizerConfig(tokenizer_path=config.tokenizer_path)
        self._tok = ShallowSeekTokenizer(tok_config).load(config.tokenizer_path)

    def encode(self) -> DataEncoderArtifact:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        eos_id: int = self._tok.eos_id
        all_ids: list[int] = []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                ids = self._tok._tok.encode(line, add_special_tokens=False)
                all_ids.extend(ids)
                all_ids.append(eos_id)

        arr = np.array(all_ids, dtype=np.uint16)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        arr.tofile(str(self.output_path))

        print(f"Encoded {len(all_ids):,} tokens -> {self.output_path}")
        return DataEncoderArtifact(output_path=self.output_path, num_tokens=len(all_ids))


if __name__ == "__main__":
    
    config = DataEncoderConfig()
    encoder = DataEncoder(config)
    artifact = encoder.encode()

    print(f"\n=== Artifact ===")
    print(f"  output_path : {artifact.output_path}")
    print(f"  num_tokens  : {artifact.num_tokens:,}")

    #Verify: load back and decode first 50 tokens
    arr = np.fromfile(str(artifact.output_path), dtype=np.uint16)
    sample_ids = arr[:50].tolist()
    sample_text = decode(encoder._tok._tok, sample_ids, skip_special_tokens=False)
    print(f"\n=== First 50 tokens decoded ===")
    print(f"  ids  : {sample_ids}")
    print(f"  text : {sample_text}")
