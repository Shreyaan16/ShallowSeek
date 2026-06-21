from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from src.constants import *
import os
from src.entities.config_entity import TokenizerConfig
from src.entities.artifact_entity import TokenizerArtifact
from src.utils import encode, decode

class ShallowSeekTokenizer:
    def __init__(self, config: TokenizerConfig):
        self.data_path: Path = config.data_path
        self.tokenizer_path: Path = config.tokenizer_path
        self.vocab_size: int = config.vocab_size

    def train(self) -> None:
        if self.tokenizer_path.exists():
            print(f"Tokenizer already exists at {self.tokenizer_path}, skipping training.")
            return
        
        if not self.data_path.exists() or self.data_path.stat().st_size == 0:
            raise FileNotFoundError(f"Training data not found or empty: {self.data_path}")

        tokenizer = Tokenizer(BPE(unk_token=None))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(vocab_size=self.vocab_size, min_frequency=MIN_FREQ, special_tokens=SPECIAL_TOKENS, show_progress=True)
        print(f"Training BPE tokenizer on {self.data_path}  (vocab_size={self.vocab_size})")
        tokenizer.train([str(self.data_path)], trainer)
        print(f"Trained vocab size: {tokenizer.get_vocab_size()}")

        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token=SOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,
                                        additional_special_tokens=[SYSTEM, USER, ASSISTANT])
        
        os.makedirs(self.tokenizer_path.parent, exist_ok=True)
        fast.save_pretrained(str(self.tokenizer_path))
        print(f"Saved to {self.tokenizer_path}")

    def _load(self, path: Path) -> TokenizerArtifact:
        fast = PreTrainedTokenizerFast.from_pretrained(str(path))
        fast.pad_token = PAD_TOKEN
        self._tok = fast
        self._bind_ids(fast)
        return TokenizerArtifact(tokenizer_dir=path, vocab_size=self.vocab_size, sos_id=self.sos_id, eos_id=self.eos_id, pad_id=self.pad_id,
                                system_id=self.system_id, user_id=self.user_id, assistant_id=self.assistant_id)

    def _bind_ids(self, fast: PreTrainedTokenizerFast) -> None:
        vocab = fast.get_vocab()
        self.sos_id = vocab[SOS_TOKEN]
        self.eos_id = vocab[EOS_TOKEN]
        self.pad_id = vocab[PAD_TOKEN]
        self.system_id = vocab[SYSTEM]
        self.user_id = vocab[USER]
        self.assistant_id = vocab[ASSISTANT]

    def load(self, path: Path) -> "ShallowSeekTokenizer":
        fast = PreTrainedTokenizerFast.from_pretrained(str(path))
        fast.pad_token = PAD_TOKEN
        self._tok = fast
        self._bind_ids(fast)
        return self


if __name__ == "__main__":
    config = TokenizerConfig()
    tok = ShallowSeekTokenizer(config)
    tok.train()
    tok.load(config.tokenizer_path)

    print("\n=== Special token IDs ===")
    print(f"  {SOS_TOKEN!r:15} -> {tok.sos_id}")
    print(f"  {EOS_TOKEN!r:15} -> {tok.eos_id}")
    print(f"  {PAD_TOKEN!r:15} -> {tok.pad_id}")
    print(f"  {SYSTEM!r:15} -> {tok.system_id}")
    print(f"  {USER!r:15} -> {tok.user_id}")
    print(f"  {ASSISTANT!r:15} -> {tok.assistant_id}")

    samples = ["Hello, world!", f"{SYSTEM} You are a helpful assistant. {USER} What is 2+2? {ASSISTANT} 4.",]
    print("\n=== Test ===")
    for text in samples:
        ids = encode(tok._tok, text)
        recovered = decode(tok._tok, ids)
        print(f"  input   : {text}")
        print(f"  ids     : {ids}")
        print(f"  decoded : {recovered}")