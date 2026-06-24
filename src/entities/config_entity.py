from dataclasses import dataclass, field
from src.constants import * 
from pathlib import Path
from src.configurations.paths_config import DATA_PATH, ENCODED_PATH, TOKENIZER_DIR, CHECKPOINTS_DIR
import torch

@dataclass
class TokenizerConfig:
    data_path:  Path = field(default_factory=lambda: DATA_PATH)
    tokenizer_path: Path = field(default_factory=lambda: TOKENIZER_DIR)
    vocab_size: int  = VOCAB_SIZE

@dataclass
class DataEncoderConfig:
    data_path: Path = field(default_factory=lambda: DATA_PATH)
    tokenizer_path: Path = field(default_factory=lambda: TOKENIZER_DIR)
    output_path: Path = field(default_factory=lambda: ENCODED_PATH)

@dataclass
class DatasetConfig:
    encoded_path: Path = field(default_factory=lambda: ENCODED_PATH)
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

@dataclass
class MoEConfig:
    d_model: int = D_MODEL
    d_ff_shared: int = D_FF_SHARED
    d_ff_routed: int = D_FF_ROUTED
    n_shared_experts: int = N_SHARED_EXPERTS
    n_routed_experts: int = N_ROUTED_EXPERTS
    top_k: int = TOP_K
    bias_update_rate: float = 0.01

@dataclass
class MLAConfig:
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    d_head_nope: int = D_HEAD_NOPE
    d_head_rope: int = D_HEAD_ROPE
    d_c_kv: int = D_C_KV
    d_c_q: int = D_C_Q
    dropout: float = DROPOUT

@dataclass
class MTPConfig:
    d_model: int = D_MODEL
    n_mtp: int = N_MTP
    mtp_lambda: float = MTP_LAMBDA

@dataclass
class TransformerBlockConfig:
    d_model: int = D_MODEL
    mla: MLAConfig = field(default_factory=MLAConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)

@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    n_layers: int = N_LAYERS
    norm: RMSNormConfig = field(default_factory=RMSNormConfig)
    rope: RoPEConfig = field(default_factory=RoPEConfig)
    block: TransformerBlockConfig = field(default_factory=TransformerBlockConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)

@dataclass
class InferenceConfig:
    tokenizer_path: Path = field(default_factory=lambda: TOKENIZER_DIR)
    checkpoint_path: Path = field(default_factory=lambda: Path("models/checkpoint.pt"))
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    repetition_penalty: float = 1.3  # >1.0 penalises already-generated tokens
    device: str = "cpu"

@dataclass
class TrainingConfig:
    checkpoint_dir: Path = field(default_factory=lambda: Path(CHECKPOINTS_DIR))
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    encoder: DataEncoderConfig = field(default_factory=DataEncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train_dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(split="train"))
    val_dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(split="val"))
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS
    grad_clip: float = GRAD_CLIP
    eval_interval: int = EVAL_INTERVAL
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class InferencePipelineConfig:
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    prompts: list = field(default_factory=list)
    use_chat_format: bool = False  # False = pretrain completion, True = SFT chat mode