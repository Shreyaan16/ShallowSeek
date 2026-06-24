from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR        = ROOT_DIR / "data"
MODELS_DIR      = ROOT_DIR / "models"
TOKENIZER_DIR   = MODELS_DIR / "tokenizer"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

DATA_PATH    = DATA_DIR / "demo.txt"
ENCODED_PATH = DATA_DIR / "encoded.bin"
BEST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "best.pt"