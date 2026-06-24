import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
TOKENIZER_DIR = os.path.join(MODELS_DIR, "tokenizer")

DATA_PATH = os.path.join(DATA_DIR, "demo.txt")
ENCODED_PATH = os.path.join(DATA_DIR, "encoded.bin")

