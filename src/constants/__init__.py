# Vocabulary
VOCAB_SIZE = 30000 # BPE vocabulary size

# Tokenizer
SOS_TOKEN = "<SOS>"  # Start-of-sequence marker
EOS_TOKEN = "<EOS>" # End-of-sequence / document boundary
PAD_TOKEN = "<PAD>" # Padding token (fills shorter sequences in a batch)
SYSTEM = "|SYSTEM|" # Chat role: system prompt
USER = "|USER|" # Chat role: user turn
ASSISTANT = "|ASSISTANT|" # Chat role: assistant turn
SPECIAL_TOKENS = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SYSTEM, USER, ASSISTANT]
MIN_FREQ = 2 # Minimum token frequency to be kept in BPE vocab

# Dataset
BLOCK_SIZE = 128 # Sliding-window context length (tokens per sample)
VAL_SPLIT  = 0.1 # Fraction of encoded data held out for validation

# RMSNorm
D_MODEL = 128 # Hidden dimension shared across all components

# RoPE 
D_HEAD_ROPE = 16 # Per-head positional (RoPE) dimension; head-shared in MLA

# Expert
D_FF_ROUTED = 32 # Hidden dim of each tiny routed expert