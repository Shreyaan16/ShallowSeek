# Vocabulary
VOCAB_SIZE = 520 # BPE vocabulary size

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

# MoE
D_FF_SHARED = 256 # Hidden dim of shared experts
N_SHARED_EXPERTS = 2 # Number of shared experts (run on every token)
N_ROUTED_EXPERTS = 8 # Total routed experts in the pool
TOP_K = 2  # How many routed experts are selected per token

# MLA
N_HEADS = 4 # Number of attention heads
D_HEAD_NOPE = 32 # Per-head content (NoPE) dimension
D_C_KV = 32 # KV latent compression dim
D_C_Q = 32 # Q latent compression dim
DROPOUT = 0.1 # Dropout probability applied inside MLA attention

# MTP
N_MTP = 1 # Number of extra MTP prediction heads
MTP_LAMBDA = 0.3 # Loss weight: total = main_loss + λ * avg_mtp_loss

# Model
N_LAYERS = 4 # Number of stacked TransformerBlock layers

# Training
LR            = 3e-4  # AdamW learning rate
WEIGHT_DECAY  = 0.1   # AdamW weight decay — penalises large weights to reduce overfitting
BATCH_SIZE    = 16    # Samples per gradient step
MAX_EPOCHS    = 3     # Full passes over training data
GRAD_CLIP     = 1.0   # Max gradient norm
EVAL_INTERVAL = 100   # Evaluate on val set every N steps

# KV Cache
B = 1 # Batch size for KV cache (prefill is always single-batch)