from pathlib import Path

# Vocabulary 
VOCAB_SIZE = 30000

# Tokenizer 
SOS_TOKEN   = "<SOS>"
EOS_TOKEN   = "<EOS>"
PAD_TOKEN   = "<PAD>"
SYSTEM      = "|SYSTEM|"
USER        = "|USER|"
ASSISTANT   = "|ASSISTANT|"
SPECIAL_TOKENS = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SYSTEM, USER, ASSISTANT]
MIN_FREQ = 2
