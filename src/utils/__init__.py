from transformers import PreTrainedTokenizerFast
from typing import List

def encode(tokenizer: PreTrainedTokenizerFast, text: str, add_special_tokens: bool = False) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=add_special_tokens)

def decode(tokenizer: PreTrainedTokenizerFast, ids: List[int], skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

def encode_batch(tokenizer: PreTrainedTokenizerFast, texts: List[str], add_special_tokens: bool = False) -> List[List[int]]:
    return [encode(tokenizer, t, add_special_tokens) for t in texts]

def decode_batch(tokenizer: PreTrainedTokenizerFast, batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
    return [decode(tokenizer, ids, skip_special_tokens) for ids in batch]
