# ShallowSeek

A from-scratch PyTorch implementation of the DeepSeek V3 pretraining architecture — built to understand every component from the ground up.

> **Status: Active development.** Pretraining architecture complete. IFT, RLHF, and KV-cache inference planned next.

**Dataset:** Plaintext Wikipedia (full English) — https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-english (~500 MB)

---

## Architecture Overview

```
Input tokens
    │
    ▼
Embedding (vocab_size × d_model)
    │
    ▼  ×N layers
┌─────────────────────────────┐
│  RMSNorm                    │
│  Multi-Head Latent Attention│  ← MLA (NoPE + RoPE)
│  + residual                 │
│  RMSNorm                    │
│  Mixture of Experts         │  ← Shared + Routed experts
│  + residual                 │
└─────────────────────────────┘
    │
    ▼
RMSNorm → LM Head (tied weights)
    │
    ├── main loss (next-token prediction)
    │
    └── MTP heads × n_mtp    ← auxiliary multi-token prediction loss
```

---

## Components

### 1. RMSNorm — `src/components/norm.py`

Root Mean Square Layer Normalization. Used as pre-norm before every attention and FFN sublayer, and before the final LM head.

```
x_norm = x / RMS(x) * γ
```

### 2. Rotary Position Embeddings (RoPE) — `src/components/rope.py`

Position encoding applied to the RoPE-dimension split of Q and K. Uses complex-number rotation with pre-computed `freqs_cis` buffers for efficiency.

- Applied to the `d_head_rope` portion of Q per head
- Applied to a **head-shared** K projection (`W_KR`) — a key MLA optimization that avoids duplicating positional state across heads in the KV cache

### 3. Multi-Head Latent Attention (MLA) — `src/components/attention.py`

DeepSeek V3's core attention innovation. Compresses KV into a low-rank latent vector to reduce KV cache size at inference, while keeping expressivity via an up-projection at compute time.

**Projection structure:**

| Projection | Shape | Purpose |
|---|---|---|
| `W_DKV` | d_model → d_c_kv | Compress KV into latent `c_kv` |
| `W_UK` | d_c_kv → n_heads × d_head_nope | Up-project K content (NoPE) |
| `W_UV` | d_c_kv → n_heads × d_head_nope | Up-project V |
| `W_DQ` | d_model → d_c_q | Compress Q into latent `c_q` |
| `W_UQ` | d_c_q → n_heads × d_head_nope | Up-project Q content (NoPE) |
| `W_QR` | d_c_q → n_heads × d_head_rope | Q RoPE portion (per-head) |
| `W_KR` | d_model → d_head_rope | K RoPE portion (head-shared) |
| `W_O` | n_heads × d_head_nope → d_model | Output projection |

Each head's full Q and K are `[NoPE | RoPE]` concatenated. Attention is computed on this combined representation; only the NoPE values are used for output projection.

**KV cache at inference:** only `c_kv` (`d_c_kv` dims) and `k_rope` (`d_head_rope` dims) need caching per token — far smaller than standard MHA which caches `n_heads × d_head` per token for both K and V.

### 4. Mixture of Experts (MoE) — `src/components/moe.py`

Each `TransformerBlock` replaces the standard FFN with a MoE layer containing two expert types:

- **Shared experts** — always active, run on every token in parallel and summed
- **Routed experts** — gated; only `top_k` out of `n_routed_experts` are selected per token by a softmax router

Expert FFN uses SwiGLU activation:
```
Expert(x) = W2(SiLU(W1(x)) ⊙ W3(x))
```

Top-k weights are renormalized to sum to 1 before weighting expert outputs.

**Load balancing** uses dynamic router bias (matching DeepSeek V3, not an auxiliary loss). Per-expert biases are updated every forward pass to nudge actual token counts toward a uniform target:
```
bias_i += bias_update_rate * (target_count - actual_count_i)
```

Bias is used only for routing decisions; scores without bias are used for output weighting.

### 5. TransformerBlock — `src/components/block.py`

Standard pre-norm residual block wiring MLA and MoE together:

```
x = x + MLA(RMSNorm(x))
x = x + MoE(RMSNorm(x))
```

### 6. Multi-Token Prediction (MTP) — `src/components/mtp.py`

Auxiliary prediction heads that learn to predict tokens multiple steps ahead, improving gradient signal during pretraining. Each MTP module:

1. Normalizes the previous hidden state `h_prev` and the next-token embedding `e_k` separately
2. Concatenates and projects them back to `d_model` via a linear layer
3. Runs a full `TransformerBlock`
4. Applies the **shared** LM head (tied weights) to produce logits for the `k+1`-ahead token

MTP modules are chained: the output hidden state of module `k` feeds into module `k+1` as `h_prev`.

The total training loss is:
```
L = L_main + λ × mean(L_mtp_1, ..., L_mtp_n)
```

### 7. ShallowSeek (full model) — `src/components/model.py`

Assembles all components into the full autoregressive LM. Key design choices:

- **Tied weights:** `lm_head.weight = embed.weight`, reducing unique parameter count
- **MTP-aware loss:** the effective sequence is trimmed by `n_mtp + 1` tokens so all MTP heads have valid targets
- **`get_artifact()`** reports both total and unique parameter counts (the difference is the tied embedding)

---

## Data Pipeline

### Tokenizer — `src/components/tokenizer.py`

Byte-level BPE tokenizer trained from scratch using HuggingFace `tokenizers`. Special tokens for chat formatting are baked in from the start so the same tokenizer works for both pretraining and instruction tuning.

| Token | Role |
|---|---|
| `<SOS>` | Start of sequence |
| `<EOS>` | End of sequence / document boundary |
| `<PAD>` | Padding |
| `\|SYSTEM\|` | System prompt role marker |
| `\|USER\|` | User turn marker |
| `\|ASSISTANT\|` | Assistant turn marker |

### Data Encoder — `src/components/encoder.py`

Tokenizes a raw text file line-by-line, appending `<EOS>` after each line, and saves the result as a flat binary `uint16` array (`.bin` file) for efficient `numpy.memmap` access during training.

### Dataset — `src/components/dataset.py`

Sliding-window `torch.utils.data.Dataset` over the encoded binary file. Each sample is a context window of `block_size` tokens; the target is the same window shifted by one position. Train/val split is a contiguous prefix/suffix cut.

---

## Configuration

All hyperparameters live in [`src/constants/__init__.py`](src/constants/__init__.py) and are surfaced as typed dataclasses in [`src/entities/config_entity.py`](src/entities/config_entity.py).

| Constant | Default | Description |
|---|---|---|
| `VOCAB_SIZE` | 30,000 | BPE vocabulary size |
| `BLOCK_SIZE` | 128 | Context window (tokens) |
| `D_MODEL` | 128 | Hidden dimension |
| `N_LAYERS` | 4 | Transformer depth |
| `N_HEADS` | 4 | Attention heads |
| `D_HEAD_NOPE` | 32 | Per-head content (NoPE) dim |
| `D_HEAD_ROPE` | 16 | Per-head positional (RoPE) dim |
| `D_C_KV` | 32 | KV latent compression dim |
| `D_C_Q` | 32 | Q latent compression dim |
| `N_SHARED_EXPERTS` | 2 | Shared experts (always active) |
| `N_ROUTED_EXPERTS` | 8 | Total routed experts |
| `TOP_K` | 2 | Routed experts selected per token |
| `D_FF_SHARED` | 256 | Shared expert hidden dim |
| `D_FF_ROUTED` | 32 | Routed expert hidden dim |
| `N_MTP` | 1 | Number of MTP prediction heads |
| `MTP_LAMBDA` | 0.3 | MTP loss weight λ |

---

## Project Structure

```
ShallowSeek/
├── src/
│   ├── constants/              # Global hyperparameter constants
│   ├── entities/
│   │   ├── config_entity.py    # Typed config dataclasses
│   │   └── artifact_entity.py  # Typed output/stats dataclasses
│   ├── utils/                  # encode / decode helpers
│   ├── components/
│   │   ├── norm.py             # RMSNorm
│   │   ├── rope.py             # Rotary Position Embeddings
│   │   ├── attention.py        # Multi-Head Latent Attention (MLA)
│   │   ├── moe.py              # Expert + ShallowSeekMoE
│   │   ├── block.py            # TransformerBlock (MLA + MoE)
│   │   ├── mtp.py              # Multi-Token Prediction module
│   │   ├── model.py            # ShallowSeek (full model)
│   │   ├── tokenizer.py        # BPE tokenizer
│   │   ├── encoder.py          # Text → binary token file
│   │   └── dataset.py          # Sliding-window Dataset
│   └── pipeline/               # Training pipeline (WIP)
└── data/
    ├── demo.txt                # Raw text corpus
    └── encoded.bin             # Tokenized binary output
```

---

## Roadmap

### Done — Pretraining Architecture
- [x] RMSNorm
- [x] Rotary Position Embeddings (RoPE) with complex-number formulation
- [x] Multi-Head Latent Attention (MLA) with low-rank KV compression and head-shared RoPE K
- [x] Mixture of Experts with shared + routed experts and dynamic bias load balancing
- [x] TransformerBlock (pre-norm, residual)
- [x] Multi-Token Prediction (MTP) auxiliary heads chained with shared weights
- [x] Full ShallowSeek model with tied embeddings
- [x] BPE tokenizer with chat special tokens
- [x] Binary data encoder + sliding-window Dataset
- [x] **KV-cache inference** — exploit MLA's compressed cache (`c_kv` + `k_rope`) for efficient autoregressive generation

### In Progress
- [ ] Training pipeline (optimizer, LR scheduler, gradient clipping, checkpointing, logging)

### Planned
- [ ] **Instruction Fine-Tuning (IFT)** — supervised fine-tuning on chat-formatted data using the `|SYSTEM|`/`|USER|`/`|ASSISTANT|` tokens
- [ ] **RLHF** — reward modeling and PPO/DPO alignment

---
