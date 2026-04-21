# Attention Block

## Overview

Self-attention is the mechanism that allows tokens to communicate with each other.
It answers the question: "given all previous tokens, what information is relevant to my current position?"

All tokens are processed in **parallel** during training and prefill.
During generation, one new token is processed at a time using the KV cache.

---

## Q, K, V Projections

For every token, three vectors are computed by multiplying the token's current hidden state
against three learned weight matrices:

```
Q = input @ q_proj.weight.T    # "what am I looking for?"
K = input @ k_proj.weight.T    # "what context do I offer?"
V = input @ v_proj.weight.T    # "what information do I carry?"
```

For Llama 3.1 8B:
- `q_proj`: 4096 → 4096 (32 heads × 128 head_dim)
- `k_proj`: 4096 → 1024 (8 KV heads × 128 head_dim)
- `v_proj`: 4096 → 1024 (8 KV heads × 128 head_dim)

K and V are separate because **relevance** and **content** are decoupled:
- K determines who gets attended to (matched against Q)
- V determines what information is actually extracted

---

## Attention Score Computation

### Step 1: Dot product Q against every K

```
score[i] = Q · K[i] / sqrt(head_dim)
```

High score = Q and K point in similar directions = this token is relevant.
Division by sqrt(head_dim) prevents scores from becoming too large before softmax.

### Step 2: Softmax → attention weights

```
weights = softmax([score[0], score[1], ..., score[N]])
```

Weights sum to 1.0 — they represent a probability distribution over past tokens.

### Step 3: Weighted sum of V → output

```
output = sum(weights[i] * V[i] for i in 0..N)
```

The output is a blend of all past tokens' V vectors, weighted by relevance.
This output vector now carries context from the whole sequence.

---

## Concrete Example: "The cat sat on"

Token "on" (position 4) attends over all previous tokens:

```
scores:   dot(Q_on, K_The)=0.3,  dot(Q_on, K_cat)=1.8,
          dot(Q_on, K_sat)=2.4,  dot(Q_on, K_on)=1.1

weights:  softmax → [0.04, 0.21, 0.58, 0.17]

output:   0.04*V_The + 0.21*V_cat + 0.58*V_sat + 0.17*V_on
```

"on" pays 58% attention to "sat" — it has learned that "on" most relevantly
follows a verb in this context.

---

## Multi-Head Attention

Each layer has **32 attention heads**, each operating in a 128-dim subspace (4096 / 32).
Every head independently computes its own Q, K, V and attention scores.

```
Head 1:  Q[token] · K[0..N]  → N dot products → 1 output vector
Head 2:  Q[token] · K[0..N]  → N dot products → 1 output vector
...
Head 32: Q[token] · K[0..N]  → N dot products → 1 output vector

32 output vectors concatenated → fed into o_proj (4096 → 4096)
```

Each head specializes in a different type of relationship (emerges from training, not programmed):
- Some heads track syntactic relationships (subject-verb)
- Some heads track coreference ("it" → "cat")
- Some heads focus on recent/positional context
- Some heads track semantic similarity

---

## Grouped Query Attention (GQA)

Llama 3.1 uses GQA: `num_attention_heads=32` but `num_key_value_heads=8`.

Every 4 Q heads share one K/V head:
```
Q heads 1-4   share   KV head 1
Q heads 5-8   share   KV head 2
...
Q heads 29-32 share   KV head 8
```

Benefit: KV cache is 4× smaller with minimal quality loss.

---

## KV Cache (Inference)

During generation, K and V for all previous tokens are cached and reused.
Only Q is computed fresh for each new token — Q is never cached.

```
Turn 1 (prefill):  process ["The", "cat", "sat"] in parallel
                   cache = 3 K/V pairs per layer

Turn 2 (generate): new token = "on"
                   compute Q, K, V for "on"
                   append K, V to cache → cache = 4 K/V pairs
                   Q_on attends over all 4 cached K/V pairs
                   produce next token

Turn 3 (generate): new token = "the"
                   cache = 5 K/V pairs
                   ...
```

**Q is discarded after use. Only K and V grow the cache.**

---

## Computation Cost Per New Token

For token #1001 at one layer with 32 heads:
```
32 heads × 1001 dot products = 32,032 dot products
```

Across all 32 layers:
```
32 layers × 32 heads × 1001 = ~1,025,024 dot products
```

This is just for attention scores — one token of output.

---

## KV Cache Memory Cost

```
cache size = num_layers × 2 × seq_len × num_kv_heads × head_dim × bytes

For Llama 3.1 8B (bfloat16):
= 32 × 2 × seq_len × 8 × 128 × 2
≈ seq_len × 128 KB
```

| Sequence Length | Cache Size |
|---|---|
| 1,000 tokens | ~128 MB |
| 10,000 tokens | ~1.3 GB |
| 100,000 tokens | ~13 GB |
| 131,072 tokens (max) | ~16 GB |

At max context the KV cache equals the model's own weight size (~16 GB).

---

## Causal Masking (Training Only)

During training, all tokens are processed in parallel but future tokens must be invisible.
A causal mask sets future positions to -inf before softmax:

```
         The   cat   sat
The   [  1.2  -inf  -inf ]   ← can only see itself
cat   [  0.9   2.1  -inf ]   ← can see The, cat
sat   [  0.3   1.8   2.4 ]   ← can see all previous
```

-inf → 0 after softmax, effectively zeroing out future attention.
During inference this is unnecessary — future tokens don't exist yet.

---

## Inference Phases

| Phase | Description | Parallelism |
|---|---|---|
| Prefill | Process the entire user prompt, build KV cache, produce first output token | Fully parallel |
| Generation | Produce one new token at a time, extending the KV cache | Sequential |

Long prompts feel fast to start (prefill is parallel).
Long outputs get progressively slower (generation is sequential, cache grows).
