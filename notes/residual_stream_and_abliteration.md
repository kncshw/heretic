# Residual Stream and Abliteration

## The Residual Stream

The residual stream is the single vector (shape: 4096 for Llama 3.1 8B) that flows through
all 32 transformer layers for each token position. Every layer reads from it and writes back
to it via addition. It is never overwritten — only accumulated.

```
residual = token_embedding + positional_embedding   ← starting value

# Layer 0:
residual = residual + attn_output_0(residual)       ← attention adds its delta
residual = residual + mlp_output_0(residual)        ← MLP adds its delta

# Layer 1:
residual = residual + attn_output_1(residual)
residual = residual + mlp_output_1(residual)

...

# Layer 31:
residual = residual + attn_output_31(residual)
residual = residual + mlp_output_31(residual)
         │
         └──► unembed → logits → next token prediction
```

The residual stream **starts** as the token embedding but diverges immediately. By layer 5
it is already substantially different. By layer 32 it encodes not just the token itself but
its meaning in full context (intent, relationships, topic, etc.).

There are 64 additions total across 32 layers (32 attention + 32 MLP). Each one is a small
delta computed from the current state of the stream.

---

## Why This Design Exists (Residual Connections)

Without residuals, backpropagation multiplies gradients through every layer:

```
∂L/∂x₀ = ∂L/∂x₃₂ × W₃₂ × W₃₁ × ... × W₀
```

If each matrix shrinks vectors slightly, 32 multiplications → gradient ≈ 0 → early layers
learn nothing. This is the vanishing gradient problem.

With residuals:
```
without: x₁ = f(x₀)          →  ∂x₁/∂x₀ = f'(x₀)         (can be tiny)
with:    x₁ = x₀ + f(x₀)     →  ∂x₁/∂x₀ = 1 + f'(x₀)     (always ≥ 1)
```

The `1` is a direct gradient highway back to early layers. Gradients can always flow back
through the `+` without shrinking. This simple `+` is what makes training 32+ layer networks
possible. (Introduced in ResNet 2015, inherited by transformers.)

---

## What Writes to the Residual Stream

Each transformer layer has two write operations:

```
# Attention block:
attn_out = softmax(QK^T / √d) @ V      ← computed internally (never touches residual)
residual = residual + o_proj(attn_out)  ← o_proj writes to residual stream

# MLP block:
gate = silu(gate_proj(residual)) * up_proj(residual)   ← internal computation
residual = residual + down_proj(gate)                  ← down_proj writes to residual stream
```

Projections that do NOT write to the residual stream:
- `q_proj`, `k_proj`, `v_proj` → operate in internal query/key/value subspaces
- `gate_proj`, `up_proj` → intermediate MLP computation

Only `o_proj` and `down_proj` are the final linear transformations before the `+`.
They are the gates that decide what gets written into the shared residual stream.

---

## How Abliteration Works

The refusal direction `v` is a unit vector in residual stream space (4096 dims).
It points from "how harmless prompts are represented" toward "how harmful prompts are
represented" in the residual stream at a specific layer.

```
v = normalize(mean(harmful_residuals) - mean(harmless_residuals))
```

By the mid layers (~13-14 in Llama 3.1 8B), harmful vs harmless prompts have accumulated
enough difference in the residual stream that `v` reliably separates them.

### The Math

For each target weight matrix W (o_proj or down_proj), a rank-1 update is applied:

```
W_new = W - λ · v · (v^T W)
              ^^^^^^^^^^^^
              "the part of W that writes in direction v"
              subtract it → W becomes blind to direction v
```

This is implemented as a LoRA rank-1 adapter during optimization, then merged into W at export:
```
lora_A = v^T @ W        # shape (1, in_features)
lora_B = -λ · v         # shape (out_features, 1)
ΔW = lora_B @ lora_A    # rank-1 outer product
```

The effect: the weight matrix can no longer write the refusal direction into the residual
stream. The "I should refuse this" signal never accumulates. The model responds normally.

### What "Projecting Out" Means

Abliteration does not *reverse* the refusal direction (that would flip refusals into
something else). It *projects it out* — makes the model blind to that direction entirely.
The model cannot refuse, but it also cannot "anti-refuse". It simply stops distinguishing
harmful from harmless along that axis.

---

## Which Layers Were Modified (Llama 3.1 8B, pbmodel1)

Best trial result: 3/100 refusals, KL divergence 0.0257

| Component | Layers modified | Count |
|---|---|---|
| `self_attn.o_proj` | layers 2–31 | 30 / 32 |
| `mlp.down_proj` | layers 8–31 | 24 / 32 |
| **Total** | | **54 weight matrices** |

Layers 0–1 (`o_proj`) and layers 0–7 (`down_proj`) are untouched — early layers handle
basic token semantics, not intent or refusal behavior.

The ablation is not uniform: it follows a triangular kernel with peak strength at layers
~20 (attention) and ~23 (MLP), fading toward both ends.

---

## Numeric Reality (Actual Measured Differences)

### Layer 20 `self_attn.o_proj` (4096 × 4096)
- 85% of elements changed (14.3M / 16.8M)
- Max single weight change: ±0.121 (original weights range ±0.609)
- Diff matrix singular values: σ1=0.216, σ2=0.001 → **99.8% rank-1**

### Layer 23 `mlp.down_proj` (4096 × 14336)
- 77% of elements changed (45.3M / 58.7M)
- Max single weight change: ±0.070
- Diff matrix singular values: σ1=0.127, σ2=0.002 → **99.4% rank-1**

### Layer 0 `self_attn.o_proj`
- 0% changed — byte-for-byte identical to original

The same output dimensions (4055, 1658, 1059, 4062, 291) are the most changed in both
o_proj and down_proj — these are the hidden dimensions with the largest component along
the refusal direction v. Both write to the same residual stream, so the same features
are affected in both.
