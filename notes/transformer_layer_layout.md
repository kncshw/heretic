# Transformer Layer Layout (Llama 3.1 8B)

One full transformer layer, showing data flow through the residual stream:

```
residual stream (4096-dim)
        │
        ├─────────────────────────────┐
        │                            │  (residual connection)
        ▼                            │
   RMSNorm                           │
        │                            │
        ├──► q_proj → Q              │
        ├──► k_proj → K              │
        ├──► v_proj → V              │
        │                            │
        │   softmax(QK^T/√d) @ V     │
        │          │                 │
        ▼          ▼                 │
       o_proj ◄────┘                 │
        │                            │
        └──────────────► + ◄─────────┘  ← residual addition #1
                         │
                         │  (now updated residual stream)
                         │
        ┌────────────────┤
        │                │  (residual connection)
        ▼                │
   RMSNorm               │
        │                │
        ├──► gate_proj   │
        ├──► up_proj     │
        │       │        │
        │   silu(gate) * up
        │       │        │
        ▼       ▼        │
      down_proj          │
        │                │
        └──────► + ◄─────┘  ← residual addition #2
                 │
                 ▼
        residual stream (updated, passed to next layer)
```

---

## Execution Order Within One Layer

```
1. RMSNorm
2. q_proj / k_proj / v_proj → attention scores → o_proj    ← attention block
3. residual + (addition #1)
4. RMSNorm
5. gate_proj / up_proj → silu → down_proj                  ← MLP block
6. residual + (addition #2)
```

Attention block and MLP block are **sequential**, not parallel.

---

## Key Observations

**o_proj is the last gate of the attention block.**
q/k/v_proj operate entirely in internal subspaces. Only o_proj writes the
attention result back to the residual stream.

**down_proj is the last gate of the MLP block.**
gate_proj and up_proj are internal intermediate computations. Only down_proj
writes the MLP result back to the residual stream.

**There are exactly two write points per layer** — o_proj and down_proj.
Everything else (q/k/v/gate/up projections, attention scores, RMSNorm) is
internal computation that never touches the residual stream directly.

This is why abliteration targets exactly these two matrices and nothing else:
they are the only points where the refusal direction can be intercepted.

---

## Dimensions (Llama 3.1 8B)

| Matrix | Shape | Notes |
|---|---|---|
| `q_proj` | 4096 → 4096 | 32 heads × 128 head_dim |
| `k_proj` | 4096 → 1024 | 8 KV heads × 128 head_dim (GQA) |
| `v_proj` | 4096 → 1024 | 8 KV heads × 128 head_dim (GQA) |
| `o_proj` | 4096 → 4096 | writes to residual stream |
| `gate_proj` | 4096 → 14336 | expands to intermediate dim |
| `up_proj` | 4096 → 14336 | expands to intermediate dim |
| `down_proj` | 14336 → 4096 | compresses back, writes to residual stream |

32 layers × (o_proj + down_proj) = **64 total write matrices** in the model.
Abliteration modified 54 of them (30 o_proj + 24 down_proj).
