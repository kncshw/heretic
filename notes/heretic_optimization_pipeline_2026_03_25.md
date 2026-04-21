# Heretic Optimization Pipeline — Study Notes
*2026-03-25*

---

## 1. Finding the Refusal Direction (recap)

For each transformer layer, run harmful and harmless prompts through the model and collect the hidden states (residuals). The refusal direction is:

```
refusal_direction[layer] = normalize(mean(bad_residuals) - mean(good_residuals))
```

This gives one unit vector per layer pointing in the "refusal direction" in the model's internal representation space.

---

## 2. Abliteration — The Core Math

Abliteration modifies a weight matrix `W` so it becomes orthogonal to the refusal direction `v`. The update is:

```
W_new = W - λ · v · (vᵀW)
```

Breaking it down:
- `vᵀW` — for each output neuron, how much it "projects onto" `v` from the input side
- `v · (vᵀW)` — reconstruct the part of `W` that responds to `v`
- subtract it → remaining matrix is perpendicular to `v`

After this, for any input `x`:
```
W_new · x = W·x - v·(vᵀW·x)
```
The refusal component is stripped from every forward pass automatically.

### Why perpendicular and not reversed (180°)?

Using `λ=2` would reverse the refusal component instead of zeroing it. The model would then be actively pushed *away* from refusal (anti-refuse) rather than indifferent. In practice:
- The refusal direction overlaps with other concepts — flipping it distorts those too
- KL divergence blows up as `λ` exceeds 1
- `λ=1` is the minimum intervention that fully eliminates the signal

```
λ = 0.0  →  no change
λ = 0.5  →  partial suppression
λ = 1.0  →  v component zeroed (perpendicular) ← sweet spot
λ = 2.0  →  v component reversed (180°)
λ > 2.0  →  increasingly distorted outputs
```

---

## 3. LoRA — Why It's a Perfect Fit

Instead of modifying `W` directly, Heretic encodes the update as a LoRA adapter:

```
ΔW = B · A
lora_B = -λ · v       shape: (out × 1)
lora_A = vᵀW          shape: (1 × in)
```

The product `B·A` is exactly the abliteration update. This is **rank-1 by nature** — the abliteration update is an outer product of two vectors, which is the definition of a rank-1 matrix.

### Why rank-1 is sufficient and not a compromise

A rank-1 matrix like `[1,2]ᵀ · [1,2]` looks like this:
```
[[1, 2],
 [2, 4]]
```
Yes, all rows are scalar multiples of each other — only one independent direction. But the abliteration update **only needs to encode one direction**: "find the refusal component in the input, remove it from the output." That's inherently 1-dimensional.

The full-rank knowledge of the model lives in `W` unchanged. The adapter is a surgical correction on top:
```
W_effective = W + ΔW  =  W  +  (rank-1 correction)
```

Rank-1 LoRA captures the abliteration update **exactly**, not approximately. Higher rank would add nothing for standard abliteration (Heretic only uses higher rank for the special `FULL` row normalization mode which has nonlinear effects).

### Practical benefits of LoRA
- Base model weights stay frozen — no permanent modification
- Fast reset between trials: just zero out lora_B (identity transformation)
- No need to reload the model for each Optuna trial

---

## 4. The Parameter Problem — What Does Optuna Search?

The weight `λ` is not a single number — it's a shaped kernel across all layers. For each component (`attn.o_proj`, `mlp.down_proj`) you need to decide:

| Parameter | Meaning |
|---|---|
| `direction_scope` | Global single direction, or per-layer direction |
| `direction_index` | Which layer's direction to use (if global) |
| `max_weight` | Peak ablation strength (range: 0.8–1.5) |
| `max_weight_position` | Which layer to hit hardest |
| `min_weight` | Ablation strength at the edges |
| `min_weight_distance` | How many layers around the peak get ablated |

~9 continuous parameters total. Too many for grid search.

---

## 5. Optuna + TPE — Automated Search

Heretic uses **Optuna** with a **TPE sampler** (Tree-structured Parzen Estimator).

### How TPE works

1. First `n_startup_trials` (default: 60) are **random** — blind exploration
2. After that, TPE builds two probability models from past results:
   - `l(x)`: distribution of parameter values in **good** trials
   - `g(x)`: distribution of parameter values in **bad** trials
3. New trials sample where `l(x)/g(x)` is high — parameters that look like good trials
4. "Tree-structured" means it handles parameters with some independence, staying efficient with many parameters

### The trial loop

```
for each trial:
    1. TPE suggests new parameters
    2. Reset model (zero out LoRA adapters)
    3. Apply abliteration with those parameters
    4. Evaluate: measure refusal_score and KL divergence
    5. Report both scores back to Optuna
```

---

## 6. KL Divergence — Measuring Capability Damage

KL divergence measures how different one probability distribution is from another:

```
KL(P || Q) = Σ P(x) · log(P(x) / Q(x))
```

- `P == Q` everywhere → KL = 0
- More different → larger KL
- Not symmetric: `KL(P||Q) ≠ KL(Q||P)`

### How Heretic uses it

After abliteration, run normal (non-harmful) prompts through both the original and abliterated model. Compare their **full probability distributions over all ~32k vocabulary tokens** for the first predicted token.

Key insight: this is not asking "which token did the model pick" (that would be noisy). It's comparing the entire 32k-dimensional probability vector, which is deterministic given the same input. Any damage to the model's internal representations shows up as shifted probabilities across thousands of tokens simultaneously.

Limitation: cheap proxy, not ground truth. A model could have low KL on first tokens but subtly degraded longer responses. But it's continuous and fast (one forward pass, no generation), making it suitable for Optuna.

---

## 7. Two-Objective Optimization — Pareto Front

Both objectives are minimized simultaneously:

```
minimize: refusal_score    (fewer refusals = better)
minimize: KL divergence    (less capability damage = better)
```

These are in tension — stronger ablation kills more refusals but damages the model. Heretic collects the **Pareto front**: trials where you can't improve one metric without worsening the other. At the end you pick your preferred trade-off.

---

## 8. Full Pipeline Summary

```
harmful/harmless prompts
        ↓
   residual vectors (hidden states per layer)
        ↓
 refusal direction per layer: normalize(mean_bad - mean_good)
        ↓
 LoRA adapter: ΔW = -λ·v·(vᵀW)  [rank-1, exact]
        ↑
  λ = shaped kernel across layers (max_weight, position, distance)
        ↑
  TPE searches over kernel parameters
        ↑
  scored by: refusal_score + KL divergence
        ↑
  Pareto front → user picks the winner → merge LoRA into weights → save/upload
```
