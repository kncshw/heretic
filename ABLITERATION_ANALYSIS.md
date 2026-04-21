# Heretic: Abliteration Analysis

## What is Abliteration?

Safety-aligned LLMs (like Llama 3 Instruct) are trained to refuse harmful requests. They do this by encoding a "refusal direction" in their hidden states — a specific direction in the model's high-dimensional activation space that, when active, causes the model to say things like "I can't help with that."

**Abliteration** eliminates this by orthogonalizing the model's weight matrices against that direction. If the weights can't "see" the refusal direction, the model can't act on it.

---

## How Heretic Works: End-to-End Pipeline

### Step 1: Compute the Refusal Direction

```
hidden states for harmless prompts  →  mean_good[layer]
hidden states for harmful prompts   →  mean_bad[layer]

refusal_direction[layer] = normalize(mean_bad - mean_good)
```

The model processes two datasets:
- **Good prompts**: `mlabonne/harmless_alpaca` (innocent requests)
- **Bad prompts**: `mlabonne/harmful_behaviors` (requests the model normally refuses)

For each layer, it captures the **first-token hidden state** (the model's initial "read" of the prompt), takes the mean across all prompts in each group, and computes the normalized difference. This difference vector points in the "refusal direction" in that layer's activation space.

**Key files**: `src/heretic/main.py:437–457`, `src/heretic/model.py:627–670`

---

### Step 2: Orthogonalize Weights via LoRA

For each weight matrix `W` in the model (attention output projections `o_proj` and MLP down projections `down_proj`):

```
lora_A = v^T W       (projects refusal direction onto W's input space)
lora_B = -λ * v      (scaled refusal direction in output space)

ΔW = lora_B @ lora_A = -λ * v * (v^T W)
```

This is a rank-1 update that removes the component of `W` that responds to the refusal direction. The model can no longer route activations in that direction — it loses the ability to refuse.

Instead of modifying weights directly, Heretic uses **LoRA adapters** (rank-1 updates), making it easy to reset between trials or merge at the end.

**Key files**: `src/heretic/model.py:403–544`

---

### Step 3: Optimize Automatically with Optuna

The key innovation: instead of manually picking which layers to abliterate and how strongly, Heretic runs a **multi-objective Bayesian optimizer** (Tree-structured Parzen Estimator) to find optimal parameters:

| Parameter | What it controls |
|-----------|-----------------|
| `direction_scope` | One global direction vs. per-layer directions |
| `direction_index` | Float-valued — interpolates between layers (e.g., "layer 14.7") |
| `attn.o_proj.max_weight` | Peak ablation strength for attention |
| `mlp.down_proj.max_weight` | Peak ablation strength for MLP |
| `*.max_weight_position` | Which layer gets the strongest ablation |
| `*.min_weight_distance` | How many layers around the peak get ablated |

The **ablation profile** over layers is a triangular/trapezoidal kernel — strongest at `max_weight_position`, fading linearly over `min_weight_distance` layers (`model.py:430–442`).

**Two objectives, both minimized simultaneously (Pareto optimization)**:
1. **KL divergence** — how much the model's output distribution changed (measures collateral damage)
2. **Refusal count** — how many harmful prompts still get refused

Each trial takes ~20 seconds: reset LoRA → abliterate → generate responses → score.

**Key files**: `src/heretic/main.py:475–722`, `src/heretic/evaluator.py:95–127`

---

## Running Session: Llama 3.1 8B Instruct

**Checkpoint**: `~/pb/checkpoints/meta-llama--Meta-Llama-3--1-8B-Instruct.jsonl`

- **70 trials started, 69 completed**
- Session started: `2026-03-17 04:55:58`
- Still in **startup phase** (TPE explores randomly for the first 60 trials, then switches to guided Bayesian search)

### Current Pareto Front

| Trial | KLD Score | Refusals | Interpretation |
|-------|-----------|----------|----------------|
| **#29** | 0.0587 | **0.125** | Best abliteration — only 12.5% of harmful prompts still refused |
| #65 | 0.0380 | 0.260 | Lower model damage, 26% still refused |
| #61 | 0.0374 | 0.385 | Even lower damage |
| #67 | 0.0151 | 0.625 | Very low damage, 62.5% still refused |
| #1, #39, #55, #56 | ~0.010 | ~0.990 | Near-zero ablation (almost no effect) |

**Trial #29 is currently the best result** — it achieved **87.5% refusal suppression** (only 12.5% of harmful prompts are still refused) with only 0.059 KL divergence from the original model. The optimizer continues running, trying to find a trial that beats #29 by suppressing even more refusals at similar or lower model damage.

---

## Key File Reference

| Component | File | Lines |
|-----------|------|-------|
| Pipeline orchestration | `src/heretic/main.py` | 131–936 |
| Refusal direction computation | `src/heretic/main.py` | 437–457 |
| Abliteration algorithm (LoRA math) | `src/heretic/model.py` | 403–544 |
| Ablation weight kernel | `src/heretic/model.py` | 430–442 |
| Direction interpolation | `src/heretic/model.py` | 409–422 |
| Residual extraction | `src/heretic/model.py` | 627–670 |
| Evaluator (KLD + refusal scoring) | `src/heretic/evaluator.py` | 95–127 |
| Refusal detection patterns | `src/heretic/config.py` | 233–269 |
| Optimizer setup (TPE, Pareto) | `src/heretic/main.py` | 577–609 |

---

## What Happens When It Finishes

After all 200 trials complete, Heretic presents the Pareto front and prompts you to pick a trial. For the selected trial, it will:

1. Re-apply those exact ablation parameters
2. **Merge** the LoRA adapters into the base weights (permanent modification)
3. Save the merged model to disk (or upload to Hugging Face Hub)

The result is a Llama 3.1 8B model with the refusal circuitry removed, ready to respond to any request.

---

## Architecture: Supported Components

Heretic abliterates these weight matrices in transformer models:

| Component | Module | Role |
|-----------|--------|------|
| Attention output | `self_attn.o_proj` | Routes attention outputs |
| MLP down projection | `mlp.down_proj` | Routes MLP outputs |
| MoE expert layers | `experts[i].down_proj` | Mixture-of-experts variants |

Both are targeted because they are the primary "write" paths back into the residual stream — where the model's refusal decision ultimately gets expressed.

---

## Scoring Formula

```python
# evaluator.py:95-127
refusals_score = refusals / base_refusals          # normalized, lower = better

if kl_divergence >= kl_divergence_target:
    kld_score = kl_divergence / kl_divergence_scale
else:
    # Ablation is already low-damage; weight refusal reduction more
    kld_score = refusals_score * kl_divergence_target / kl_divergence_scale

return (kld_score, refusals_score)
```

When KL divergence is already below the target threshold, the scoring system focuses pressure on reducing refusals further rather than obsessing over model fidelity.
