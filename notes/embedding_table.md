# LLM Embedding Table

## What Is It?

The embedding table (`embed_tokens.weight`) is a matrix of shape `[vocab_size × hidden_size]`.
For Llama 3.1 8B: `128256 × 4096` ≈ **524 million parameters**.

It is a lookup table: given a token ID (integer), return a 4096-dimensional float vector.
That vector is the token's starting representation before it enters the transformer layers.

## Role in the Model

The embedding table is the **only entry point** for token identity into the model.
Every token in the input sequence is looked up here first, producing a `[sequence_length × 4096]`
matrix that flows into layer 0 of the transformer.

The 4096-dim vector is **not a fixed representation** — it is a starting point. As it passes
through 32 transformer layers, the vector is updated by attention and MLP operations, accumulating
contextual information from other tokens. By layer 32, the vector encodes not just the token
itself but its meaning in context.

## What Do the 4096 Numbers Mean?

The individual numbers have no human-interpretable meaning. You cannot look at dimension #42
and say "this axis means animal-ness." What matters is the **geometric relationships** between
vectors in that 4096-dimensional space.

After training, tokens with similar meanings end up with similar vectors:
```
embed("king") - embed("man") + embed("woman") ≈ embed("queen")
```
This structure emerges purely from training — it is not designed in.

## Initialization: Random Noise

Before training, every embedding vector is filled with random numbers drawn from a normal
distribution:
```
N(mean=0, std=0.02)
```
The `initializer_range: 0.02` in `config.json` controls this.

At initialization:
- All tokens are random points near the origin in 4096-dimensional space
- "cat" and "dog" are no closer than "cat" and "democracy"
- The model outputs complete gibberish

The std=0.02 is a practical engineering choice:
- Too large → activations explode through 32 layers of matrix multiplications
- Too small → gradients vanish, early layers cannot be updated
- 0.02 keeps activations in a workable range at the start of training

## How It Is Trained

The embedding table is trained **end-to-end with the entire model** via backpropagation.
There is no separate embedding training step.

Training objective: predict the next token at every position.
```
Input:  ["The", "cat", "sat", "on",  "the"]
Target: ["cat", "sat", "on",  "the", "mat"]
```

Backpropagation computes a gradient for every parameter that contributed to the loss,
including the embedding rows that were looked up in that batch. Those rows are updated
slightly in the direction that reduces prediction error.

**Only rows for tokens that appeared in the batch are updated.** Rare tokens get updated
rarely and tend to have noisier embeddings.

After training on trillions of tokens, the random noise has been sculpted into a meaningful
geometric space where proximity reflects semantic and syntactic similarity.

## Why Embeddings Must Be Trainable

A natural question: can we just use fixed random vectors and let the attention/MLP layers
do all the work?

This has been tried (frozen embeddings). It works but is noticeably worse, because:

1. The embedding is the only place token identity enters the model. Random initialization
   may accidentally place "cat" close to "democracy" and far from "dog."

2. If embeddings are frozen, the downstream layers are permanently constrained by those
   accidental similarities. They must spend capacity compensating rather than reasoning.

3. When embeddings are trainable, the whole model **negotiates jointly**: the embedding
   table and all 32 layers adjust together to find the representation that makes everything
   work best.

The embedding table and `lm_head` (the output projection) work as a pair:
- `embed_tokens`: token ID → 4096-dim vector (input side)
- `lm_head`: 4096-dim vector → scores over 128256 tokens (output side)

They are essentially learning inverse mappings of each other. In some models these matrices
are shared (tied weights), but Llama 3.1 keeps them separate (`"tie_word_embeddings": false`).

## Size in Context

| Component | Parameters | % of Total |
|---|---|---|
| `embed_tokens.weight` | ~524M | ~6.5% |
| 32 transformer layers | ~7.0B | ~87% |
| `lm_head.weight` | ~524M | ~6.5% |
| **Total** | **~8.03B** | 100% |
