# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Heretic is a fully automatic censorship removal tool for transformer-based LLMs. It combines directional ablation ("abliteration") with TPE-based hyperparameter optimization (Optuna) to co-minimize refusal rate and KL divergence from the original model.

## Development Commands

```bash
# Install in editable mode with dev dependencies
uv sync --dev

# Install with research extras
uv sync --extra research

# Lint
uv run ruff check src/
uv run ruff format src/

# Type check
uv run ty check src/

# Run the tool (entry point)
uv run heretic <model_name>

# Evaluate a model
uv run heretic --model <base_model> --evaluate-model <abliterated_model>
```

There is no test suite.

## Architecture

The package lives in `src/heretic/` with 7 modules:

**Execution flow:**
1. `main.py` — CLI entry point (via `heretic.main:main`). Parses args, loads settings, benchmarks batch size, runs the Optuna optimization loop, then offers interactive save/upload/chat actions.
2. `config.py` — Pydantic `Settings` class loaded from TOML + CLI overrides. Defines all tuneable parameters including dataset specs, ablation weights, quantization, and KL divergence targets.
3. `model.py` — Loads the base model (with dtype fallback: auto → float16 → bfloat16 → float32), applies LoRA via PEFT, computes residual vectors, runs abliteration (orthogonalization of attention out-projection and MLP down-projection matrices), and handles inference.
4. `analyzer.py` — Computes refusal directions per layer (difference-of-means between harmful/harmless residuals). Optionally runs PaCMAP projections and generates visualizations (requires `research` extra).
5. `evaluator.py` — Scores each Optuna trial: counts refusals in model responses and computes KL divergence of first-token distributions vs. original model.
6. `utils.py` — Dataset loading (HuggingFace Hub or local), prompt formatting, memory reporting, interactive TUI helpers (questionary).
7. `__init__.py` — Empty.

**Key design points:**
- Abliteration is applied via LoRA adapters, not in-place weight modification, keeping the base model unchanged until export.
- Optuna uses a multivariate TPE sampler. Default: 200 trials, 60 startup (random) trials. Pareto-optimal trials on refusal vs. KL divergence are presented for interactive selection.
- The refusal direction is a float-valued index allowing interpolation between per-layer directions.
- Separate ablation weight parameters exist for attention vs. MLP components, each with `max_weight`, `position`, `min_weight`, and `distance` knobs that shape a kernel across layers.
- Optimization checkpoints are stored as Optuna SQLite journals in `./checkpoints/`.

## Configuration

- `config.default.toml` — Primary defaults (well-commented, read this first).
- `config.noslop.toml` — Alternative preset.
- CLI flags override TOML. Run `heretic --help` for full option list.
- Key parameters: `n_trials`, `n_startup_trials`, `kl_divergence_scale`, `kl_divergence_target`, `quantization` (`none` or `bnb_4bit`), `batch_size` (0 = auto-detect).
