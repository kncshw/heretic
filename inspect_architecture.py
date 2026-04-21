#!/usr/bin/env python3
"""
Dump a HuggingFace model's module tree without loading weights.

Uses torch.device("meta") so the model is instantiated with shape/dtype
metadata only — no VRAM or RAM allocation for the weights. Intended
for diagnosing which attribute paths Heretic's get_layer_modules()
should target for a given architecture.

Usage:
  python inspect_architecture.py                                   # default: Gemma 4 26B A4B
  python inspect_architecture.py google/gemma-4-26B-A4B-it
  python inspect_architecture.py Qwen/Qwen3-235B-A22B-Instruct-2507
"""

import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM


DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"

INTERESTING = ("mlp", "moe", "expert", "attn", "proj", "gate", "ffn", "down", "up")


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"Inspecting: {model_name}")
    print()

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print("architecture :", cfg.architectures)
    for k in (
        "num_hidden_layers",
        "num_local_experts",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ):
        v = getattr(cfg, k, None)
        if v is not None:
            print(f"{k:<30s} {v}")
    print()

    with torch.device("meta"):
        m = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    # Locate the transformer layer list (matches Heretic's get_layers logic).
    root = m.model if hasattr(m, "model") else m
    layers = None
    for path in ("layers", "language_model.layers"):
        obj = root
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.ModuleList):
                layers = obj
                break
        except AttributeError:
            continue
    if layers is None:
        print("ERROR: could not locate layer list under model.model")
        sys.exit(1)

    print(f"num_layers   : {len(layers)}")
    print()

    print("=== layer[0] filtered module tree ===")
    for name, mod in layers[0].named_modules():
        if name and any(k in name.lower() for k in INTERESTING):
            print(f"  {name:<60s} {type(mod).__name__}")
    print()

    # For MoE models, also dump layer[1] because some architectures alternate
    # dense/sparse layers (e.g. Qwen3.5 MoE hybrid).
    if len(layers) > 1:
        print("=== layer[1] filtered module tree (for hybrid architectures) ===")
        for name, mod in layers[1].named_modules():
            if name and any(k in name.lower() for k in INTERESTING):
                print(f"  {name:<60s} {type(mod).__name__}")


if __name__ == "__main__":
    main()
