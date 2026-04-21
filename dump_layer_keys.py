#!/usr/bin/env python3
"""
Dump the parameter names for layers 0 and 1 of a locally cached HF model
by reading model.safetensors.index.json — no weight loading required.

The parameter names reveal the exact attribute paths Heretic's
get_layer_modules() needs to target (e.g. mlp.experts.N.down_proj).

Usage:
  python dump_layer_keys.py                                      # default: Gemma 4 26B A4B
  python dump_layer_keys.py google/gemma-4-26B-A4B-it
  python dump_layer_keys.py Qwen/Qwen3-235B-A22B-Instruct-2507
"""

import glob
import json
import os
import sys


DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    cache_dir_name = "models--" + model.replace("/", "--")
    hf_cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

    pattern = os.path.join(
        hf_cache, "hub", cache_dir_name, "snapshots", "*", "model.safetensors.index.json"
    )
    matches = glob.glob(pattern)
    if not matches:
        print(f"No index file found at: {pattern}", file=sys.stderr)
        print("Is the model downloaded? Check: ls ~/.cache/huggingface/hub/", file=sys.stderr)
        sys.exit(1)

    index_path = matches[0]
    print(f"Index: {index_path}")
    print()

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    for layer_idx in (0, 1):
        tag = f".layers.{layer_idx}."
        keys = sorted(k for k in weight_map if tag in k)
        if not keys:
            continue
        print(f"=== layer {layer_idx} ===")
        for k in keys:
            print(f"  {k}")
        print()


if __name__ == "__main__":
    main()
