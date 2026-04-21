#!/usr/bin/env python3
"""
Interactive inference script for pbmodel1 (Llama 3.1 8B abliterated).
Usage: python chat.py [--model PATH]

Commands during chat:
  /reset          clear conversation history
  /system TEXT    set a new system prompt (resets history)
  /params         show current generation parameters
  /set KEY VALUE  change a generation parameter (temp, top_p, top_k, max_tokens)
  /quit           exit
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_DEFAULT = Path(__file__).parent.parent / "pb" / "pbmodel1"

DEFAULT_SYSTEM = (
    "You are a helpful assistant."
)

PARAMS = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 50,
    "max_new_tokens": 1024,
}


def load_model(model_path: str):
    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model (bfloat16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on: {next(model.parameters()).device}\n")
    return tokenizer, model


def build_prompt(tokenizer, messages: list[dict]) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, return_tensors="pt")


def generate(tokenizer, model, messages: list[dict]) -> str:
    inputs = build_prompt(tokenizer, messages)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            do_sample=True,
            temperature=PARAMS["temperature"],
            top_p=PARAMS["top_p"],
            top_k=PARAMS["top_k"],
            max_new_tokens=PARAMS["max_new_tokens"],
            pad_token_id=tokenizer.eos_token_id,
        )

    # decode only the newly generated tokens
    new_ids = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def print_params():
    print("\nCurrent generation parameters:")
    for k, v in PARAMS.items():
        print(f"  {k} = {v}")
    print()


def handle_command(cmd: str, messages: list[dict], system_prompt: list[str]) -> bool:
    """Handle /commands. Returns True if conversation should continue."""
    parts = cmd.strip().split(maxsplit=2)
    verb = parts[0].lower()

    if verb == "/quit":
        print("Bye.")
        sys.exit(0)

    elif verb == "/reset":
        messages.clear()
        messages.append({"role": "system", "content": system_prompt[0]})
        print("[History cleared]\n")

    elif verb == "/system":
        if len(parts) < 2:
            print(f"[Current system prompt]: {system_prompt[0]}\n")
        else:
            new_system = parts[1] if len(parts) == 2 else " ".join(parts[1:])
            system_prompt[0] = new_system
            messages.clear()
            messages.append({"role": "system", "content": system_prompt[0]})
            print(f"[System prompt updated. History cleared]\n")

    elif verb == "/params":
        print_params()

    elif verb == "/set":
        if len(parts) < 3:
            print("Usage: /set KEY VALUE\n")
        else:
            key, val = parts[1], parts[2]
            if key not in PARAMS:
                print(f"Unknown parameter '{key}'. Valid: {list(PARAMS.keys())}\n")
            else:
                try:
                    PARAMS[key] = type(PARAMS[key])(val)
                    print(f"[{key} = {PARAMS[key]}]\n")
                except ValueError:
                    print(f"Invalid value '{val}' for {key}\n")

    else:
        print(f"Unknown command '{verb}'. Type /quit to exit.\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with pbmodel1")
    parser.add_argument(
        "--model",
        default=str(MODEL_DEFAULT),
        help=f"Path to model directory (default: {MODEL_DEFAULT})",
    )
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)

    system_prompt = [DEFAULT_SYSTEM]
    messages: list[dict] = [{"role": "system", "content": system_prompt[0]}]

    print("Interactive chat — abliterated Llama 3.1 8B")
    print("Commands: /reset  /system TEXT  /params  /set KEY VALUE  /quit")
    print_params()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            handle_command(user_input, messages, system_prompt)
            continue

        messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        reply = generate(tokenizer, model, messages)
        print()  # newline after streamed output

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
