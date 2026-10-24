"""Inference script for Ghana-LLM.

Supports both Hugging Face Hub models and local PyTorch checkpoints.
Allows interactive streaming CLI chat and one-shot generation.
"""

import argparse
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

def main():
    parser = argparse.ArgumentParser(description="Ghana-LLM Inference & Interactive Chat")
    parser.add_argument(
        "--model_name_or_path",
        default="ghanaopenai/ghana-llm-base",
        type=str,
        help="Hugging Face model ID or local directory"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--dtype", default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16", type=str)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--prompt", default=None, type=str, help="One-shot prompt. If not set, interactive chat is started.")
    args = parser.parse_args()

    print(f"Loading tokenizer & model from '{args.model_name_or_path}' on {args.device}...")
    torch_dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(args.device)
    model.eval()

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    if args.prompt:
        # One-shot mode
        messages = [{"role": "user", "content": args.prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(args.device)
        print(f"\nUser: {args.prompt}")
        print("Assistant: ", end="", flush=True)
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                repetition_penalty=1.1,
                streamer=streamer,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        print()
        return

    # Interactive mode
    print("\n" + "=" * 50)
    print("Ghana-LLM Interactive Chat (English or Asante Twi)")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 50 + "\n")

    history = []
    while True:
        try:
            user_input = input("\nUser 💬: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Akwaaba! (Goodbye!)")
                break

            history.append({"role": "user", "content": user_input})
            formatted = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt").to(args.device)

            print("Ghana-LLM 🇬🇭: ", end="", flush=True)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    repetition_penalty=1.1,
                    streamer=streamer,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            generated_ids = out[0][inputs.input_ids.shape[1]:]
            resp = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            history.append({"role": "assistant", "content": resp})

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break

if __name__ == "__main__":
    main()
