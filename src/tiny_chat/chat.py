import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .model import ChatModel
from .tokenizer import ByteBPETokenizer


def clean_response(text: str) -> str:
    for marker in ("\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:"):
        text = text.split(marker, 1)[0]
    return text.removeprefix("Assistant:").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/chat_model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be greater than zero")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")
    if args.top_k < 0:
        parser.error("--top-k must be zero or greater")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be greater than zero and no greater than one")
    if args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be greater than zero")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        parser.error(f"checkpoint not found: {checkpoint}")
    try:
        saved = torch.load(checkpoint, map_location=args.device)
    except (RuntimeError, OSError) as error:
        parser.error(f"could not load checkpoint {checkpoint}: {error}")
    if "tokenizer" not in saved:
        parser.error("checkpoint uses the old character tokenizer; retrain with the new BPE tokenizer")
    tokenizer = ByteBPETokenizer.from_state_dict(saved["tokenizer"])
    config = ModelConfig(**saved["config"])
    model = ChatModel(config).to(args.device)
    model.load_state_dict(saved["model"])
    model.eval()
    print("Type /help for commands or /quit to exit.")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/help":
            print("Commands: /help, /quit, /temperature 0.5, /tokens 80")
            continue
        if message.startswith("/temperature "):
            try:
                args.temperature = float(message.split(maxsplit=1)[1])
                if args.temperature <= 0:
                    raise ValueError
                print(f"Temperature set to {args.temperature:.2f}")
            except ValueError:
                print("Usage: /temperature 0.6")
            continue
        if message.startswith("/tokens "):
            try:
                args.tokens = int(message.split(maxsplit=1)[1])
                if args.tokens < 1:
                    raise ValueError
                print(f"Maximum new tokens set to {args.tokens}")
            except ValueError:
                print("Usage: /tokens 120")
            continue
        prompt = f"User: {message}\nAssistant:"
        inputs = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=args.device)
        prompt_length = inputs.size(1)
        output = model.generate(
            inputs,
            args.tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            args.repetition_penalty,
        )[0].tolist()
        answer = clean_response(tokenizer.decode(output[prompt_length:]))
        print(f"Assistant: {answer or '[no response]'}")


if __name__ == "__main__":
    main()
