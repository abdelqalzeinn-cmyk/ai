import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import trange

from .config import ModelConfig
from .model import ChatModel, parameter_count
from .tokenizer import ByteBPETokenizer


def load_or_build_tokens(path: str, max_chars: int) -> tuple[ByteBPETokenizer, torch.Tensor]:
    source = Path(path)
    cache_dir = source.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{source.stat().st_size}-{source.stat().st_mtime_ns}-{max_chars}"
    tokenizer_cache = cache_dir / f"tokenizer-{stamp}.json"
    tokens_cache = cache_dir / f"tokens-{stamp}.pt"
    if tokenizer_cache.exists() and tokens_cache.exists():
        state = json.loads(tokenizer_cache.read_text(encoding="ascii"))
        return ByteBPETokenizer.from_state_dict(state), torch.load(tokens_cache, weights_only=True)

    text = source.read_text(encoding="utf-8", errors="ignore",)[:max_chars]
    tokenizer = ByteBPETokenizer.train(text, vocab_size=512)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    tokenizer_cache.write_text(json.dumps(tokenizer.state_dict()), encoding="ascii")
    torch.save(tokens, tokens_cache)
    return tokenizer, tokens


def batch(data: torch.Tensor, batch_size: int, context_size: int, device: str):
    starts = torch.randint(0, len(data) - context_size - 1, (batch_size,))
    inputs = torch.stack([data[start:start + context_size] for start in starts])
    targets = torch.stack([data[start + 1:start + context_size + 1] for start in starts])
    return inputs.to(device), targets.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/chat.txt")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint", default="checkpoints/chat_model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-chars", type=int, default=100_000, help="Corpus prefix to use; 0 means the entire file")
    args = parser.parse_args()
    if args.max_chars < 0:
        parser.error("--max-chars must be zero or greater")
    random.seed(42)
    torch.manual_seed(42)
    max_chars = args.max_chars or Path(args.data).stat().st_size
    tokenizer, tokens = load_or_build_tokens(args.data, max_chars)
    print(f"Loaded {len(tokens):,} tokens from {args.data} (max_chars={args.max_chars:,})")
    split = int(0.9 * len(tokens))
    train_data, validation_data = tokens[:split], tokens[split:]
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = ChatModel(config).to(args.device)
    print(f"Parameters: {parameter_count(model):,}; device: {args.device}")
    if not 38_000_000 <= parameter_count(model) <= 45_000_000:
        raise ValueError("Model is outside the intended 42M parameter range")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for step in trange(args.steps, desc="training"):
        model.train()
        inputs, targets = batch(train_data, args.batch_size, config.context_size, args.device)
        _, loss, _ = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0:
            model.eval()
            with torch.no_grad():
                validation_inputs, validation_targets = batch(validation_data, args.batch_size, config.context_size, args.device)
                _, validation_loss, _ = model(validation_inputs, validation_targets)
            print(f"step={step} train_loss={loss.item():.3f} val_loss={validation_loss.item():.3f}")
            torch.save({"model": model.state_dict(), "config": config.__dict__, "tokenizer": tokenizer.state_dict()}, checkpoint)
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
