import argparse
import json
from pathlib import Path

from datasets import load_dataset


def format_dialogue(dialogue: list[dict[str, str]]) -> str:
    turns = []
    for message in dialogue:
        speaker = "Assistant" if message["role"] == "assistant" else "User"
        clean = " ".join(message["content"].strip().split())
        if clean:
            turns.append(f"{speaker}: {clean}")
    return "\n".join(turns) + "\n\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/chat.txt")
    parser.add_argument("--hub-repo", help="Optional dataset repo, for example your-name/tiny-chat-data")
    args = parser.parse_args()

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open("w", encoding="ascii", errors="ignore") as stream:
        for item in dataset:
            text = format_dialogue(item["messages"])
            if text.strip():
                stream.write(text)
                rows.append({"text": text})
    metadata = output.with_suffix(".json")
    metadata.write_text(json.dumps({"source": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "examples": len(rows)}), encoding="ascii")
    if args.hub_repo:
        from datasets import Dataset
        Dataset.from_list(rows).push_to_hub(args.hub_repo)
    print(f"Wrote {len(rows)} dialogues to {output}")


if __name__ == "__main__":
    main()
