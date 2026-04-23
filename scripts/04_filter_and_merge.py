import json
import random
import re
from pathlib import Path

SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic"
DATA_DIR = Path(__file__).parent.parent / "data"
MIN_ASSISTANT_CHARS = 200
QUOTE_WORD_LIMIT = 30
TRAIN_RATIO = 0.90
SEED = 42


def get_assistant_content(example: dict) -> str:
    messages = example.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def get_user_content(example: dict) -> str:
    messages = example.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def has_think_block(text: str) -> bool:
    return "<think>" in text and "</think>" in text


def has_long_verbatim_quote(text: str) -> bool:
    quoted = re.findall(r'"([^"]{100,})"', text)
    for q in quoted:
        if len(q.split()) > QUOTE_WORD_LIMIT:
            return True
    return False


def load_all_examples() -> list[dict]:
    examples = []
    for jsonl_path in sorted(SYNTHETIC_DIR.glob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return examples


def filter_examples(examples: list[dict]) -> list[dict]:
    seen_user_prefixes: set[str] = set()
    kept = []
    dropped = {"too_short": 0, "no_think": 0, "duplicate": 0, "verbatim_quote": 0}

    for ex in examples:
        assistant = get_assistant_content(ex)
        user = get_user_content(ex)
        prefix = user[:100].strip().lower()

        if len(assistant) < MIN_ASSISTANT_CHARS:
            dropped["too_short"] += 1
            continue
        if not has_think_block(assistant):
            dropped["no_think"] += 1
            continue
        if prefix in seen_user_prefixes:
            dropped["duplicate"] += 1
            continue
        if has_long_verbatim_quote(assistant):
            dropped["verbatim_quote"] += 1
            continue

        seen_user_prefixes.add(prefix)
        kept.append(ex)

    print(f"Dropped: {dropped}")
    print(f"Kept: {len(kept)} / {len(examples)}")
    return kept


def strip_metadata(example: dict) -> dict:
    return {"messages": example["messages"]}


def main():
    examples = load_all_examples()
    print(f"Loaded {len(examples)} raw examples")

    examples = filter_examples(examples)

    random.seed(SEED)
    random.shuffle(examples)

    split = int(len(examples) * TRAIN_RATIO)
    train = examples[:split]
    val = examples[split:]

    train_path = DATA_DIR / "combined_train.jsonl"
    val_path = DATA_DIR / "combined_val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(strip_metadata(ex), ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(strip_metadata(ex), ensure_ascii=False) + "\n")

    print(f"Train: {len(train)} examples -> {train_path.name}")
    print(f"Val:   {len(val)} examples -> {val_path.name}")


if __name__ == "__main__":
    main()
