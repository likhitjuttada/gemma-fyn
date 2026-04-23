import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

PRINCIPLES_DIR = Path(__file__).parent.parent / "data" / "principles"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "synthetic"
EXAMPLES_PER_PRINCIPLE = 3

MINI = "gpt-4o-mini"   # Type A — cost-efficient
FULL = "gpt-4o"        # Type B & C — higher quality reasoning

TYPE_A_PROMPT = """\
Principle from "{book}": {name}
Definition: {description}
Application: {application}

Generate a realistic personal finance scenario question that a real person might ask, \
where the best answer draws on this principle. Then write a full answer that reasons through \
the scenario using this principle. Do NOT quote the book or mention it by name.

Format your response exactly as:
<think>
[Step-by-step reasoning that names and applies the principle "{name}" explicitly]
</think>

[Final practical answer, 2-4 paragraphs]

QUESTION:
[your scenario question here]

ANSWER:
[the full think+answer block here]\
"""

TYPE_B_PROMPT = """\
Principle from "{book}": {name}
Definition: {description}
Mental model: {mental_model}

Create a financial dilemma someone is facing. Write a response in first-person advisor voice \
that thinks through the dilemma step by step, explicitly invoking the framework "{name}". \
The reasoning should feel like genuine deliberation, not a lecture.

Format exactly as:
DILEMMA:
[the situation, 2-3 sentences]

RESPONSE:
<think>
[step-by-step reasoning that names and uses "{name}" at least twice, \
explores the principle's implications, considers edge cases]
</think>

[Final advice, 2-3 paragraphs]\
"""

TYPE_C_PROMPT = """\
Principle A from "{book_a}": {name_a} — {description_a}
Principle B from "{book_b}": {name_b} — {description_b}

Write a financial scenario where BOTH principles seem relevant, but one fits better. \
Write a response that briefly acknowledges both principles, then picks the more applicable \
one and explains why using the frameworks of both.

Format exactly as:
SCENARIO:
[the situation]

RESPONSE:
<think>
[acknowledge both principles, reason about which fits better, explain why]
</think>

[Final recommendation, 2-3 paragraphs]\
"""


def call_openai(client: OpenAI, model: str, prompt: str) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(5)


def parse_type_a(raw: str, book: str, principle_name: str) -> dict | None:
    if "<think>" not in raw or "</think>" not in raw:
        return None
    q_start = raw.find("QUESTION:")
    a_start = raw.find("ANSWER:")
    if q_start == -1 or a_start == -1:
        return None
    question = raw[q_start + len("QUESTION:"):a_start].strip()
    answer = raw[a_start + len("ANSWER:"):].strip()
    if not question or not answer:
        return None
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {"book": book, "principle": principle_name, "type": "A"},
    }


def parse_type_b(raw: str, book: str, principle_name: str) -> dict | None:
    if "<think>" not in raw or "</think>" not in raw:
        return None
    d_start = raw.find("DILEMMA:")
    r_start = raw.find("RESPONSE:")
    if d_start == -1 or r_start == -1:
        return None
    dilemma = raw[d_start + len("DILEMMA:"):r_start].strip()
    response = raw[r_start + len("RESPONSE:"):].strip()
    if not dilemma or not response:
        return None
    return {
        "messages": [
            {"role": "user", "content": dilemma},
            {"role": "assistant", "content": response},
        ],
        "metadata": {"book": book, "principle": principle_name, "type": "B"},
    }


def parse_type_c(raw: str, book_a: str, book_b: str, name_a: str, name_b: str) -> dict | None:
    if "<think>" not in raw or "</think>" not in raw:
        return None
    s_start = raw.find("SCENARIO:")
    r_start = raw.find("RESPONSE:")
    if s_start == -1 or r_start == -1:
        return None
    scenario = raw[s_start + len("SCENARIO:"):r_start].strip()
    response = raw[r_start + len("RESPONSE:"):].strip()
    if not scenario or not response:
        return None
    return {
        "messages": [
            {"role": "user", "content": scenario},
            {"role": "assistant", "content": response},
        ],
        "metadata": {"books": [book_a, book_b], "principles": [name_a, name_b], "type": "C"},
    }


def load_all_principles() -> list[dict]:
    all_principles = []
    for json_path in sorted(PRINCIPLES_DIR.glob("*.json")):
        entries = json.loads(json_path.read_text(encoding="utf-8"))
        for chapter_entry in entries:
            book = chapter_entry.get("book", json_path.stem)
            for p in chapter_entry.get("principles", []):
                all_principles.append({**p, "book": book})
    return all_principles


def already_generated(out_path: Path) -> set[tuple]:
    done = set()
    if not out_path.exists():
        return done
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                meta = obj.get("metadata", {})
                t = meta.get("type", "")
                if t in ("A", "B"):
                    done.add((meta.get("book"), meta.get("principle"), t))
                elif t == "C":
                    books = tuple(sorted(meta.get("books", [])))
                    principles = tuple(sorted(meta.get("principles", [])))
                    done.add((books, principles, "C"))
            except json.JSONDecodeError:
                pass
    return done


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_principles = load_all_principles()
    if not all_principles:
        print(f"No principles found in {PRINCIPLES_DIR}. Run 02_extract_principles.py first.")
        return

    print(f"Loaded {len(all_principles)} principles from {PRINCIPLES_DIR}")

    for p in tqdm(all_principles, desc="Type A+B examples"):
        book = p["book"]
        name = p["name"]
        out_path = OUTPUT_DIR / (book.replace(" ", "_").lower() + ".jsonl")
        done = already_generated(out_path)

        if (book, name, "A") not in done:
            prompt = TYPE_A_PROMPT.format(
                book=book, name=name,
                description=p.get("description", ""),
                application=p.get("application", ""),
            )
            raw = call_openai(client, MINI, prompt)
            example = parse_type_a(raw, book, name)
            if example:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            time.sleep(0.4)

        if (book, name, "B") not in done:
            prompt = TYPE_B_PROMPT.format(
                book=book, name=name,
                description=p.get("description", ""),
                mental_model=p.get("mental_model", ""),
            )
            raw = call_openai(client, FULL, prompt)
            example = parse_type_b(raw, book, name)
            if example:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            time.sleep(0.4)

    out_path_c = OUTPUT_DIR / "type_c_contrasts.jsonl"
    done_c = already_generated(out_path_c)
    pairs_needed = min(len(all_principles), 300)
    random.shuffle(all_principles)

    for i in tqdm(range(0, min(pairs_needed * 2, len(all_principles) - 1), 2), desc="Type C examples"):
        pa = all_principles[i]
        pb = all_principles[i + 1]
        if pa["book"] == pb["book"]:
            continue
        key = (tuple(sorted([pa["book"], pb["book"]])), tuple(sorted([pa["name"], pb["name"]])), "C")
        if key in done_c:
            continue
        prompt = TYPE_C_PROMPT.format(
            book_a=pa["book"], name_a=pa["name"], description_a=pa.get("description", ""),
            book_b=pb["book"], name_b=pb["name"], description_b=pb.get("description", ""),
        )
        raw = call_openai(client, FULL, prompt)
        example = parse_type_c(raw, pa["book"], pb["book"], pa["name"], pb["name"])
        if example:
            with out_path_c.open("a", encoding="utf-8") as f:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        time.sleep(0.5)

    total = sum(
        sum(1 for line in p.open(encoding="utf-8") if line.strip())
        for p in OUTPUT_DIR.glob("*.jsonl")
    )
    print(f"\nDone. Total examples generated: {total}")


if __name__ == "__main__":
    main()
