import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

INPUT_DIR = Path(__file__).parent.parent / "data" / "raw_text"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "principles"
MODEL = "gpt-4o-mini"
MAX_CHUNK_CHARS = 12_000
RETRY_DELAY = 5

SYSTEM_PROMPT = """\
You are an expert at extracting the core principles and mental models from books.
Given a book chapter, extract ALL meaningful principles as structured JSON.
Return ONLY valid JSON — no explanation, no markdown fences.\
"""

USER_TEMPLATE = """\
Book: {book_title}
Chapter: {chapter_label}

Chapter text:
{chapter_text}

Extract all principles from this chapter. Return JSON exactly matching this schema:
{{
  "book": "{book_title}",
  "chapter": "{chapter_label}",
  "principles": [
    {{
      "name": "short principle name",
      "description": "what this principle says in 1-2 sentences",
      "application": "how someone should apply this principle in real financial decisions",
      "mental_model": "the underlying cognitive/psychological framework"
    }}
  ]
}}\
"""


def _fixed_chunks(text: str, label_prefix: str) -> list[tuple[str, str]]:
    chunks = []
    for idx in range(0, len(text), MAX_CHUNK_CHARS):
        chunk = text[idx:idx + MAX_CHUNK_CHARS].strip()
        if chunk:
            chunks.append((f"{label_prefix} Part {idx // MAX_CHUNK_CHARS + 1}", chunk))
    return chunks


def split_into_chapters(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"(### CHAPTER \d+ ###)", text)
    chapters = []

    # Parts[0] is text before the first chapter marker.
    # If it's substantial (>1000 chars of actual prose), chunk it — this handles
    # books that don't use "Chapter N" headings (e.g. The Richest Man in Babylon).
    pre_chapter = parts[0].strip()
    if len(pre_chapter) > 1000:
        chapters.extend(_fixed_chunks(pre_chapter, "Section"))

    i = 1
    while i < len(parts):
        label = parts[i].strip("# ").strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        if body.strip():
            chapters.append((label, body[:MAX_CHUNK_CHARS]))
        i += 2

    if not chapters and text.strip():
        chapters = _fixed_chunks(text, "Section")

    return chapters


def extract_principles(client: OpenAI, book_title: str, chapter_label: str, chapter_text: str) -> dict:
    prompt = USER_TEMPLATE.format(
        book_title=book_title,
        chapter_label=chapter_label,
        chapter_text=chapter_text.strip(),
    )
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"    Retry {attempt + 1}/3 after error: {e}")
            time.sleep(RETRY_DELAY)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    txt_files = sorted(INPUT_DIR.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {INPUT_DIR}. Run 01_extract_text.py first.")
        return

    for txt_path in tqdm(txt_files, desc="Books"):
        out_path = OUTPUT_DIR / (txt_path.stem + ".json")
        book_title = txt_path.stem.replace("_", " ").replace("-", " ").title()

        existing: list[dict] = []
        done_chapters: set[str] = set()
        if out_path.exists():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            done_chapters = {entry["chapter"] for entry in existing}

        text = txt_path.read_text(encoding="utf-8")
        chapters = split_into_chapters(text)

        new_entries = list(existing)
        for chapter_label, chapter_text in tqdm(chapters, desc=book_title, leave=False):
            if chapter_label in done_chapters:
                continue
            result = extract_principles(client, book_title, chapter_label, chapter_text)
            new_entries.append(result)
            out_path.write_text(json.dumps(new_entries, indent=2, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.3)

        print(f"  {book_title}: {len(new_entries)} chapters processed -> {out_path.name}")


if __name__ == "__main__":
    main()
