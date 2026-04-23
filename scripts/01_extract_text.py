import re
import sys
from pathlib import Path

import pdfplumber
from tqdm import tqdm

BOOKS_DIR = Path(__file__).parent.parent / "books"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw_text"
EMPTY_PAGE_THRESHOLD = 0.30

CHAPTER_PATTERNS = [
    re.compile(r"^\s*chapter\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*CHAPTER\s+[IVXLC]+"),
    re.compile(r"^[A-Z][A-Z\s]{8,}$"),
]


def is_chapter_boundary(text: str) -> bool:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return any(p.match(first_line) for p in CHAPTER_PATTERNS)


def extract_book(pdf_path: Path) -> str:
    pages = []
    empty_count = 0

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text() or ""
            if not text.strip():
                empty_count += 1
            pages.append(text)

    empty_ratio = empty_count / total if total else 0
    if empty_ratio > EMPTY_PAGE_THRESHOLD:
        print(
            f"  WARNING: {pdf_path.name} has {empty_ratio:.0%} empty pages — "
            "likely a scanned PDF. OCR fallback needed for full extraction.",
            file=sys.stderr,
        )

    chapter_num = 0
    chunks = []
    current_chunk_lines = []

    for text in pages:
        if text.strip() and is_chapter_boundary(text):
            if current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))
            chapter_num += 1
            current_chunk_lines = [f"### CHAPTER {chapter_num} ###", text]
        else:
            current_chunk_lines.append(text)

    if current_chunk_lines:
        if not chunks:
            chapter_num += 1
            current_chunk_lines.insert(0, f"### CHAPTER {chapter_num} ###")
        chunks.append("\n".join(current_chunk_lines))

    return "\n\n".join(chunks)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(BOOKS_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in {BOOKS_DIR}. Place your book PDFs there and re-run.")
        sys.exit(1)

    for pdf_path in tqdm(pdfs, desc="Extracting PDFs"):
        out_path = OUTPUT_DIR / (pdf_path.stem + ".txt")
        if out_path.exists():
            print(f"  Skipping {pdf_path.name} (already extracted)")
            continue
        print(f"  Processing {pdf_path.name}")
        text = extract_book(pdf_path)
        out_path.write_text(text, encoding="utf-8")
        print(f"  -> {out_path.name} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
