import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()

HF_USERNAME = "likhitjuttada"
REPO_NAME = "finance-reasoning-sft-dataset"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

DATA_DIR = Path(__file__).parent.parent / "data"
README = Path(__file__).parent.parent / "README.md"

UPLOAD_FILES = [
    (DATA_DIR / "combined_train.jsonl", "combined_train.jsonl"),
    (DATA_DIR / "combined_val.jsonl", "combined_val.jsonl"),
    (README, "README.md"),
]


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not set in .env")

    api = HfApi(token=token)

    print(f"Creating dataset repo: {REPO_ID}")
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
        exist_ok=True,
        private=False,
    )

    for local_path, repo_filename in UPLOAD_FILES:
        if not local_path.exists():
            print(f"  SKIP (not found): {local_path.name}")
            continue
        print(f"  Uploading {local_path.name} -> {repo_filename}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_filename,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
        )

    print(f"\nDone. Dataset live at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
