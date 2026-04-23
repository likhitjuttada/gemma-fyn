import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()

HF_USERNAME = "likhitjuttada"
DATASET_REPO = f"{HF_USERNAME}/finance-reasoning-sft-dataset"
MODEL_REPO   = f"{HF_USERNAME}/gemma-4-2b-finance-qlora"

ROOT = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
ADAPTER_DIR = ROOT / "gemma-lora"


def push_dataset(api, token):
    print(f"Creating dataset repo: {DATASET_REPO}")
    create_repo(repo_id=DATASET_REPO, repo_type="dataset", token=token, exist_ok=True, private=False)

    for fname in ["combined_train.jsonl", "combined_val.jsonl"]:
        local = DATA_DIR / fname
        if not local.exists():
            print(f"  SKIP (not found): {fname}")
            continue
        print(f"  Uploading {fname}")
        api.upload_file(path_or_fileobj=str(local), path_in_repo=fname,
                        repo_id=DATASET_REPO, repo_type="dataset", token=token)

    print(f"Dataset live at: https://huggingface.co/datasets/{DATASET_REPO}")


def push_model(api, token):
    print(f"\nCreating model repo: {MODEL_REPO}")
    create_repo(repo_id=MODEL_REPO, repo_type="model", token=token, exist_ok=True, private=False)

    print(f"  Uploading adapter folder: {ADAPTER_DIR}")
    api.upload_folder(folder_path=str(ADAPTER_DIR), repo_id=MODEL_REPO,
                      repo_type="model", token=token)

    print(f"Model live at: https://huggingface.co/models/{MODEL_REPO}")


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not set in .env")

    api = HfApi(token=token)
    push_dataset(api, token)
    push_model(api, token)


if __name__ == "__main__":
    main()
