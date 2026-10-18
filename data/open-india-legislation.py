
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path

REPO_ID = "vaquill/open-india-law"
OUTPUT_DIR = Path("./open-india-law-legislation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi()

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

legislation_files = [
    f for f in files
    if f.endswith(".parquet")
    and "regulations" not in Path(f).name.lower()
]

if not legislation_files:
    raise RuntimeError("No legislation Parquet files found.")

print(f"Found {len(legislation_files)} Parquet files.")

# Download each file
for i, filename in enumerate(legislation_files, start=1):
    print(f"[{i}/{len(legislation_files)}] Downloading {filename}")

    hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(OUTPUT_DIR)
    )

print(f"\nDataset downloaded successfully!")
print(f"Saved to: {OUTPUT_DIR.resolve()}")
