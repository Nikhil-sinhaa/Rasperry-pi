
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path

REPO_ID = "vaquill/open-india-law"
OUTPUT_DIR = Path("./open-india-law-regulations")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi()

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

regulation_files = [
    f for f in files
    if Path(f).name.startswith("in_")
    and Path(f).name.endswith("_regulations.parquet")
]

if not regulation_files:
    raise RuntimeError("No regulation Parquet files found.")

print(f"Found {len(regulation_files)} regulation files.")

for i, filename in enumerate(regulation_files, start=1):
    print(f"[{i}/{len(regulation_files)}] Downloading {filename}")

    hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(OUTPUT_DIR)
    )

print(f"\nDownload complete: {OUTPUT_DIR.resolve()}")
