#!/usr/bin/env python3
"""Download OpenVLA-UAV model weights from HuggingFace.

Usage:
    python tools/download_weights.py
    python tools/download_weights.py --repo CogACT/OpenVLA-UAV --dest weights/OpenVLA-UAV
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "CogACT/OpenVLA-UAV"
DEFAULT_DEST = REPO_ROOT / "weights" / "OpenVLA-UAV"


def main():
    parser = argparse.ArgumentParser(description="Download OpenVLA-UAV weights from HuggingFace")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"HuggingFace repo ID (default: {DEFAULT_REPO})")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"Local destination (default: {DEFAULT_DEST})")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is not installed. Run: pip install huggingface-hub", file=sys.stderr)
        sys.exit(1)

    dest = args.dest
    if dest.exists() and any(dest.glob("*.safetensors")):
        print(f"Weights already present at {dest}")
        print("To re-download, delete the directory first.")
        return

    print(f"Downloading {args.repo} -> {dest}")
    print("This may take a while (model is several GB)...")
    dest.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )

    print(f"Weights downloaded to {dest}")


if __name__ == "__main__":
    main()
