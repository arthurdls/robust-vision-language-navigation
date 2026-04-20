#!/usr/bin/env python3
"""Download OpenVLA-UAV model weights from HuggingFace.

Default repo: ``wangxiangyu0814/OpenVLA-UAV``. Private or gated checkpoints need
``huggingface-cli login`` / ``HF_TOKEN`` and accepting terms on the model page.

Usage:
    python tools/download_weights.py
    python tools/download_weights.py --repo wangxiangyu0814/OpenVLA-UAV --dest weights/OpenVLA-UAV
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "wangxiangyu0814/OpenVLA-UAV"
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

    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        from huggingface_hub.utils import RepositoryNotFoundError

    dest = args.dest
    if dest.exists() and any(dest.glob("*.safetensors")):
        print(f"Weights already present at {dest}")
        print("To re-download, delete the directory first.")
        return

    print(f"Downloading {args.repo} -> {dest}")
    print("This may take a while (model is several GB)...")
    dest.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=args.repo,
            local_dir=str(dest),
        )
    except RepositoryNotFoundError as err:
        print(
            "Hugging Face refused access to this model (often 401/404).\n"
            "  - Log in: huggingface-cli login   (or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN)\n"
            "  - If the repo is gated: open the model page while logged in and accept the terms\n"
            "  - If you see 'Invalid username or password': your stored token is wrong or expired; "
            "log in again or unset a bad HF_TOKEN in your shell/env files\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from err

    print(f"Weights downloaded to {dest}")


if __name__ == "__main__":
    main()
