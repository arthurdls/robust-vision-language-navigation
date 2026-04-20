#!/usr/bin/env python3
"""Download the Unreal Engine simulation environment and textures.

Downloads from ModelScope (UnrealZoo) and extracts under runtime/unreal/
(by default; matches ``rvln.paths.UNREAL_ENV_ROOT``).
Based on UAV-Flow-Eval/load_env.py.

Usage:
    python tools/download_simulator.py                # Downloads simulator + textures
    python tools/download_simulator.py --env-only     # Simulator only
    python tools/download_simulator.py --textures-only  # Textures only
"""

import argparse
import os
import platform
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = REPO_ROOT / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))
try:
    from rvln.paths import UNREAL_ENV_ROOT as DEFAULT_UNREAL_ROOT
except ImportError:
    DEFAULT_UNREAL_ROOT = REPO_ROOT / "runtime" / "unreal"

MODELSCOPE_REPO_UE4 = "UnrealZoo/UnrealZoo-UE4"

BINARIES_LINUX = {
    "simulator": "Collection_v4_LinuxNoEditor.zip",
    "textures": "Textures.zip",
}

BINARIES_WINDOWS = {
    "simulator": "UE4_ExampleScene_Win.zip",
    "textures": "Textures.zip",
}

BINARIES_MAC = {
    "simulator": "UE4_ExampleScene_Mac.zip",
    "textures": "Textures.zip",
}


def get_platform_binaries():
    system = platform.system().lower()
    if "linux" in system:
        return BINARIES_LINUX
    elif "darwin" in system:
        return BINARIES_MAC
    elif "win" in system:
        return BINARIES_WINDOWS
    else:
        print(f"Unsupported platform: {system}", file=sys.stderr)
        sys.exit(1)


def download_from_modelscope(filename, dest_dir):
    """Download a file from ModelScope using the CLI."""
    cmd = [
        "modelscope", "download",
        "--dataset", MODELSCOPE_REPO_UE4,
        "--include", filename,
        "--local_dir", dest_dir,
    ]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: modelscope CLI not found. Install with: pip install modelscope", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)


def extract_and_move(zip_path, dest_dir, is_textures=False):
    """Extract zip and move contents to the UnrealEnv directory."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(str(zip_path.parent))

    if is_textures:
        folder_name = zip_path.stem
    else:
        folder_name = zip_path.stem

    source = zip_path.parent / folder_name
    target = dest_dir / folder_name

    if target.exists():
        print(f"Target {target} already exists, removing old version...")
        shutil.rmtree(target)

    print(f"Moving {source} -> {target}")
    shutil.move(str(source), str(target))

    zip_path.unlink()
    print(f"Cleaned up {zip_path}")

    if not is_textures and "linux" in platform.system().lower():
        for f in target.rglob("*.sh"):
            f.chmod(f.stat().st_mode | stat.S_IEXEC)
        for f in target.rglob("Collection"):
            if f.is_file():
                f.chmod(f.stat().st_mode | stat.S_IEXEC)


def main():
    parser = argparse.ArgumentParser(description="Download Unreal simulation environment")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_UNREAL_ROOT,
        help=f"Destination directory (default: {DEFAULT_UNREAL_ROOT})",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--env-only", action="store_true", help="Download simulator only (skip textures)")
    group.add_argument("--textures-only", action="store_true", help="Download textures only (skip simulator)")
    parser.add_argument("--force", action="store_true", help="Re-download even if already installed")
    args = parser.parse_args()

    binaries = get_platform_binaries()
    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    to_download = []
    if not args.textures_only:
        to_download.append(("simulator", binaries["simulator"], False))
    if not args.env_only:
        to_download.append(("textures", binaries["textures"], True))

    for name, filename, is_tex in to_download:
        target = dest / Path(filename).stem
        if target.exists() and not args.force:
            print(f"{name} already installed at {target}")
            continue

        zip_path = dest / filename
        if not zip_path.exists():
            print(f"\nDownloading {name}: {filename}")
            download_from_modelscope(filename, str(dest))

        if not zip_path.exists():
            print(f"Error: expected {zip_path} after download", file=sys.stderr)
            continue

        extract_and_move(zip_path, dest, is_textures=is_tex)

    print(f"\nSimulator files installed to {dest}")
    print("Set UnrealEnv environment variable if needed:")
    print(f"  export UnrealEnv={dest}")


if __name__ == "__main__":
    main()
