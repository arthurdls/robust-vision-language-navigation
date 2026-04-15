#!/usr/bin/env python3
"""
Start the OpenVLA inference server.

Loads weights from weights/OpenVLA-UAV/ by default and serves action predictions
via Flask HTTP on port 5007.

Usage (from repo root):
  python scripts/start_server.py
  python scripts/start_server.py --port 5007 --gpu-id 0
  python scripts/start_server.py --model-dir /path/to/checkpoint
  python scripts/start_server.py --device cpu        # pure CPU (fp32, slow)
  python scripts/start_server.py --device auto       # split GPU+CPU (bf16)
"""

import argparse
import logging
import sys
from pathlib import Path

from rvln.paths import REPO_ROOT
from rvln.server.openvla import OpenVLAActionAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def _resolve_model_path(raw: str) -> Path:
    """Resolve model path, auto-detecting a single subdirectory with config.json."""
    model_path = Path(raw)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    if not model_path.is_dir():
        print(f"Error: model dir not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not (model_path / "config.json").exists():
        subdirs = [
            d for d in model_path.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ]
        if len(subdirs) == 1:
            model_path = subdirs[0]
        elif subdirs:
            print(
                f"Error: {model_path} has multiple subdirs with config.json. "
                "Pass --model-dir to the specific checkpoint dir.",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(
                f"Error: no config.json in {model_path}.",
                file=sys.stderr,
            )
            sys.exit(1)

    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="Start OpenVLA inference server"
    )
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "weights"),
        help="Path to model checkpoint dir (default: weights/)",
    )
    parser.add_argument(
        "--port", type=int, default=5007,
        help="HTTP port for /predict endpoint (default: 5007)",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0,
        help="CUDA GPU id (only used when --device=cuda, default: 0)",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help=(
            "Inference device. 'cuda': single GPU with bf16 + flash-attn (default). "
            "'cpu': pure CPU fp32 (no GPU needed, slow). "
            "'auto': split layers across GPU+CPU via accelerate (bf16, no flash-attn)."
        ),
    )
    args = parser.parse_args()

    if args.device != "cuda" and args.gpu_id != 0:
        log.warning("--gpu-id=%d is ignored when --device=%s", args.gpu_id, args.device)

    model_path = _resolve_model_path(args.model_dir)
    log.info("Model path: %s", model_path)

    cfg = {
        "gpu_id": args.gpu_id,
        "device": args.device,
        "model_path": str(model_path),
        "http_port": args.port,
        "unnorm_key": "sim",
        "do_sample": False,
    }

    agent = OpenVLAActionAgent(cfg)

    # Add /reset so batch_run_act_all's reset_model() gets 200 instead of 404.
    # The OpenVLA server (openvla_act.py) only defines /predict; we add /reset here.
    @agent.app.route("/reset", methods=["POST"])
    def reset():
        return "", 200

    agent.run()


if __name__ == "__main__":
    main()
