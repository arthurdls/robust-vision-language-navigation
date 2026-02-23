#!/usr/bin/env python3
"""
Start the OpenVLA inference server using the original UAV-Flow code
(UAV-Flow/OpenVLA-UAV/vla-scripts/openvla_act.py) with no modifications.

Loads weights from weights/OpenVLA-UAV/ by default.

Usage (from repo root):
  python scripts/start_openvla_server.py
  python scripts/start_openvla_server.py --port 5007 --gpu-id 0
  python scripts/start_openvla_server.py --model-dir /path/to/checkpoint
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OPENVLA_ACT_PATH = (
    _REPO_ROOT / "UAV-Flow" / "OpenVLA-UAV" / "vla-scripts" / "openvla_act.py"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def _load_openvla_act_module():
    """Load the original openvla_act module via importlib (vla-scripts has a hyphen)."""
    if not _OPENVLA_ACT_PATH.exists():
        print(
            f"Error: original server code not found at {_OPENVLA_ACT_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("openvla_act", str(_OPENVLA_ACT_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_model_path(raw: str) -> Path:
    """Resolve model path, auto-detecting a single subdirectory with config.json."""
    model_path = Path(raw)
    if not model_path.is_absolute():
        model_path = _REPO_ROOT / model_path

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
        description="Start OpenVLA server using original UAV-Flow code"
    )
    parser.add_argument(
        "--model-dir",
        default=str(_REPO_ROOT / "weights"),
        help="Path to model checkpoint dir (default: weights/)",
    )
    parser.add_argument(
        "--port", type=int, default=5007,
        help="HTTP port for /predict endpoint (default: 5007)",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0,
        help="CUDA GPU id (default: 0)",
    )
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model_dir)
    log.info("Model path: %s", model_path)

    openvla_act = _load_openvla_act_module()
    log.info("Loaded original OpenVLAActionAgent from %s", _OPENVLA_ACT_PATH)

    cfg = {
        "gpu_id": args.gpu_id,
        "model_path": str(model_path),
        "http_port": args.port,
        "unnorm_key": "sim",
        "do_sample": False,
    }

    agent = openvla_act.OpenVLAActionAgent(cfg)

    # Add /reset so batch_run_act_all's reset_model() gets 200 instead of 404.
    # The OpenVLA server (openvla_act.py) only defines /predict; we add /reset here.
    @agent.app.route("/reset", methods=["POST"])
    def reset():
        return "", 200

    agent.run()


if __name__ == "__main__":
    main()
