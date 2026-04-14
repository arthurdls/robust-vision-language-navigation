#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== rvln-adls setup ==="
echo "Repo root: $REPO_ROOT"
cd "$REPO_ROOT"

# --- Conda environments ---
echo ""
echo "--- Setting up conda environments ---"

for envfile in rvln-sim_env.yml rvln-server_env.yml; do
    if [ -f "$envfile" ]; then
        envname=$(head -5 "$envfile" | grep "^name:" | awk '{print $2}')
        if conda info --envs 2>/dev/null | grep -q "^$envname "; then
            echo "Updating existing env: $envname"
            conda env update -f "$envfile" --prune
        else
            echo "Creating env: $envname"
            conda env create -f "$envfile"
        fi
    fi
done

# --- pip install (editable) ---
echo ""
echo "--- Installing rvln package (editable) ---"
echo "Note: Run this inside your target conda env (rvln-sim or rvln-server)"
echo "  conda activate rvln-sim && pip install -e ."
echo "  conda activate rvln-server && pip install -e '.[server]'"

# --- .env file ---
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo ""
        echo "Created .env from .env.example -- edit it with your API keys."
    fi
else
    echo ""
    echo ".env already exists."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Download model weights:    python tools/download_weights.py"
echo "  3. Download simulator:        python tools/download_simulator.py"
echo "  4. Start the server:          conda activate rvln-server && python scripts/start_server.py"
echo "  5. Run the sim:               conda activate rvln-sim && python scripts/run_integration.py"
