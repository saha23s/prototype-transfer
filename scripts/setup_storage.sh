#!/usr/bin/env bash
# One-time setup: put heavy files (data, model cache, venv) on a large disk and
# symlink them into the repo, then install dependencies with uv.
#
# Usage: scripts/setup_storage.sh /path/to/big/disk/prototype-transfer
#   (or set PROTO_STORAGE_DIR instead of passing the argument)
set -euo pipefail

STORAGE_DIR="${1:-${PROTO_STORAGE_DIR:-}}"
if [[ -z "$STORAGE_DIR" ]]; then
    echo "Usage: $0 <storage_dir>   (or export PROTO_STORAGE_DIR)" >&2
    exit 1
fi
command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$STORAGE_DIR"/{data,models}

for name in data models; do
    link="$REPO_DIR/$name"
    if [[ -e "$link" || -L "$link" ]]; then
        echo "skip: $link already exists"
    else
        ln -s "$STORAGE_DIR/$name" "$link"
        echo "linked: $link -> $STORAGE_DIR/$name"
    fi
done

# Create the venv on the big disk, then expose it as ./.venv
cd "$REPO_DIR"
UV_PROJECT_ENVIRONMENT="$STORAGE_DIR/venv" uv sync
[[ -e .venv || -L .venv ]] || ln -s "$STORAGE_DIR/venv" .venv

echo "Setup complete. Run: scripts/run_pipeline.sh --smoke"
