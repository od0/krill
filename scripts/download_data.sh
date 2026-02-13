#!/usr/bin/env bash
# Download ALFWorld SFT trajectories from HuggingFace
# Source: https://huggingface.co/datasets/agent-eto/eto-sft-trajectory

set -euo pipefail

DEST="${1:-data/alfworld_raw.json}"
URL="https://huggingface.co/datasets/agent-eto/eto-sft-trajectory/resolve/main/data/alfworld_sft.json"

mkdir -p "$(dirname "$DEST")"
echo "Downloading ALFWorld SFT trajectories to $DEST..."
curl -L -o "$DEST" "$URL"
echo "Done ($(du -h "$DEST" | cut -f1) downloaded)"
