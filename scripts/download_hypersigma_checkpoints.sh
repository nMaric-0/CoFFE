#!/usr/bin/env bash
# Download the released HyperSIGMA SpatViT-B and SpecViT-B checkpoints
# into ./checkpoints/hypersigma/. The upstream artifacts live on
# HuggingFace under https://huggingface.co/WHU-Sigma/HyperSIGMA.
#
# The exact files we want are the ImageClassification MAE pretraining
# checkpoints (the ones ss_fusion_cls.py expects to load via
# init_weights()). At the time of vendoring (upstream commit
# c5981e0666b39047f8896be450d8a17cc20f38b0) these are named:
#     spat-vit-base-ultra-checkpoint-1599.pth
#     spec-vit-base-ultra-checkpoint-1599.pth
#
# If the HuggingFace tree layout changes, override the URLs with the
# environment variables SPAT_URL / SPEC_URL.

set -euo pipefail

DEST_DIR="${1:-./checkpoints/hypersigma}"
mkdir -p "$DEST_DIR"

SPAT_URL="${SPAT_URL:-https://huggingface.co/WHU-Sigma/HyperSIGMA/resolve/main/spat-vit-base-ultra-checkpoint-1599.pth}"
SPEC_URL="${SPEC_URL:-https://huggingface.co/WHU-Sigma/HyperSIGMA/resolve/main/spec-vit-base-ultra-checkpoint-1599.pth}"

download() {
    local url="$1"
    local out="$2"
    if [[ -f "$out" ]]; then
        echo "[skip] $out already exists"
        return
    fi
    echo "[get ] $url"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 4 --retry-delay 4 -o "$out" "$url"
    else
        wget --tries=4 --timeout=120 -O "$out" "$url"
    fi
}

download "$SPAT_URL" "$DEST_DIR/spat-vit-base.pth"
download "$SPEC_URL" "$DEST_DIR/spec-vit-base.pth"

echo
echo "SHA256 of downloaded checkpoints:"
sha256sum "$DEST_DIR/spat-vit-base.pth" "$DEST_DIR/spec-vit-base.pth"
