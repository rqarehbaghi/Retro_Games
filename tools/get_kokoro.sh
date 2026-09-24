#!/usr/bin/env bash
# The Kokoro voice model's weights. Two files, about 350MB, downloaded once.
#
# They are NOT in git and never should be: the repo would be unclonable. They
# live in a cache directory that tools/progression/speech.py already looks in,
# so nothing needs configuring after this runs. Set KOKORO_ONNX_MODEL and
# KOKORO_ONNX_VOICES to override the location.
set -euo pipefail

DIR="${1:-$HOME/.cache/kokoro-onnx}"
BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"

mkdir -p "$DIR"
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
    if [ -s "$DIR/$f" ]; then
        echo "already have $DIR/$f"
    else
        echo "downloading $f ..."
        curl -sSL --fail -o "$DIR/$f" "$BASE/$f"
    fi
done
ls -la "$DIR"
echo ""
echo "Done. Check it with:  python tools/doctor.py"
