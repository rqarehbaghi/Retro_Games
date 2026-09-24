#!/usr/bin/env bash
# Press Start 2P -- the face every caption, title and panel label is drawn in.
#
# It is not an apt package. DejaVu (apt) is the fallback the code uses when
# this is absent, so a machine without it still renders; it just does not look
# like the channel.
set -euo pipefail

DIR="$HOME/.local/share/fonts"
URL="https://github.com/google/fonts/raw/main/ofl/pressstart2p/PressStart2P-Regular.ttf"

mkdir -p "$DIR"
if [ -s "$DIR/PressStart2P-Regular.ttf" ]; then
    echo "already have $DIR/PressStart2P-Regular.ttf"
else
    curl -sSL --fail -o "$DIR/PressStart2P-Regular.ttf" "$URL"
    command -v fc-cache >/dev/null && fc-cache -f "$DIR" >/dev/null 2>&1 || true
    echo "installed $DIR/PressStart2P-Regular.ttf"
fi
