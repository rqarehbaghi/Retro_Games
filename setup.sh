#!/usr/bin/env bash
# One-shot project setup for Ubuntu -- unchanged in WSL2 or on an EC2 box.
#
#   git clone <this repo> && cd Retro_Games
#   bash setup.sh
#
# It installs the system packages pip cannot, makes ./venv, installs
# requirements.txt, fetches the font, and finishes by running the doctor, which
# is the thing that actually tells you whether this machine can run the
# project. Three things it deliberately does NOT do:
#
#   ROMs      copyrighted, never in git. Put yours in ROMs/ and run
#             `python tools/import_fixer.py` afterwards.
#   the voice ~350MB of weights that most runs do not need. `bash
#             tools/get_kokoro.sh` when you want --voice.
#   login      Codex authentication opens a browser/device flow. The CLI is
#             installed below; run `codex login --device-auth` yourself.
#
# Re-running is safe: every step checks before it acts.
set -euo pipefail

SKIP_APT="${SKIP_APT:-0}"

if [ "$SKIP_APT" != "1" ]; then
    echo "==> System packages (sudo)"
    sudo apt update
    # python3-venv/pip  the venv below
    # build-essential cmake  stable-retro compiles its cores
    # ffmpeg  encodes and measures every video in this project
    # fonts-dejavu-core  the stats card, and the fallback face everywhere else
    # libgl1 libglu1-mesa freeglut3-dev mesa-utils  OpenGL, for the play window
    # espeak-ng  phonemes for some TTS paths (kokoro-onnx bundles its own)
    sudo apt install -y python3 python3-venv python3-pip build-essential cmake \
        curl ffmpeg git fonts-dejavu-core libglu1-mesa libgl1 freeglut3-dev mesa-utils
else
    echo "==> Skipping apt (SKIP_APT=1)"
fi

echo ""
echo "==> Virtual environment (./venv)"
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install --upgrade pip
# Qwen3-TTS and Chatterbox pin incompatible Transformers versions. Chatterbox
# is the supported narrator; remove the old optional backend before resolving
# requirements so an existing checkout upgrades as cleanly as a fresh one.
if ./venv/bin/pip show qwen-tts >/dev/null 2>&1; then
    ./venv/bin/pip uninstall -y qwen-tts
fi
./venv/bin/pip install -r requirements.txt

echo ""
echo "==> Fonts"
bash tools/get_fonts.sh || echo "  (font download failed -- DejaVu will be used instead)"

echo ""
echo "==> ChatGPT writer (Codex CLI)"
if command -v codex >/dev/null 2>&1; then
    codex --version
else
    curl -fsSL https://chatgpt.com/codex/install.sh | sh
fi
if ! codex login status >/dev/null 2>&1; then
    echo "  Codex is installed but not logged in. After setup, run:"
    echo "    codex login --device-auth"
    echo "    codex login status"
fi

echo ""
echo "==> GPU video encoding (optional, never fatal)"
bash tools/check_nvenc.sh || true

echo ""
echo "==> Checking this machine"
./venv/bin/python tools/doctor.py || true

cat <<'NEXT'

----------------------------------------------------------------------
Next, in order:

  source venv/bin/activate

  # 1. Attach the writer to the existing ChatGPT subscription inside WSL:
  codex login --device-auth
  codex login status   # must say: Logged in using ChatGPT

  # 2. ROMs are not in git. Put yours in ROMs/ and import them:
  python tools/import_fixer.py

  # 3. Play, record and cut a video in one command:
  python studio.py --game SuperMarioBros3-Nes-v0

  # 4. Train an agent (its algorithm and rewards come from games.json):
  python train.py --game TetrisTime-Nes-v0 --describe

  # 5. Optional, only for spoken narration (~350MB):
  bash tools/get_kokoro.sh
----------------------------------------------------------------------
NEXT
