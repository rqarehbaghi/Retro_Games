#!/usr/bin/env bash
set -euo pipefail

# Reproducible CosyVoice 3 environment, isolated from RetroGames' trainer venv.
USER_ROOT="$(getent passwd "$(id -u)" | cut -d: -f6)"
CONDA_BIN="${COSYVOICE_CONDA:-$USER_ROOT/miniconda3/bin/conda}"
COSY_ROOT="${COSYVOICE_HOME:-$USER_ROOT/CosyVoice}"
ENV_ROOT="${COSYVOICE_ENV:-$USER_ROOT/miniconda3/envs/cosyvoice}"
MODEL_ROOT="${COSYVOICE_MODEL:-$COSY_ROOT/pretrained_models/Fun-CosyVoice3-0.5B}"
COSY_REF="${COSYVOICE_REF:-074ca6d}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Miniconda is required at $CONDA_BIN (or set COSYVOICE_CONDA)." >&2
  exit 1
fi
if [[ ! -d "$COSY_ROOT/.git" ]]; then
  git clone --recursive https://github.com/QwenAudio/CosyVoice.git "$COSY_ROOT"
fi
git -C "$COSY_ROOT" checkout "$COSY_REF"
git -C "$COSY_ROOT" submodule update --init --recursive

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  "$CONDA_BIN" create --prefix "$ENV_ROOT" -y --override-channels \
    -c conda-forge python=3.10 pip
fi
"$ENV_ROOT/bin/python" -m pip install setuptools==80.9.0 numpy==1.26.4 "Cython<3"
"$ENV_ROOT/bin/python" -m pip install torch==2.3.1 torchaudio==2.3.1 \
  --extra-index-url https://download.pytorch.org/whl/cu121
# DeepSpeed is training-only and TensorRT is an optional accelerator. Both
# require a full CUDA compiler/toolkit; ordinary GPU inference needs neither.
"$ENV_ROOT/bin/python" -m pip install --no-build-isolation \
  -r <(grep -vE '^(deepspeed==|tensorrt-cu12)' "$COSY_ROOT/requirements.txt")

cd "$COSY_ROOT"
"$ENV_ROOT/bin/python" -c \
  "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$MODEL_ROOT')"
"$ENV_ROOT/bin/python" -c \
  "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
echo "CosyVoice 3 is ready: $MODEL_ROOT"
