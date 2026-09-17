#!/usr/bin/env bash
# Can ffmpeg encode video on the NVIDIA GPU (NVENC)?
#
#   bash tools/check_nvenc.sh
#
# Nothing here is installed with pip. NVENC is part of the NVIDIA DRIVER: in
# WSL2 the Windows driver maps libcuda / libnvidia-encode into /usr/lib/wsl/lib,
# and on native Linux the driver package provides them. Ubuntu's ffmpeg is
# already built with h264_nvenc. So this proves it with a real one-second test
# encode -- listing the encoder proves nothing, Ubuntu's ffmpeg always lists it
# -- and, when it fails, says which of the three layers is the problem.
#
# It never fails setup: without NVENC, rendering falls back to libx264.
set -uo pipefail

ok()   { printf '  [ok]   %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; }
note() { printf '         %s\n' "$*"; }

echo "NVENC check"

if ! command -v ffmpeg >/dev/null 2>&1; then
    bad "ffmpeg is not installed"
    note "sudo apt install -y ffmpeg"
    exit 0
fi
# Captured first: piping straight into `grep -q` lets grep exit on the first
# match, ffmpeg dies of SIGPIPE, and under pipefail that reads as "missing".
encoders=$(ffmpeg -hide_banner -encoders 2>/dev/null)
if ! printf '%s\n' "$encoders" | grep -q h264_nvenc; then
    bad "this ffmpeg was built without NVENC"
    note "Ubuntu's own package has it: sudo apt install -y ffmpeg"
    exit 0
fi
ok "ffmpeg has h264_nvenc"

err=$(ffmpeg -nostdin -v error -f lavfi -i color=black:s=256x256:d=1 \
      -c:v h264_nvenc -f null - 2>&1)
if [ $? -eq 0 ]; then
    ok "test encode on the GPU succeeded -- NVENC is available (set style \"encoder\": \"nvenc\" or \"auto\" to use it)"
    exit 0
fi
bad "test encode failed: $(printf '%s' "$err" | tail -n 1)"

in_wsl=0
grep -qi microsoft /proc/version 2>/dev/null && in_wsl=1

# Ask the CUDA driver directly: its error code separates "no GPU visible"
# from "driver libraries missing".
cuda=$(python3 - <<'EOF' 2>&1
import ctypes
try:
    lib = ctypes.CDLL("libcuda.so.1")
except OSError as exc:
    print("NOLIB", exc)
else:
    n = ctypes.c_int()
    print("RC", lib.cuInit(0), lib.cuDeviceGetCount(ctypes.byref(n)), n.value)
EOF
)

case "$cuda" in
  NOLIB*)
    bad "libcuda.so.1 not found -- the NVIDIA driver is not reachable"
    if [ $in_wsl -eq 1 ]; then
        note "Install/update the NVIDIA driver on WINDOWS (not inside WSL),"
        note "then run 'wsl --shutdown' in PowerShell and reopen Ubuntu."
        note "Expected files: /usr/lib/wsl/lib/libcuda.so.1 and libnvidia-encode.so.1"
    else
        note "Install the NVIDIA driver, e.g.: sudo ubuntu-drivers install"
    fi
    ;;
  "RC 100"*)
    bad "CUDA reports NO GPU (error 100): the driver loads but sees no device"
    if [ $in_wsl -eq 1 ]; then
        note "WSL cannot see the GPU because Windows cannot. Seen on a laptop whose"
        note "discrete GPU was powered off: Device Manager listed it as not present."
        note "  1. Set the laptop's GPU mode to Hybrid or dGPU (e.g. Lenovo Vantage /"
        note "     Legion Space), and plug in AC power."
        note "  2. If it is still missing, reboot Windows."
        note "  3. Then run 'wsl --shutdown' in PowerShell and reopen Ubuntu."
        note "Check from PowerShell: Get-PnpDevice -Class Display  (Present must be True)"
    else
        note "Check 'nvidia-smi'. A GPU that is missing there is a driver/BIOS issue."
    fi
    ;;
  RC\ 0*)
    bad "CUDA works, so the ENCODER library or its version is the problem"
    if [ $in_wsl -eq 1 ]; then
        ls /usr/lib/wsl/lib/libnvidia-encode.so* >/dev/null 2>&1 \
            || note "libnvidia-encode is missing from /usr/lib/wsl/lib: update the Windows driver."
    else
        note "Install the encoder library for your driver, e.g.: sudo apt install libnvidia-encode-<version>"
    fi
    note "An error mentioning 'API version' means the driver is older than this ffmpeg needs:"
    note "update the NVIDIA driver."
    ;;
  *)
    bad "could not query CUDA: $cuda"
    ;;
esac
note "Renders use libx264 by default anyway; this only matters if style encoder is nvenc or auto."
exit 0
