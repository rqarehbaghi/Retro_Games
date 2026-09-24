#!/usr/bin/env python3
"""Can this machine run the project? Run it first on any new box.

    python tools/doctor.py

Every check says what is missing AND the one command that fixes it, because
the failures here are all the same kind: something that is not a pip package
(ffmpeg, a font, a ROM, a GPU driver, a model's weights) and therefore was not
installed by `pip install -r requirements.txt`.

REQUIRED things exit non-zero. OPTIONAL ones are features you may not want --
a writer backend, a voice, a GPU -- and only print what they would unlock.
"""
import importlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OK, BAD, MEH = "  ok  ", " MISS ", " note "
FAILED = []


def report(state, name, detail="", fix=""):
    print("[%s] %-26s %s" % (state, name, detail))
    if fix:
        print("%s%s" % (" " * 10, fix))
    if state == BAD:
        FAILED.append(name)


def need_import(module, package, why):
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "") or getattr(
            getattr(m, "version", None), "ver", "")
        report(OK, module, str(version))
        return True
    except Exception as exc:                                       # noqa: BLE001
        report(BAD, module, "%s -- %s" % (why, exc.__class__.__name__),
               "pip install %s" % package)
        return False


def optional_import(module, package, unlocks):
    try:
        importlib.import_module(module)
        report(OK, module, unlocks)
        return True
    except Exception:                                              # noqa: BLE001
        report(MEH, module, "not installed -- %s" % unlocks,
               "pip install %s" % package)
        return False


def need_binary(name, why, fix):
    path = shutil.which(name)
    report(OK if path else BAD, name, path or why, "" if path else fix)
    return bool(path)


def main():
    print("PYTHON")
    v = sys.version_info
    report(OK if v >= (3, 10) else BAD, "python", "%d.%d.%d" % v[:3],
           "" if v >= (3, 10) else "python3.10 or newer is required")
    if v >= (3, 13):
        report(MEH, "python >= 3.13",
               "the PyTorch build of kokoro caps at 3.12; kokoro-onnx is used instead")

    print("\nREQUIRED PYTHON PACKAGES")
    need_import("numpy", "numpy", "arrays, everywhere")
    need_import("gymnasium", "gymnasium", "the env API")
    need_import("stable_retro", "stable-retro", "the emulator")
    need_import("stable_baselines3", "stable-baselines3[extra]", "PPO training")
    need_import("torch", "torch", "the value net and PPO")
    need_import("cv2", "opencv-python-headless", "frame preprocessing")
    need_import("PIL", "Pillow", "every overlay and the stats card")
    need_import("pygame", "pygame-ce", "interactive play and the gamepad")
    need_import("matplotlib", "matplotlib", "training plots")

    print("\nREQUIRED SYSTEM TOOLS")
    need_binary("ffmpeg", "renders every video",
                "sudo apt install -y ffmpeg")
    need_binary("ffprobe", "measures every video",
                "sudo apt install -y ffmpeg")

    print("\nFONTS")
    press = os.path.expanduser("~/.local/share/fonts/PressStart2P-Regular.ttf")
    report(OK if os.path.exists(press) else MEH, "Press Start 2P",
           "the house face for captions and labels",
           "" if os.path.exists(press) else "bash tools/get_fonts.sh")
    dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    report(OK if os.path.exists(dejavu) else BAD, "DejaVu Sans",
           "the stats card and every fallback",
           "" if os.path.exists(dejavu) else "sudo apt install -y fonts-dejavu-core")

    print("\nGAMES")
    integrations = os.path.join(ROOT, "integrations")
    games = sorted(d for d in os.listdir(integrations)) if os.path.isdir(integrations) else []
    with_rom = [g for g in games
                if any(f.startswith("rom.") for f in
                       os.listdir(os.path.join(integrations, g)))]
    report(OK if with_rom else BAD, "integrations",
           "%d with a ROM: %s" % (len(with_rom), ", ".join(with_rom) or "none"),
           "" if with_rom else
           "ROMs are not in git, by design. Put yours in ROMs/ and run:\n"
           "          python tools/import_fixer.py")
    if games and len(games) != len(with_rom):
        report(MEH, "integrations without a ROM",
               ", ".join(sorted(set(games) - set(with_rom))))

    print("\nGPU (optional -- training works on CPU, slowly)")
    try:
        import torch
        if torch.cuda.is_available():
            report(OK, "cuda", "%s, torch %s" % (torch.cuda.get_device_name(0),
                                                 torch.__version__))
        else:
            report(MEH, "cuda", "not visible to torch %s -- training will use the CPU"
                   % torch.__version__)
    except Exception:                                              # noqa: BLE001
        pass

    print("\nWRITERS (needed for captions and narration -- any ONE will do)")
    writers = []
    if shutil.which("claude"):
        writers.append("claude-code")
        report(OK, "claude CLI", "billed to the Claude subscription")
    else:
        report(MEH, "claude CLI", "not installed", "npm i -g @anthropic-ai/claude-code")
    for env, name, note in (("ANTHROPIC_API_KEY", "Claude API", "prepaid API credits"),
                            ("GEMINI_API_KEY", "Gemini API", "Google AI Studio key")):
        if os.environ.get(env):
            writers.append(name)
            report(OK, name, note)
        else:
            report(MEH, name, "%s is not set -- %s" % (env, note))
    if shutil.which("ollama"):
        writers.append("ollama")
        report(OK, "ollama", "local and free")
    else:
        report(MEH, "ollama", "not installed -- the free local writer",
               "curl -fsSL https://ollama.com/install.sh | sh")
    if not writers:
        report(BAD, "a writer", "none available: captions and narration cannot be written")

    print("\nVOICE (optional -- only for --voice)")
    try:
        from tools.progression import speech
        for backend in ("kokoro", "qwen", "piper", "elevenlabs"):
            if speech.available(backend):
                extra = ""
                if backend == "kokoro":
                    extra = " (weights: %s)" % (speech.kokoro_files()[0] or "torch build")
                report(OK, backend, "installed%s" % extra)
            elif backend == "kokoro":
                have_pkg = importlib.util.find_spec("kokoro_onnx") is not None
                report(MEH, "kokoro",
                       "weights missing" if have_pkg else "not installed",
                       "bash tools/get_kokoro.sh" if have_pkg else
                       "pip install kokoro-onnx soundfile && bash tools/get_kokoro.sh")
            else:
                report(MEH, backend, "not installed")
    except Exception as exc:                                       # noqa: BLE001
        report(MEH, "voice", "could not be checked: %s" % exc)

    print("")
    if FAILED:
        print("%d required check%s failed: %s"
              % (len(FAILED), "" if len(FAILED) == 1 else "s", ", ".join(FAILED)))
        return 1
    print("Everything required is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
