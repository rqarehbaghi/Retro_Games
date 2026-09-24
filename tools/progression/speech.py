"""Which voice speaks the narration, and how consistent it stays.

The pipeline's own Qwen3-TTS was rendered ONE SENTENCE PER CALL, which is what
made it sound like a machine reading a list: every call is a fresh inference,
so pitch, pace and emphasis reset at each full stop, and the joins land on
silence rather than on breath. Two things fix that, and this module does both:

  1. A BLOCK, not a line. A whole paragraph goes to the model in one call, so
     the sentences inside it are spoken as one thought, with the model's own
     pauses between them.
  2. A model whose voice is a FIXED EMBEDDING rather than a fresh
     interpretation of a description each time.

Backends, with what each one costs:

    kokoro    Kokoro-82M, Apache-2.0, ~82M params, local, free. Named voices
              are fixed embeddings, so intonation does not drift between
              blocks. The recommended default for narration. Chunks long text
              itself and keeps one voice across the chunks.
              pip install "kokoro>=0.9.4" soundfile   (plus: apt install espeak-ng)
    qwen      Qwen3-TTS CustomVoice, Apache-2.0, local, free, ~4GB. What the
              studio pipeline uses. Expressive, but drifts between calls.
    piper     Piper, MIT, local, free, tiny and fast. Flat but utterly
              consistent -- the safe fallback on a machine with no GPU.
    elevenlabs  PAID, and it sends the script to a third party. Only used if
              explicitly asked for and ELEVENLABS_API_KEY is set.

Nothing here is installed by this repo. `available()` reports what is actually
importable on this machine so a run fails before the emulator does, not after.
"""
import os
import subprocess

SAMPLE_RATE = 24000
DEFAULT = "kokoro"

# Kokoro's voices are named embeddings. a = American English, b = British.
# The first letter after that is the gender. These are the ones worth trying
# for commentary; the full list is in the model card.
KOKORO_VOICES = ("am_michael", "am_adam", "bm_george", "bm_lewis",
                 "af_heart", "af_bella", "bf_emma")
KOKORO_DEFAULT = "am_michael"


def available(backend=DEFAULT):
    """Is this backend actually usable HERE? Checked before a run starts."""
    if backend == "kokoro":
        try:
            import kokoro                                          # noqa: F401
            return True
        except Exception:                                          # noqa: BLE001
            return False
    if backend == "qwen":
        try:
            import tts
            return tts.available()
        except Exception:                                          # noqa: BLE001
            return False
    if backend == "piper":
        return bool(_which("piper"))
    if backend == "elevenlabs":
        return bool(os.environ.get("ELEVENLABS_API_KEY"))
    return False


def _which(name):
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(path, name)
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def seconds(path):
    """Measured, never estimated. soundfile is a dependency of every backend
    here, so this cannot be the thing that is missing."""
    import soundfile as sf
    info = sf.info(path)
    return info.frames / float(info.samplerate)


# ------------------------------------------------------------------ kokoro --
def _speak_kokoro(blocks, out_dir, voice=None, speed=1.0, verbose=True):
    """One wav per block, one voice for the whole run.

    KPipeline splits a long block into chunks itself and returns them in
    order; they are joined here so a block is a single continuous file. The
    voice is a named embedding, so block 9 sounds like block 1."""
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=(voice or KOKORO_DEFAULT)[0])
    out = []
    for i, text in enumerate(blocks):
        chunks = [audio for _gs, _ps, audio in
                  pipeline(text, voice=voice or KOKORO_DEFAULT, speed=speed)]
        if not chunks:
            continue
        audio = np.concatenate([np.asarray(c, dtype="float32") for c in chunks])
        path = os.path.join(out_dir, "block_%02d.wav" % i)
        sf.write(path, audio, SAMPLE_RATE)
        out.append(path)
        if verbose:
            print("    [%2d/%2d] %.1fs  %s" % (i + 1, len(blocks),
                                               len(audio) / float(SAMPLE_RATE),
                                               text[:58]))
    return out


# -------------------------------------------------------------------- qwen --
def _speak_qwen(blocks, out_dir, voice=None, speed=1.0, verbose=True):
    """The studio pipeline's voice, but a block at a time rather than a line."""
    import tts
    lines = [{"text": t} for t in blocks]
    model = None
    try:
        model = tts.load()
    except Exception as exc:                                       # noqa: BLE001
        print("  [voice] GPU load failed (%s), falling back to CPU" % exc)
        model = tts.load(device="cpu")
    clips = tts.speak_lines(lines, out_dir, model=model,
                            speaker=voice or tts.DEFAULT_SPEAKER, verbose=verbose)
    return [path for _at, path, _closing in clips]


# ------------------------------------------------------------------- piper --
def _speak_piper(blocks, out_dir, voice=None, speed=1.0, verbose=True):
    """Piper wants a model path; `voice` IS that path here."""
    if not voice:
        raise RuntimeError("piper needs --voice-name pointing at a .onnx model")
    out = []
    for i, text in enumerate(blocks):
        path = os.path.join(out_dir, "block_%02d.wav" % i)
        subprocess.run(["piper", "--model", voice, "--output_file", path],
                       input=text.encode("utf-8"), check=True, capture_output=True)
        out.append(path)
        if verbose:
            print("    [%2d/%2d] %s" % (i + 1, len(blocks), text[:58]))
    return out


# -------------------------------------------------------------- elevenlabs --
def _speak_elevenlabs(blocks, out_dir, voice=None, speed=1.0, verbose=True):
    """PAID, and the script leaves this machine. Opt in only.

    Kept deliberately thin: a key, a voice id, one request per block."""
    import json
    import urllib.request

    key = os.environ["ELEVENLABS_API_KEY"]
    voice_id = voice or "JBFqnCBsd6RMkjVDRZzb"
    out = []
    for i, text in enumerate(blocks):
        req = urllib.request.Request(
            "https://api.elevenlabs.io/v1/text-to-speech/%s" % voice_id,
            data=json.dumps({"text": text, "model_id": "eleven_multilingual_v2"}).encode(),
            headers={"xi-api-key": key, "Content-Type": "application/json",
                     "Accept": "audio/mpeg"})
        mp3 = os.path.join(out_dir, "block_%02d.mp3" % i)
        with urllib.request.urlopen(req, timeout=120) as resp, open(mp3, "wb") as fh:
            fh.write(resp.read())
        wav = os.path.join(out_dir, "block_%02d.wav" % i)
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", mp3,
                        "-ar", str(SAMPLE_RATE), "-ac", "1", wav], check=True)
        out.append(wav)
        if verbose:
            print("    [%2d/%2d] %s" % (i + 1, len(blocks), text[:58]))
    return out


SPEAKERS = {"kokoro": _speak_kokoro, "qwen": _speak_qwen,
            "piper": _speak_piper, "elevenlabs": _speak_elevenlabs}


def speak(blocks, out_dir, backend=DEFAULT, voice=None, speed=1.0, verbose=True):
    """Render each block to its own wav. Returns [(path, seconds)]."""
    if backend not in SPEAKERS:
        raise ValueError("unknown voice backend %r; have %s"
                         % (backend, ", ".join(sorted(SPEAKERS))))
    if not available(backend):
        raise RuntimeError("voice backend %r is not installed here" % backend)
    os.makedirs(out_dir, exist_ok=True)
    paths = SPEAKERS[backend]([b for b in blocks if b and b.strip()],
                              out_dir, voice=voice, speed=speed, verbose=verbose)
    return [(p, seconds(p)) for p in paths]
