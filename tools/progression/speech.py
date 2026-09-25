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

# How fast each voice actually speaks. The script is sized from this, so a
# wrong figure is silence at the end of the film or lines cut off it.
#
# kokoro and qwen are MEASURED on rendered blocks on this machine (kokoro:
# 17 words in 5.72s and 13 in 4.78s; qwen: 3:07 of speech for a ~300 word
# script). piper and elevenlabs are not measured here and take the general
# default -- check one block and correct them rather than trusting these.
WPM = {"kokoro": 170, "qwen": 130}
DEFAULT_WPM = 140


def words_per_minute(backend):
    return WPM.get(backend, DEFAULT_WPM)

# Kokoro's voices are named embeddings. a = American English, b = British.
# The first letter after that is the gender. These are the ones worth trying
# for commentary; the full list is in the model card.
KOKORO_VOICES = ("am_michael", "am_adam", "bm_george", "bm_lewis",
                 "af_heart", "af_bella", "bf_emma")
KOKORO_DEFAULT = "am_michael"


# Where the ONNX weights live. Two files, ~350MB, downloaded once -- too big
# for the repo, so they sit outside it and are found by env var, then by the
# usual cache, then beside the project.
KOKORO_MODEL_ENV = "KOKORO_ONNX_MODEL"
KOKORO_VOICES_ENV = "KOKORO_ONNX_VOICES"
KOKORO_DIRS = ("~/.cache/kokoro-onnx", "~/.local/share/kokoro-onnx", ".")


def kokoro_files():
    """(model, voices) if both are on disk, else (None, None)."""
    model = os.environ.get(KOKORO_MODEL_ENV)
    voices = os.environ.get(KOKORO_VOICES_ENV)
    if model and voices and os.path.exists(model) and os.path.exists(voices):
        return model, voices
    for d in KOKORO_DIRS:
        d = os.path.expanduser(d)
        m, v = os.path.join(d, "kokoro-v1.0.onnx"), os.path.join(d, "voices-v1.0.bin")
        if os.path.exists(m) and os.path.exists(v):
            return m, v
    return None, None


def available(backend=DEFAULT):
    """Is this backend actually usable HERE? Checked before a run starts."""
    if backend == "kokoro":
        # Two packages, same model. `kokoro` is the PyTorch one and caps at
        # Python < 3.13; `kokoro_onnx` runs anywhere onnxruntime does and
        # pulls no torch at all, which matters in a venv whose torch is
        # pinned by the trainer. Either will do.
        try:
            import kokoro                                          # noqa: F401
            return True
        except Exception:                                          # noqa: BLE001
            pass
        try:
            import kokoro_onnx                                     # noqa: F401
            return all(kokoro_files())
        except Exception:                                          # noqa: BLE001
            return False
    if backend == "qwen":
        # find_spec rather than import: importing qwen_tts loads the whole
        # stack and prints a flash-attn banner, which is a lot of noise for a
        # yes/no question asked before every run.
        import importlib.util
        return all(importlib.util.find_spec(m) is not None
                   for m in ("qwen_tts", "soundfile"))
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


class _FixSpeedDtype:
    """Works around a real bug in kokoro-onnx 0.4.7.

    For the "newer export" models -- the ones with an `input_ids` input, which
    is what the published v1.0 weights are -- the library builds its feed with
    `speed` as int32, while the model declares `speed` as float. Every call
    dies on "Unexpected input data type. Actual: (tensor(int32)), expected:
    (tensor(float))".

    Patching site-packages would fix it until the next pip install, so the
    coercion is wrapped around the session instead: it lives in this repo,
    survives reinstalls, and becomes a no-op the day upstream fixes it."""

    def __init__(self, sess):
        self._sess = sess

    def run(self, outputs, feed, *args, **kwargs):
        import numpy as np
        want = {i.name: i.type for i in self._sess.get_inputs()}
        fixed = {k: (np.asarray(v, dtype=np.float32)
                     if want.get(k) == "tensor(float)" else v)
                 for k, v in feed.items()}
        return self._sess.run(outputs, fixed, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._sess, name)


# ------------------------------------------------------------------ kokoro --
def _speak_kokoro(blocks, out_dir, voice=None, speed=1.0, verbose=True):
    """One wav per block. One voice for the whole run, or one PER BLOCK.

    Both packages split a long block into chunks themselves and return them in
    order; they are joined here so a block is a single continuous file. The
    voice is a named embedding either way, so block 9 sounds like block 1 --
    which is the entire reason this is the default."""
    import numpy as np
    import soundfile as sf

    # The voice is a name, or a list of names the same length as the blocks --
    # one speaker per turn, which is what makes two people talking possible
    # without loading the model twice.
    per_block = voice if isinstance(voice, (list, tuple)) else None
    voice = (voice if isinstance(voice, str) else None) or KOKORO_DEFAULT
    try:
        from kokoro import KPipeline
        pipelines = {}

        def render(text, who):
            pipe = pipelines.get(who[0]) or pipelines.setdefault(
                who[0], KPipeline(lang_code=who[0]))
            chunks = [a for _gs, _ps, a in pipe(text, voice=who, speed=speed)]
            return (np.concatenate([np.asarray(c, dtype="float32") for c in chunks]),
                    SAMPLE_RATE) if chunks else (None, SAMPLE_RATE)
    except ImportError:
        from kokoro_onnx import Kokoro
        model, voices = kokoro_files()
        if not model:
            raise RuntimeError(
                "kokoro-onnx is installed but its weights are not. Fetch them once:\n"
                "  mkdir -p ~/.cache/kokoro-onnx && cd ~/.cache/kokoro-onnx && curl -sSL -O "
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
                "model-files-v1.1/kokoro-v1.0.onnx -O "
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
                "model-files-v1.1/voices-v1.0.bin")
        engine = Kokoro(model, voices)
        engine.sess = _FixSpeedDtype(engine.sess)

        def render(text, who):
            # lang follows the voice's own first letter: a = American, b =
            # British, so two hosts can be from different places.
            lang = "en-gb" if who.startswith("b") else "en-us"
            samples, rate = engine.create(text, voice=who, speed=speed, lang=lang)
            return np.asarray(samples, dtype="float32"), rate

    out = []
    for i, text in enumerate(blocks):
        audio, rate = render(text, per_block[i] if per_block else voice)
        if audio is None or not len(audio):
            continue
        path = os.path.join(out_dir, "block_%02d.wav" % i)
        sf.write(path, audio, rate)
        out.append(path)
        if verbose:
            print("    [%2d/%2d] %-10s %.1fs  %s"
                  % (i + 1, len(blocks), per_block[i] if per_block else voice,
                     len(audio) / float(rate), text[:46]))
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
    keep = [i for i, b in enumerate(blocks) if b and b.strip()]
    if isinstance(voice, (list, tuple)):
        voice = [voice[i] for i in keep]
    paths = SPEAKERS[backend]([blocks[i] for i in keep],
                              out_dir, voice=voice, speed=speed, verbose=verbose)
    return [(p, seconds(p)) for p in paths]
