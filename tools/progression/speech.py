"""Which voice speaks the narration, and how consistent it stays.

Speech rendered ONE SENTENCE PER CALL sounds like a machine reading a list:
pitch, pace and emphasis reset at each full stop, and the joins land on silence
rather than on breath. Two things fix that, and this module does both:

  1. A BLOCK, not a line. A whole paragraph goes to the model in one call, so
     the sentences inside it are spoken as one thought, with the model's own
     pauses between them.
  2. A model whose voice is a FIXED EMBEDDING rather than a fresh
     interpretation of a description each time.

Backends, with what each one costs:

    cosyvoice3  Fun-CosyVoice 3, Apache-2.0, local and free. The default for
              expressive narration and two-host podcasts. It runs in its own
              Python 3.10 environment and uses one authorized reference WAV
              per host, so it cannot disturb the trainer's Python environment.
    kokoro    Kokoro-82M, Apache-2.0, ~82M params, local, free. Named voices
              are fixed embeddings, so intonation does not drift between
              blocks. The recommended default for narration. Chunks long text
              itself and keeps one voice across the chunks.
              pip install "kokoro>=0.9.4" soundfile   (plus: apt install espeak-ng)
    chatterbox  Original Chatterbox, MIT, local and free. Uses its built-in
              voice or a reference WAV per host.
    chatterbox-turbo  Chatterbox Turbo, local and free. The recommended
              English podcast backend: conversational, faster, and one fixed
              cloned reference voice per host.
    piper     Piper, MIT, local, free, tiny and fast. Flat but utterly
              consistent -- the safe fallback on a machine with no GPU.
    elevenlabs  PAID, and it sends the script to a third party. Only used if
              explicitly asked for and ELEVENLABS_API_KEY is set.

Nothing here is installed by this repo. `available()` reports what is actually
importable on this machine so a run fails before the emulator does, not after.
"""
import os
import json
import re
import subprocess
import tempfile

SAMPLE_RATE = 24000
DEFAULT = "cosyvoice3"

# How fast each voice actually speaks. The script is sized from this, so a
# wrong figure is silence at the end of the film or lines cut off it.
#
# Kokoro is measured on rendered blocks on this machine (17 words in 5.72s and
# 13 in 4.78s). Chatterbox starts from the general estimate until its first
# complete narration is measured here.
WPM = {"cosyvoice3": 125, "kokoro": 170, "chatterbox": 140,
       "chatterbox-turbo": 140}
DEFAULT_WPM = 140
_CHATTERBOX_MODEL = None
_CHATTERBOX_TURBO_MODEL = None
# Resemble's documented expressive-speech starting point. The neutral 0.5/0.5
# pair made a two-host film sound like two people reading a list; lower CFG
# restores more deliberate pacing while the higher exaggeration adds variation.
CHATTERBOX_EXAGGERATION = 0.7
CHATTERBOX_CFG_WEIGHT = 0.3

# CosyVoice deliberately lives outside the trainer's Python 3.14 venv. These
# defaults are the locations written by tools/install_cosyvoice.sh; environment
# variables make the backend portable without introducing more CLI flags.
COSYVOICE_PYTHON = os.path.expanduser(os.environ.get(
    "COSYVOICE_PYTHON", "~/miniconda3/envs/cosyvoice/bin/python"))
COSYVOICE_HOME = os.path.expanduser(os.environ.get(
    "COSYVOICE_HOME", "~/CosyVoice"))
COSYVOICE_MODEL = os.path.expanduser(os.environ.get(
    "COSYVOICE_MODEL",
    os.path.join(COSYVOICE_HOME, "pretrained_models", "Fun-CosyVoice3-0.5B")))
COSYVOICE_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cosyvoice_runner.py")
COSYVOICE_STYLE = (
    "Speak at a normal conversational volume in a natural two-person podcast. "
    "React to the other host rather than reading prepared copy. Use varied "
    "pacing, natural pauses, and clear but subtle emotion. Do not sound like "
    "an announcer, audiobook narrator, or synthetic assistant."
    "<|endofprompt|>")
COSYVOICE_HOST_STYLE = {
    1: ("You are host one, the person who built the project. Sound personally "
        "invested, warm, confident, and occasionally self-deprecating."),
    2: ("You are host two, the curious co-host. Sound playfully sceptical, "
        "quick to react, and genuinely interested rather than scripted."),
}

# Every TTS block is generated independently. Without levelling, the model's
# phrase-dependent output gain becomes an audible jump at every turn. Normalize
# the *turns* (not only the final track) so the two hosts remain equally audible
# while dynamics within a sentence stay intact.
LOUDNESS_I = -18.0
LOUDNESS_TP = -2.0
LOUDNESS_LRA = 7.0


def words_per_minute(backend):
    return WPM.get(backend, DEFAULT_WPM)


def validate_voice(backend, voice=None, podcast=False):
    """Refuse bad voice inputs before an emulator run spends real time."""
    if backend not in ("cosyvoice3", "chatterbox", "chatterbox-turbo"):
        return
    references = [part.strip() for part in (voice or "").split(",") if part.strip()]
    if podcast and len(references) != 2:
        raise ValueError("%s podcast mode needs two reference WAVs in "
                         "--voice-name host1.wav,host2.wav" % backend)
    if not podcast and len(references) > 1:
        raise ValueError("%s monologue mode accepts one reference WAV" % backend)
    if backend in ("cosyvoice3", "chatterbox-turbo") and not references:
        raise ValueError("%s needs a reference WAV; podcast mode needs two" % backend)
    if not references:
        return                              # the built-in voice is valid
    import soundfile as sf
    for reference in references:
        if not os.path.isfile(reference):
            raise ValueError("%s reference WAV does not exist: %s"
                             % (backend, reference))
        info = sf.info(reference)
        if info.duration < 5.0:
            raise ValueError("%s reference WAV must be at least 5 seconds: %s"
                             % (backend, reference))

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
    if backend == "cosyvoice3":
        return (os.path.isfile(COSYVOICE_PYTHON)
                and os.access(COSYVOICE_PYTHON, os.X_OK)
                and os.path.isdir(COSYVOICE_HOME)
                and os.path.isfile(os.path.join(COSYVOICE_MODEL, "cosyvoice3.yaml"))
                and os.path.isfile(COSYVOICE_RUNNER))
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
    if backend in ("chatterbox", "chatterbox-turbo"):
        # find_spec rather than import: importing a TTS stack loads Torch and
        # model helpers just for a yes/no check before every run.
        import importlib.util
        modules = (("chatterbox.tts_turbo", "soundfile")
                   if backend == "chatterbox-turbo" else
                   ("chatterbox", "soundfile"))
        return all(importlib.util.find_spec(m) is not None for m in modules)
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


def _cosyvoice_instruction(text, host=1, responding=True):
    """A concise supported Instruct2 prompt, varied by speaker and turn type."""
    delivery = ("Deliver this as a direct response to the preceding speaker."
                if responding else
                "Open the discussion naturally and draw the other host in.")
    words = len(text.split())
    if "?" in text:
        delivery += " Let the question sound genuinely curious, not rhetorical."
    elif words <= 8:
        delivery += " Give the short reaction crisp, dry timing; do not whisper it."
    elif any(mark in text for mark in ("!", "—", " - ")):
        delivery += " Let the change in thought carry a little extra energy."
    else:
        delivery += " Emphasize the meaning, with restrained conversational emotion."
    return ("You are a helpful assistant. %s %s %s"
            % (COSYVOICE_STYLE.removesuffix("<|endofprompt|>"),
               COSYVOICE_HOST_STYLE.get(host, COSYVOICE_HOST_STYLE[2]), delivery)
            + "<|endofprompt|>")


def normalize_loudness(path, target_i=LOUDNESS_I, target_tp=LOUDNESS_TP,
                       target_lra=LOUDNESS_LRA, sample_rate=SAMPLE_RATE):
    """Two-pass EBU R128 normalization of one spoken turn, in place.

    The first pass measures the utterance; the second applies one linear gain
    whenever the true-peak constraint permits it. FFmpeg falls back to its
    dynamic mode only when a peak would otherwise clip. The source file is not
    replaced unless the complete normalized WAV was written successfully.
    """
    target = "I=%g:TP=%g:LRA=%g" % (target_i, target_tp, target_lra)
    measured = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", path,
         "-af", "loudnorm=%s:print_format=json" % target,
         "-f", "null", "-"], capture_output=True, text=True, check=True)
    matches = re.findall(r"\{[^{}]*\}", measured.stderr, flags=re.DOTALL)
    if not matches:
        raise RuntimeError("ffmpeg loudnorm returned no measurement for %s" % path)
    stats = json.loads(matches[-1])
    if stats.get("input_i") in (None, "-inf"):
        raise RuntimeError("cannot loudness-normalize silent speech block %s" % path)

    filt = (
        "loudnorm=%s:measured_I=%s:measured_TP=%s:measured_LRA=%s:"
        "measured_thresh=%s:offset=%s:linear=true:print_format=summary"
        % (target, stats["input_i"], stats["input_tp"], stats["input_lra"],
           stats["input_thresh"], stats["target_offset"]))
    handle, temp = tempfile.mkstemp(prefix=".loudnorm_", suffix=".wav",
                                    dir=os.path.dirname(os.path.abspath(path)))
    os.close(handle)
    os.unlink(temp)                 # ffmpeg should create it, not overwrite a stub
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
             "-i", path, "-af", filt, "-ar", str(sample_rate), "-ac", "1",
             "-c:a", "pcm_f32le", temp], check=True)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return path


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


# -------------------------------------------------------------- chatterbox --
def _speak_chatterbox(blocks, out_dir, voice=None, speed=1.0, verbose=True):
    """The studio pipeline's Chatterbox model, one continuous podcast turn per block."""
    import tts
    global _CHATTERBOX_MODEL
    lines = [{"text": t} for t in blocks]
    if _CHATTERBOX_MODEL is None:
        try:
            _CHATTERBOX_MODEL = tts.load()
        except Exception as exc:                                   # noqa: BLE001
            print("  [voice] GPU load failed (%s), falling back to CPU" % exc)
            _CHATTERBOX_MODEL = tts.load(device="cpu")
    clips = tts.speak_lines(lines, out_dir, model=_CHATTERBOX_MODEL,
                            speaker=voice or tts.DEFAULT_SPEAKER,
                            exaggeration=CHATTERBOX_EXAGGERATION,
                            cfg_weight=CHATTERBOX_CFG_WEIGHT,
                            verbose=verbose)
    return [path for _at, path, _closing in clips]


def _speak_chatterbox_turbo(blocks, out_dir, voice=None, speed=1.0, verbose=True):
    """Higher-quality English dialogue, with one stable clone per host."""
    import tts
    global _CHATTERBOX_TURBO_MODEL
    if not voice:
        raise RuntimeError("chatterbox-turbo needs a reference WAV per host")
    lines = [{"text": text} for text in blocks]
    if _CHATTERBOX_TURBO_MODEL is None:
        try:
            _CHATTERBOX_TURBO_MODEL = tts.load(tts.TURBO_MODEL)
        except Exception as exc:                                   # noqa: BLE001
            print("  [voice] Turbo GPU load failed (%s), falling back to CPU" % exc)
            _CHATTERBOX_TURBO_MODEL = tts.load(tts.TURBO_MODEL, device="cpu")
    clips = tts.speak_lines(lines, out_dir, model=_CHATTERBOX_TURBO_MODEL,
                            model_name=tts.TURBO_MODEL, speaker=voice,
                            verbose=verbose)
    return [path for _at, path, _closing in clips]


# --------------------------------------------------------------- CosyVoice --
def _speak_cosyvoice3(blocks, out_dir, voice=None, speed=1.0, verbose=True):
    """Render a complete batch in CosyVoice's isolated Python environment.

    Loading the 9 GB checkpoint once per sentence would be unusably slow, so a
    single subprocess owns the model for the whole body or card batch. The
    runner receives only text, authorized reference paths, and output paths.
    """
    references = list(voice) if isinstance(voice, (list, tuple)) else [voice] * len(blocks)
    if len(references) != len(blocks) or any(not item for item in references):
        raise RuntimeError("cosyvoice3 needs one reference WAV for every spoken block")
    references = [os.path.abspath(reference) for reference in references]
    identities = []
    hosts = []
    for reference in references:
        identity = os.path.normcase(os.path.realpath(reference))
        if identity not in identities:
            identities.append(identity)
        hosts.append(identities.index(identity) + 1)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    outputs = [os.path.join(out_dir, "block_%02d.wav" % i)
               for i in range(len(blocks))]
    job = {
        "model_dir": COSYVOICE_MODEL,
        "style": COSYVOICE_STYLE,
        "speed": speed,
        "blocks": [{"text": text, "reference": reference, "output": output,
                    "style": _cosyvoice_instruction(text, host, index > 0)}
                   for index, (text, reference, output, host)
                   in enumerate(zip(blocks, references, outputs, hosts))],
    }
    handle, job_path = tempfile.mkstemp(prefix="cosyvoice_job_", suffix=".json",
                                        dir=out_dir)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(job, stream, ensure_ascii=False, indent=2)
        subprocess.run([COSYVOICE_PYTHON, COSYVOICE_RUNNER, job_path],
                       cwd=COSYVOICE_HOME, check=True)
    finally:
        if os.path.exists(job_path):
            os.unlink(job_path)
    missing = [path for path in outputs if not os.path.isfile(path)]
    if missing:
        raise RuntimeError("CosyVoice returned without writing: %s" % missing[0])
    if verbose:
        for i, (text, reference) in enumerate(zip(blocks, references)):
            print("    [%2d/%2d] %-10s %s"
                  % (i + 1, len(blocks), os.path.basename(reference), text[:46]))
    return outputs


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


SPEAKERS = {"cosyvoice3": _speak_cosyvoice3,
            "kokoro": _speak_kokoro, "chatterbox": _speak_chatterbox,
            "chatterbox-turbo": _speak_chatterbox_turbo,
            "piper": _speak_piper, "elevenlabs": _speak_elevenlabs}


def speak(blocks, out_dir, backend=DEFAULT, voice=None, speed=1.0, verbose=True):
    """Render each block to its own wav. Returns [(path, seconds)]."""
    if backend not in SPEAKERS:
        raise ValueError("unknown voice backend %r; have %s"
                         % (backend, ", ".join(sorted(SPEAKERS))))
    if not available(backend):
        raise RuntimeError("voice backend %r is not installed here" % backend)
    os.makedirs(out_dir, exist_ok=True)
    # Enforce this at the final TTS boundary too. In particular, --respeak can
    # load a saved script produced before link sanitizing existed.
    import writer as text_writer
    blocks = [text_writer.clean_for_speech(block) for block in blocks]
    keep = [i for i, b in enumerate(blocks) if b and b.strip()]
    if isinstance(voice, (list, tuple)):
        voice = [voice[i] for i in keep]
    paths = SPEAKERS[backend]([blocks[i] for i in keep],
                              out_dir, voice=voice, speed=speed, verbose=verbose)
    for path in paths:
        normalize_loudness(path)
    return [(p, seconds(p)) for p in paths]
