#!/usr/bin/env python3
"""
Speak the narration script, locally, with Qwen3-TTS.

narration.txt is a timed script and nothing more -- there is no audio in the
pipeline without this. This renders each line, places it at its timestamp, and
ducks the game audio underneath so the commentary sits on top of the music
rather than fighting it.

    pip install -U qwen-tts soundfile
    python studio.py --game <id> --voice

WHY QWEN3-TTS. Apache 2.0 (so a monetised channel is fine), open weights, about
4GB, and it runs on any 8GB+ card -- comfortable on the same GPU already doing
the training. More to the point it does VOICE DESIGN: the voice is described in
words rather than chosen from a list, which suits a commentator persona that is
supposed to sound like one specific unimpressed person.

    Qwen3-TTS-12Hz-1.7B-VoiceDesign    voice from a description  <- default
    Qwen3-TTS-12Hz-1.7B-CustomVoice    named preset voices
    Qwen3-TTS-12Hz-1.7B-Base           cloning from reference audio
    Qwen3-TTS-12Hz-0.6B-*              lighter, faster, less expressive

TORCH CONFLICTS. qwen-tts pulls its own torch, and this project already has one
pinned by stable-baselines3. If installing it moves your torch and training
starts failing, put the TTS in its own venv and run --voice from there against
an already staged folder, rather than unpinning training.
"""
import os
import subprocess

SAMPLE_RATE = 24000
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# The commentator, described rather than picked. Keep it in step with VOICE in
# writer.py -- the words and the delivery should be the same person.
DEFAULT_VOICE = (
    "A dry, unimpressed male sports commentator in his forties. British. "
    "Measured pace, deadpan delivery, a little weary. He is amused but never "
    "excitable, and lands his punchlines flat rather than selling them."
)

# How far the game audio drops while a line is being spoken. Full silence loses
# the music, which is half of why anyone watches retro footage; leaving it at
# full volume makes the commentary unintelligible.
DUCK_TO = 0.25


def available():
    try:
        import qwen_tts        # noqa: F401
        import soundfile       # noqa: F401
        return True
    except ImportError:
        return False


def load(model_name=DEFAULT_MODEL, device="cuda:0"):
    """Load the model once; it is far too slow to load per line."""
    import torch
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(
        model_name, device_map=device, dtype=torch.bfloat16)


def speak_lines(lines, out_dir, model=None, model_name=DEFAULT_MODEL,
                voice=DEFAULT_VOICE, language="English", verbose=True):
    """Render each narration line to its own wav. Returns [(at, path)].

    Per line rather than one long read, because each line has a timestamp it
    has to land on -- a single render would drift out of sync with the run
    within about fifteen seconds."""
    import soundfile as sf
    model = model or load(model_name)
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for i, item in enumerate(lines):
        text = item["text"].strip()
        if not text:
            continue
        path = os.path.join(out_dir, "line_%03d.wav" % i)
        wavs, sr = model.generate_voice_design(
            text=text, language=language, instruct=voice)
        sf.write(path, wavs[0], sr)
        out.append((float(item["at"]), path))
        if verbose:
            print("    [%2d/%2d] %5.1fs  %s" % (i + 1, len(lines), item["at"],
                                                text[:52]))
    return out


def build_track(clips, duration_s, out_path, sample_rate=SAMPLE_RATE):
    """One wav the length of the video, each clip starting at its timestamp.

    Built against a silent bed of the right length so the track lines up with
    the footage on its own, without relying on the mux to position anything."""
    if not clips:
        return None
    inputs, chains, labels = [], [], []
    for i, (at, path) in enumerate(clips):
        inputs += ["-i", path]
        chains.append("[%d:a]aresample=%d,adelay=%d|%d[d%d]"
                      % (i + 1, sample_rate, int(at * 1000), int(at * 1000), i))
        labels.append("[d%d]" % i)
    # The silent bed is input 0 and MUST be in the mix. Without it amix ends
    # at the last clip, so a track for a 39 second video came out 27 seconds
    # long and stopped carrying the timeline it exists to carry.
    labels.insert(0, "[0:a]")
    graph = ";".join(chains) + ";" + "".join(labels) + \
        "amix=inputs=%d:normalize=0:dropout_transition=0[out]" % len(labels)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y",
         "-f", "lavfi", "-t", "%.3f" % duration_s,
         "-i", "anullsrc=channel_layout=mono:sample_rate=%d" % sample_rate,
         *inputs, "-filter_complex", graph, "-map", "[out]",
         "-t", "%.3f" % duration_s, out_path],
        check=True, capture_output=True)
    return out_path


def mux(video, narration_wav, out_path, duck_to=DUCK_TO):
    """Lay the narration over the video, ducking the game audio under it.

    sidechaincompress drives the duck from the narration itself, so the music
    drops only while a line is actually being spoken and comes back up in the
    gaps -- rather than sitting low for the whole video."""
    graph = (
        "[0:a]aresample=%d[game];"
        "[1:a]aresample=%d,asplit=2[voice][key];"
        "[game][key]sidechaincompress=threshold=0.02:ratio=%.1f:attack=20:"
        "release=400[ducked];"
        "[ducked][voice]amix=inputs=2:normalize=0:dropout_transition=0[out]"
        % (SAMPLE_RATE, SAMPLE_RATE, max(1.0, 1.0 / max(duck_to, 0.01))))
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", video, "-i", narration_wav,
         "-filter_complex", graph, "-map", "0:v", "-map", "[out]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", out_path],
        check=True, capture_output=True)
    return out_path
