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
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# A FIXED speaker, which is the whole point of using CustomVoice here.
# VoiceDesign invents a new voice from the description on every call, so a
# per-line render came out sounding like a different person each sentence --
# reported as "nothing cohesive". CustomVoice keeps one speaker identity and
# still takes a per-line `instruct`, so the delivery can change while the
# person does not.
#
# Female presets: Vivian, Serena, Ono_Anna, Sohee.
# Male presets:   Ryan, Eric, Dylan, Aiden, Uncle_Fu.
DEFAULT_SPEAKER = "Vivian"

# How the fixed speaker should sound. This is an instruction ON TOP of the
# preset voice, not a description of a new one.
DEFAULT_VOICE = (
    "Commentate live, reacting to what you are watching rather than reading. "
    "Warm and conversational, with real variation in pitch and energy. Never "
    "flat, never a newsreader."
)

# Per-line tone hints were removed. They gave a different instruct on every
# call, and a model handing back the hint alongside the text meant the voice
# read "tone amused" out loud. One instruction for the whole run is both more
# consistent and has nothing to leak.


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
                voice=DEFAULT_VOICE, speaker=DEFAULT_SPEAKER,
                language="English", verbose=True):
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
        wavs, sr = model.generate_custom_voice(
            text=text, language=language, speaker=speaker, instruct=voice)
        sf.write(path, wavs[0], sr)
        out.append((item.get("anchor"), path, bool(item.get("closing"))))
        if verbose:
            print("    [%2d/%2d] %s" % (i + 1, len(lines), text[:64]))
    return out


MIN_GAP = 0.35          # breath between one line ending and the next starting


def wav_seconds(path):
    """Actual length of a rendered line.

    Read with soundfile, which this module already needs to WRITE the lines, so
    it cannot be missing when this is called. An earlier version shelled out to
    ffprobe and returned 0.0 when that failed -- which would have quietly
    disabled the overlap correction rather than reporting anything."""
    import soundfile as sf
    info = sf.info(path)
    return info.frames / float(info.samplerate)


def space_clips(clips, duration_s, min_gap=MIN_GAP, verbose=True):
    """Lay the lines out so they neither overlap nor outlast the footage.

    Two separate failures, both reported. Lines are spaced in the script by
    ESTIMATED reading time -- words over a words-per-minute figure -- which is
    never exact, so two of them talk over each other and both become
    unintelligible. And synthesised speech is reliably slower than the
    estimate, so the track ran past the end of the video.

    Pushing alone fixes the first and worsens the second, so a line that cannot
    finish before the footage does is DROPPED rather than pushed off the end.
    The closing ask is exempt: room is reserved for it up front and it is
    placed last, because it is the one line with a job beyond being funny."""
    lengths = {}
    for item in clips:
        path = item[1]
        try:
            lengths[path] = wav_seconds(path)
        except Exception:
            lengths[path] = 0.0

    closing = [c for c in clips if len(c) > 2 and c[2]]
    body = [c for c in clips if not (len(c) > 2 and c[2])]
    reserved = (lengths[closing[0][1]] + min_gap) if closing else 0.0
    ceiling = max(0.0, duration_s - reserved)

    # End to end in order, because this is one monologue -- but a line that
    # NAMES a moment waits for it. Speaking continuously with no regard for the
    # footage put the commentary out of sync with what was on screen; pinning
    # every line to an event made it sound like captions read aloud. Holding
    # back only the anchored lines keeps the speech continuous AND lands the
    # ones that matter while their moment is visible.
    placed, cursor, dropped, held = [], 0.6, 0, 0.0
    for item in body:
        anchor, path = item[0], item[1]
        start = cursor
        if anchor is not None and anchor > cursor:
            held += anchor - cursor
            start = anchor
        if start + lengths[path] > ceiling:
            dropped += 1
            continue
        placed.append((start, path))
        cursor = start + lengths[path] + min_gap

    if closing:
        path = closing[0][1]
        start = max(cursor, duration_s - lengths[path] - 0.3)
        placed.append((max(0.0, min(start, duration_s - lengths[path])), path))

    if verbose:
        if held > 2.0:
            print("  (voice: %.0fs of silence added waiting for moments the "
                  "script names -- ask for more to say between them)" % held)
        if dropped:
            print("  (voice: %d line%s dropped -- the spoken script was longer "
                  "than the footage)" % (dropped, "" if dropped == 1 else "s"))
    return placed


def build_track(clips, duration_s, out_path, sample_rate=SAMPLE_RATE):
    """One wav the length of the video, each clip starting at its timestamp.

    Built against a silent bed of the right length so the track lines up with
    the footage on its own, without relying on the mux to position anything."""
    if not clips:
        return None
    clips = space_clips(clips, duration_s)
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
    """Lay the narration over the video with the game audio turned down.

    A FIXED reduction, not a sidechain compressor. The compressor was keyed off
    the narration and did not open -- the game audio stayed at full volume,
    which was reported. With the commentary now continuous there is nothing for
    a compressor to do anyway: the music should sit under the whole thing, and
    a plain gain is something that can be measured afterwards rather than
    tuned by ear."""
    graph = (
        "[0:a]aresample=%d,volume=%.3f[game];"
        "[1:a]aresample=%d,volume=1.6[voice];"
        "[game][voice]amix=inputs=2:normalize=0:dropout_transition=0[out]"
        % (SAMPLE_RATE, duck_to, SAMPLE_RATE))
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", video, "-i", narration_wav,
         "-filter_complex", graph, "-map", "0:v", "-map", "[out]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", out_path],
        check=True, capture_output=True)
    return out_path
