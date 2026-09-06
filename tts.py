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

# Delivery per line, appended to the description above. Qwen3-TTS designs the
# voice from words, so the emotion can change line by line -- a track read at
# one pitch throughout is what makes synthesised speech sound synthesised, and
# the difference between reading and reacting is the whole point.
#
# The writer picks one of these names per line; anything unrecognised falls
# back to the base voice, so a model inventing a tone cannot break the render.
TONES = {
    "deadpan": "Say this flatly, completely unimpressed.",
    "amused": "Say this with a laugh in your voice, like you find it funny.",
    "exasperated": "Say this with a groan, exasperated, as if this keeps happening.",
    "surprised": "Say this genuinely surprised, pitch rising, caught off guard.",
    "delighted": "Say this warmly and brightly, actually pleased.",
    "sarcastic": "Say this dripping with sarcasm.",
    "excited": "Say this fast and energetic, carried away by the moment.",
    "wistful": "Say this softly and fondly, remembering something.",
    "annoyed": "Say this sharply, genuinely irritated.",
}


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


def instruct_for(voice, tone):
    """The voice description plus this line's delivery."""
    hint = TONES.get((tone or "").strip().lower())
    return voice + " " + hint if hint else voice


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
        tone = item.get("tone", "")
        wavs, sr = model.generate_custom_voice(
            text=text, language=language, speaker=speaker,
            instruct=instruct_for(voice, tone))
        sf.write(path, wavs[0], sr)
        out.append((float(item["at"]), path, bool(item.get("closing"))))
        if verbose:
            print("    [%2d/%2d] %5.1fs  %-11s %s"
                  % (i + 1, len(lines), item["at"],
                     tone if tone in TONES else "-", text[:46]))
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

    placed, cursor, worst, dropped = [], 0.0, 0.0, 0
    for item in sorted(body):
        at, path = item[0], item[1]
        start = max(at, cursor)
        if start + lengths[path] > ceiling:
            dropped += 1
            continue
        worst = max(worst, start - at)
        placed.append((start, path))
        cursor = start + lengths[path] + min_gap

    if closing:
        path = closing[0][1]
        start = max(cursor, duration_s - lengths[path] - 0.3)
        placed.append((max(0.0, min(start, duration_s - lengths[path])), path))

    if verbose:
        if worst > 1.0:
            print("  (voice: lines pushed back by up to %.1fs so they do not "
                  "overlap)" % worst)
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
