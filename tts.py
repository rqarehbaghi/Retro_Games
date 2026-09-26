#!/usr/bin/env python3
"""Speak the narration script locally with Chatterbox.

narration.txt is a timed script and nothing more -- there is no audio in the
pipeline without this. This renders each line, places it at its timestamp, and
ducks the game audio underneath so the commentary sits on top of the music
rather than fighting it.

    pip install -U chatterbox-tts soundfile
    python studio.py --game <id> --voice

WHY CHATTERBOX. It is local, MIT licensed, and its zero-shot voice cloning
preserves the rhythm and identity of a real reference recording rather than
selecting a generic synthetic preset. A run may use the built-in voice, one
reference WAV throughout, or one reference per podcast turn. Only use audio
you own or have permission to clone.

Chatterbox and the trainer share the installed Torch. Stable-Baselines3 does
not use Transformers, so the media dependency does not enter the RL code path.
"""
import os
import subprocess

SAMPLE_RATE = 24000
DEFAULT_MODEL = "chatterbox"
DEFAULT_SPEAKER = ""                 # optional reference WAV
DEFAULT_EXAGGERATION = 0.5
DEFAULT_CFG_WEIGHT = 0.5


# How far the game audio drops while a line is being spoken. Full silence loses
# the music, which is half of why anyone watches retro footage; leaving it at
# full volume makes the commentary unintelligible.
DUCK_TO = 0.25


def available():
    try:
        import chatterbox      # noqa: F401
        import soundfile       # noqa: F401
        return True
    except Exception:          # noqa: BLE001 -- a broken Torch install is unavailable
        return False


def load(model_name=DEFAULT_MODEL, device=None):
    """Load the model once; it is far too slow to load per line."""
    import torch
    from chatterbox.tts import ChatterboxTTS
    if model_name not in (None, "", DEFAULT_MODEL):
        raise ValueError("unknown Chatterbox model %r" % model_name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return ChatterboxTTS.from_pretrained(device=device)


def validate_reference(reference):
    """Validate one optional clone source before model loading or gameplay."""
    if not reference:
        return
    if not os.path.isfile(reference):
        raise ValueError("Chatterbox reference WAV does not exist: %s" % reference)
    import soundfile as sf
    if sf.info(reference).duration < 5.0:
        raise ValueError("Chatterbox reference WAV must be at least 5 seconds: %s"
                         % reference)


def speak_lines(lines, out_dir, model=None, model_name=DEFAULT_MODEL,
                speaker=DEFAULT_SPEAKER, exaggeration=DEFAULT_EXAGGERATION,
                cfg_weight=DEFAULT_CFG_WEIGHT, verbose=True, **_ignored):
    """Render each narration line to its own wav. Returns [(at, path)].

    Per line rather than one long read, because each line has a timestamp it
    has to land on -- a single render would drift out of sync with the run
    within about fifteen seconds."""
    import numpy as np
    import soundfile as sf
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(speaker, (list, tuple)) and len(speaker) != len(lines):
        raise ValueError("one Chatterbox reference is required per narration block")
    references = list(speaker) if isinstance(speaker, (list, tuple)) else None
    for reference in references or ([speaker] if speaker else []):
        validate_reference(reference)
    model = model or load(model_name)
    out = []
    for i, item in enumerate(lines):
        text = item["text"].strip()
        if not text:
            continue
        who = references[i] if references is not None else (speaker or None)
        path = os.path.join(out_dir, "line_%03d.wav" % i)
        wav = model.generate(text, audio_prompt_path=who,
                             exaggeration=float(exaggeration),
                             cfg_weight=float(cfg_weight))
        if hasattr(wav, "detach"):
            audio = wav.detach().cpu().numpy()
        else:
            audio = np.asarray(wav)
        sf.write(path, np.asarray(audio).squeeze(), model.sr)
        out.append((item.get("anchor"), path, bool(item.get("closing"))))
        if verbose:
            print("    [%2d/%2d] %-18s %s"
                  % (i + 1, len(lines), os.path.basename(who) if who else "built-in",
                     text[:42]))
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


def build_track(clips, duration_s, out_path, sample_rate=SAMPLE_RATE, verbose=True):
    """Returns (path, placed) -- the track, and where each line actually landed.

    The placements come back because they are not knowable until the speech is
    rendered, and narration.txt should show the real times rather than what was
    asked for.

    One wav the length of the video, each clip starting at its timestamp.

    Built against a silent bed of the right length so the track lines up with
    the footage on its own, without relying on the mux to position anything."""
    if not clips:
        return None, []
    # verbose=False is for a caller that has ALREADY scheduled every line and
    # is handing the times in: space_clips then reports its held time as
    # "silence added waiting for moments the script names", which is true of
    # the studio pipeline and misleading for a deliberate spread.
    clips = space_clips(clips, duration_s, verbose=verbose)
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
    return out_path, clips


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
