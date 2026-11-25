#!/usr/bin/env python3
"""Speak the narration script locally with Chatterbox or Chatterbox Turbo.

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
TURBO_MODEL = "chatterbox-turbo"
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
    if model_name == TURBO_MODEL:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return ChatterboxTurboTTS.from_pretrained(device=device)
    if model_name not in (None, "", DEFAULT_MODEL):
        raise ValueError("unknown Chatterbox model %r" % model_name)
    from chatterbox.tts import ChatterboxTTS
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
        generate = {"audio_prompt_path": who}
        if model_name == TURBO_MODEL:
            # chatterbox-tts 0.1.7's loudness normalizer promotes an ordinary
            # PCM16 reference to float64, then Turbo's float32 mel filters
            # reject it. Our references are already clean and levelled; skip
            # that buggy optional normalization rather than rewriting them.
            generate["norm_loudness"] = False
        else:
            generate.update(exaggeration=float(exaggeration),
                            cfg_weight=float(cfg_weight))
        wav = model.generate(text, **generate)
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


# Paid ElevenLabs support (explicit opt-in; no stored credentials).
"""Opt-in ElevenLabs TTS and timed studio narration. No SDK or stored API keys."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request

elevenlabs_API = 'https://api.elevenlabs.io/v1'
elevenlabs_DEFAULT_MODEL = 'eleven_multilingual_v2'


def elevenlabs_check_config(voice_id, allow_paid, max_chars):
    if not allow_paid:
        raise ValueError('ElevenLabs requires --elevenlabs-allow-paid (requests consume account credits).')
    if not os.environ.get('ELEVENLABS_API_KEY', '').strip():
        raise ValueError('Set ELEVENLABS_API_KEY in this shell; never put it in a command argument or Git.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', voice_id or ''):
        raise ValueError('--voice-id must be an ElevenLabs Voice ID, not a URL or WAV path.')
    if max_chars <= 0:
        raise ValueError('--elevenlabs-max-chars must be positive.')


def elevenlabs_api(path, payload=None):
    req = urllib.request.Request(elevenlabs_API + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'xi-api-key': os.environ['ELEVENLABS_API_KEY'], 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError('ElevenLabs HTTP %d. Check key permissions, voice/model access and credits. '
                           'No automatic retry was made.' % exc.code) from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError('ElevenLabs connection failed. Check provider history before retrying: '
                           'a generation might already have consumed credits.') from None


def elevenlabs_preflight(voice_id, allow_paid, max_chars):
    elevenlabs_check_config(voice_id, allow_paid, max_chars)
    subscription = json.loads(elevenlabs_api('/user/subscription'))
    if subscription.get('tier') in (None, 'free'):
        raise ValueError('This studio path requires a paid ElevenLabs subscription; no speech requested.')
    elevenlabs_api('/voices/' + voice_id)
    return subscription


def elevenlabs_probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format',
                      '-show_streams', '-of', 'json', str(path)], text=True))


def elevenlabs_duration(path):
    return float(elevenlabs_probe(path)['format']['duration'])


def elevenlabs_synthesize(texts, out_dir, voice_id, model=elevenlabs_DEFAULT_MODEL, allow_paid=False, max_chars=5000):
    elevenlabs_check_config(voice_id, allow_paid, max_chars)
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError('Speech requires nonempty text passages.')
    if sum(map(len, texts)) > max_chars:
        raise ValueError('Script exceeds --elevenlabs-max-chars; no speech requested.')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for text in texts:
        payload = {'text': text, 'model_id': model}
        digest = hashlib.sha256(json.dumps({'voice': voice_id, 'payload': payload},
                                         sort_keys=True).encode()).hexdigest()
        jobs.append((payload, out / (digest + '.mp3')))
    missing = {str(path): payload for payload, path in jobs if not path.exists()}
    if missing:
        elevenlabs_preflight(voice_id, allow_paid, max_chars)
        print('ElevenLabs: %d uncached passages, %d text characters. Account credits will be used.'
              % (len(missing), sum(len(p['text']) for p in missing.values())), flush=True)
    clips = []
    for payload, mp3 in jobs:
        if not mp3.exists():
            audio = elevenlabs_api('/text-to-speech/' + voice_id + '?output_format=mp3_44100_128', payload)
            if not audio:
                raise ValueError('ElevenLabs returned empty audio; no retry made.')
            temporary = mp3.with_suffix('.mp3.tmp')
            temporary.write_bytes(audio)
            temporary.replace(mp3)
            mp3.with_suffix('.request.json').write_text(json.dumps(
                {'voice_id': voice_id, 'payload': payload}, indent=2), encoding='utf-8')
        # Preserve the paid response even if local conversion fails.
        wav = mp3.with_suffix('.wav')
        if not wav.exists():
            temporary = wav.with_suffix('.tmp.wav')
            subprocess.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', str(mp3),
                            '-af', 'loudnorm=I=-18:TP=-2:LRA=7', '-ar', '24000', '-ac', '1',
                            str(temporary)], check=True)
            temporary.replace(wav)
        elevenlabs_duration(wav)  # Fail rather than charge again for corrupt cached audio.
        clips.append(str(wav))
    return clips


def elevenlabs_speak_lines(lines, out_dir, **kwargs):
    paths = elevenlabs_synthesize([item['text'] for item in lines], out_dir, **kwargs)
    return [(item.get('anchor'), path, bool(item.get('closing'))) for item, path in zip(lines, paths)]


def elevenlabs_load_timed_script(path):
    path = Path(path)
    script = json.loads(path.read_text(encoding='utf-8'))
    segments = script.get('segments', [])
    if segments and 'slot' in segments[0]:
        manifest = json.loads((path.parent / 'manifest.json').read_text(encoding='utf-8'))
        slots = {s['id']: s for s in manifest['evidence']['segments']}
        segments = [dict(start=slots[s['slot']]['start'], end=slots[s['slot']]['end'], text=s['text'])
                    for s in segments]
    if not segments:
        raise ValueError('Narration JSON needs a nonempty segments list.')
    previous = 0.0
    for segment in segments:
        start, end = float(segment['start']), float(segment['end'])
        if not math.isfinite(start) or not math.isfinite(end) or start < previous or end <= start:
            raise ValueError('Narration slots must be finite, ordered, non-overlapping and nonnegative.')
        if not isinstance(segment.get('text'), str) or not segment['text'].strip():
            raise ValueError('Every slot needs spoken text.')
        segment.update(start=start, end=end)
        previous = end
    return segments


def elevenlabs_timed_track(segments, clips, length, out):
    if len(segments) != len(clips):
        raise ValueError('One audio clip is required per time slot.')
    measured = []
    for segment, clip in zip(segments, clips):
        seconds = elevenlabs_duration(clip)
        if segment['end'] > length + 0.05 or seconds > segment['end'] - segment['start']:
            raise ValueError('Speech does not fit slot at %.2fs (%.2fs audio). '
                             'Shorten that passage; cached audio is preserved.' % (segment['start'], seconds))
        measured.append(dict(segment, spoken_end=segment['start'] + seconds, audio=clip))
    inputs, filters, labels = [], [], ['[0:a]']
    for i, item in enumerate(measured, 1):
        inputs += ['-i', item['audio']]
        filters.append('[%d:a]adelay=%d:all=1[a%d]' % (i, round(item['start'] * 1000), i))
        labels.append('[a%d]' % i)
    filters.append(''.join(labels) + 'amix=inputs=%d:normalize=0:dropout_transition=0[out]' % len(labels))
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-f', 'lavfi', '-t', str(length),
                    '-i', 'anullsrc=r=24000:cl=mono', *inputs, '-filter_complex', ';'.join(filters),
                    '-map', '[out]', '-t', str(length), str(out)], check=True)
    return measured


def elevenlabs_mux(video, track, out, measured=None, style=None):
    """Preserve full-length framing; optionally caption speech, duck and limit audio."""
    from overlays import DEFAULT_FONT, draw, layout_text
    info = elevenlabs_probe(video)
    stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
    width, height = stream['width'], stream['height']
    graph = []
    if any(s['codec_type'] == 'audio' for s in info['streams']):
        graph += ['[0:a]volume=0.25[game]', '[1:a]anull[voice]',
                  '[game][voice]amix=inputs=2:normalize=0:dropout_transition=0,alimiter=limit=0.95:level=false[aout]']
    else:
        graph += ['[1:a]alimiter=limit=0.95:level=false[aout]']
    output_video, encoding = '0:v:0', ['-c:v', 'copy']
    if measured:
        style = style or {}
        font = style.get('font', DEFAULT_FONT)
        if not Path(font).is_file():
            font = DEFAULT_FONT
        label = '[0:v]'
        n = 0
        for item in measured:
            words = item['text'].split()
            # Phrase timing is proportional, not word-level speech alignment.
            for begin in range(0, len(words), 8):
                phrase = words[begin:begin + 8]
                span = item['spoken_end'] - item['start']
                at = item['start'] + span * begin / len(words)
                until = item['start'] + span * min(begin + 8, len(words)) / len(words)
                size, wrapped = layout_text(' '.join(phrase), font, max(14, int(min(width, height) / 25)),
                                            int(width * 0.88), ratio=style.get('char_width_ratio', 0.55))
                if ' '.join(wrapped).split() != phrase:
                    raise ValueError('Caption does not fit; refusing to drop words.')
                top = int(height * (0.79 if height > width else 0.68))
                for row, line in enumerate(wrapped):
                    new_label = '[v%d]' % n
                    graph.append(draw(label, new_label, line, font, size,
                                      top + row * int(size * 1.35), 'black', 'white',
                                      border_w=0, box=True, box_color='white', box_pad=8,
                                      enable='gte(t,%.4f)*lt(t,%.4f)' % (at, until)))
                    label, n = new_label, n + 1
        output_video, encoding = label, ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18']
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', str(video), '-i', str(track),
                    '-filter_complex', ';'.join(graph), '-map', output_video, '-map', '[aout]',
                    *encoding, '-c:a', 'aac', '-b:a', '192k', '-t', str(float(info['format']['duration'])),
                    '-movflags', '+faststart', str(out)], check=True)


def elevenlabs_narrate_folder(folder, script, out_dir, voice_id, model, allow_paid, max_chars, overlays=False, style=None):
    folder, out = Path(folder), Path(out_dir)
    segments = elevenlabs_load_timed_script(script)
    sources = []
    for aspect in ('16x9', '9x16'):
        found = list(folder.glob('*_' + aspect + '_clean.mp4'))
        if len(found) != 1:
            raise ValueError('Expected one clean %s video in %s' % (aspect, folder))
        sources.append(found[0])
    lengths = [elevenlabs_duration(source) for source in sources]
    if abs(lengths[0] - lengths[1]) > 0.15:
        raise ValueError('Use full-length vertical video (--short-seconds 0); cut videos need separate timing.')
    spec_path = folder / 'overlays.json'
    if spec_path.exists() and json.loads(spec_path.read_text()).get('segments'):
        raise ValueError('This folder has edited highlight segments. Re-render full length before timed narration.')
    if segments[-1]['end'] > min(lengths) + 0.05:
        raise ValueError('Narration slots extend beyond video; no speech requested.')
    elevenlabs_check_config(voice_id, allow_paid, max_chars)
    out.mkdir(parents=True, exist_ok=True)
    clips = elevenlabs_synthesize([s['text'] for s in segments], out / 'voice_cache', voice_id, model, allow_paid, max_chars)
    track = out / 'narration.wav'
    measured = elevenlabs_timed_track(segments, clips, min(lengths), track)
    (out / 'narration_timing.json').write_text(json.dumps(measured, indent=2), encoding='utf-8')
    outputs = []
    for source in sources:
        target = out / (source.stem.replace('_clean', '') + '_narrated.mp4')
        if target.resolve() == source.resolve():
            raise ValueError('Output must not overwrite clean source.')
        elevenlabs_mux(source, track, target, measured if overlays else None, style)
        outputs.append(str(target))
    (out / 'export.json').write_text(json.dumps({'sources': [str(s.resolve()) for s in sources],
        'outputs': outputs, 'voice_id': voice_id, 'model': model,
        'caption_timing': 'proportional phrases, not word alignment' if overlays else 'none'}, indent=2))
    print('\n'.join(outputs))
    return outputs
