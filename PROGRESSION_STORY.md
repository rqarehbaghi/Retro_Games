# Cached footage: a simpler narrated package

`tools/progression/story.py` is the new **cached-footage edit** entry point.
It does not play an emulator, train anything, or modify the input run/cache.
The older `video.py` and `short.py` commands remain available for their original
grid/elimination formats. Their writer briefs now also request plain language.

## Run in WSL

From the repository with its normal venv active, after `git pull origin main`:

```bash
python tools/progression/story.py \
  --run progression_out/runs/tetris_v54_uncapped_full \
  --out-dir progression_out/runs/tetris_v54_story \
  --count 2 \
  --seconds 120 \
  --short-seconds 60 \
  --writer chatgpt \
  --voice-model cosyvoice3 \
  --voice-name "voice_references/host1.wav,voice_references/host2.wav"
```

The input run needs its original `run.json`, cached MP4s, measurements, and
timelines. It must contain finished games for every version/start pair. Missing,
capped or truncated games are refused rather than presented as complete results.
No `--cap`, `--cached`, model-folder, or emulator-jobs flag is needed here:
this entry point ONLY reads already completed cached games.

Use `--count 3` if you want three fixed boards instead of two. Labels describe
relative practice among those selected, not an assertion that more practice
always produces better play. Exact version names remain in `selection.json`.

## What changes on screen

- Select the top two (or three) average results across the **same complete set**
  of cached starting positions. Then show the **first listed starting position
  for every selected version**, never a different flattering game per version.
  This is a selected showcase, not proof of monotonic learning/generalization.
- Fixed board positions, one shared source clock, clearly labelled jump cuts.
  Playback never exceeds 4x. Three sampled sections in each film; the long film
  has two adjacent host turns per section, with no long gap before a reply. Large
  Black Ops One captions retain the selected house font and opaque white plates.
- Long: two fixed host voices, straightforward back-and-forth about practice
  and choosing an opponent, no technical lecture. A **seven-second results
  card**, then moving replay footage under the complete next-match invitation.
- Short: one creator voice and readable captions, **no statistics card**.
  Finish on moving opponent footage with a crown and an explicit “stay tuned”
  next-match invitation. The opponent is chosen by the average over all starts,
  not necessarily whoever lasts longest in the example game on screen.
- Tied averages are disclosed. The card says “my next opponent”, not a universal
  performance claim. Full results and selection rationale remain in JSON.

The requested lengths include speech, results and closing. Speech is measured;
it is never sped up or cut off to fit. An overlong script fails with its files
preserved. The card cannot expand to 30 seconds.

## Speech safeguards and setup

CosyVoice still uses its isolated Python environment and fixed local host WAVs.
The trainer's Python dependencies do not change. Run `bash tools/install_cosyvoice.sh`
only if CosyVoice isn't already installed. New installs also cache the local
`openai/whisper-base.en` transcript checker. Existing installations download its
weights on the first new narration run (internet needed for that download only).
The voice recordings and generated speech are never sent to an API.

The worker checks each rendered turn on CPU against the intended words, retries
up to three times, and stops on repeated mismatches. It saves each attempt's
transcript, error score and duration beside the WAV as `.quality.json`. This
adds time and memory. It is a fallible **word-accuracy guard**, not a guarantee
of natural delivery: ASR can mishear legitimate speech and false-reject it.
The owner still needs to listen to the export.

Verbose emotional role instructions have been removed from synthesis. Short
delivery guidance, a conversational script, and the two fixed reference voices
now carry the performance. Per-turn loudness normalization remains enabled.
Optional exact-transcript `.txt` files beside each reference WAV still enable
CosyVoice's zero-shot path.

ChatGPT writing uses the existing authenticated Codex CLI subscription path;
there is no new billed API backend. Local Ollama is still selectable.

## Output and retry

- `narrated.mp4`: long two-host video.
- `short_9x16.mp4`: narrated vertical short.
- `long_script.json`, `short_script.json`: editable text and host assignments.
- `prompts/`: every narration/upload prompt given to the writer.
- `uploads.json`, four platform `.md` files, `UPLOAD_BRIEF.md`: YouTube long,
  YouTube Shorts, Instagram and TikTok titles, descriptions and tags.
- `selection.json`, `results.json`, `source_run.json`, `package.json`: selection,
  metrics and source/settings provenance. Source cache must stay available.
- `*_timeline.json`, `*_speech/`, `*_narration.wav`: measured edit/audio audit.

After a failure, edit the saved script if needed and rerun **the same command
with `--resume` appended**. It reuses scripts/upload copy, regenerates speech and
renders, and refuses changed inputs/settings. No previous source footage is
deleted. To intentionally change the source or duration, use a new output folder.

## Validation

```bash
python -m unittest tests.test_story tests.test_progression tests.test_writer_backends
```

Rendering is also checked locally with short synthetic-speech fixtures in both
orientations (including a three-board scene, a dead board, card and moving outro).
This does not substitute for an actual CosyVoice listening test on the owner's
WSL machine; the full export is deliberately left to the owner.
