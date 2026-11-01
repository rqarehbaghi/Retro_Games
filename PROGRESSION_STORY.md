# Cached footage: a simpler narrated package

## ElevenLabs free-tier voice setup (audition before full export)

No new pip dependency is needed. Select `--voice-model elevenlabs` explicitly;
the default is not silently changed and rejected local voices are not fallbacks.
The script text goes to ElevenLabs; your reference recordings are not uploaded.

1. In ElevenLabs, create an API key under Developers / API Keys. Enable speech
   generation and account/subscription read permissions for the quota check.
   Set a small credit limit on the key. Never paste the key into chat or Git.
2. Choose two voices available to your free account through the API and copy
   their Voice IDs. ElevenLabs explicitly excludes Voice Library voices from
   free-tier API access (https://elevenlabs.io/docs/eleven-creative/voices/voice-library).
   Website/library availability does not guarantee API access. Use permitted
   account voices, or audition through the website without enabling billing.
3. In your WSL terminal, after pulling the code:

```bash
read -rsp 'ElevenLabs API key: ' ELEVENLABS_API_KEY; echo
export ELEVENLABS_API_KEY
export HOST1_VOICE_ID='REPLACE_WITH_FIRST_VOICE_ID'
export HOST2_VOICE_ID='REPLACE_WITH_SECOND_VOICE_ID'
```

The key is only in this shell session; it is not saved to a repository file or
echoed into command history. The adapter defaults to the documented `eleven_v3`
dialogue model. `ELEVENLABS_MODEL` can select another model your account supports;
model access and voice quality must be confirmed with a real audition.

Generate one short two-host audition, consuming free credits:

```bash
python - <<'PY'
import os
from tools.progression import speech
speech.speak([
    "I trained this computer to play against me. Now I'm wondering if that was a good idea.",
    "You picked its best game to show us. Let's see how confident you are when it's your turn."
], 'voice_auditions/elevenlabs', backend='elevenlabs',
voice=[os.environ['HOST1_VOICE_ID'], os.environ['HOST2_VOICE_ID']])
print('Listen to voice_auditions/elevenlabs/dialogue_00.mp3')
PY
```

The two hosts are generated together via Text to Dialogue, not independent TTS
calls. Full dialogue audio, turn WAVs and the raw timestamp response stay in the
output folder. Repeating identical text/voices/model in that folder reuses the
response without another generation. Changed text or model consumes credits.
No automatic API retries occur. If a timeout occurs, inspect ElevenLabs history
before retrying: the server may have generated audio even without a local reply.

After approving the voices, generate the showcase with both formats:

```bash
python tools/progression/story.py \
  --run progression_out/runs/tetris_v54_uncapped_full \
  --edit-plan tools/progression/examples/tetris_v54_showcase.json \
  --out-dir progression_out/runs/tetris_v54_elevenlabs_test \
  --voice --voice-model elevenlabs \
  --voice-name "$HOST1_VOICE_ID,$HOST2_VOICE_ID"
```

The long uses the two authored hosts; the short uses host 1. Authored narration
is stored in the plan's `narration` arrays. Silent films remain intact; voiced
outputs are `long_narrated.mp4` and `short_narrated.mp4`. Exact speech duration
must fit each scene; overruns stop instead of clipping or accelerating voices.
Repeat the same command with `--resume` to reuse matching silent films and cached
speech after an interrupted run. Changing the edit plan requires a new folder.

Safety: the adapter refuses non-free accounts before new generation, checks
remaining character quota, and never enables usage-based billing or buys credits.
Actual credits depend on ElevenLabs model/voice pricing; the provider enforces
the free quota. Free-generated audio is not licensed for commercial use, and
noncommercial publication requires attribution under ElevenLabs' current terms.
Keep these as auditions until publication licensing is settled. Upgrading later
does not automatically relicense old free-generated audio.

API access, selected voices, naturalness and final narration timing cannot be
verified until the owner supplies credentials locally and auditions the output.
No account call or live synthesis was made while implementing this integration.

## Approved showcase edit (silent preview)

The current approved edit separates the formats: two strongest same-start games
in the short; all five checkpoints' best recorded games in the long film.
Do not use the old average-ranked / first-state narration command below to
reproduce this selection. That format and its local voices were rejected.

```bash
python tools/progression/story.py \
  --run progression_out/runs/tetris_v54_uncapped_full \
  --edit-plan tools/progression/examples/tetris_v54_showcase.json \
  --out-dir progression_out/runs/tetris_v54_showcase_v2
```

Use a new output folder if it already exists. This mode does not call a writer,
load TTS, launch an emulator, or purchase any service. It creates a 60-second
`short_9x16.mp4` and a 98-second `showcase_16x9.mp4`, both silent. It preserves
the edit plan and selected results beside the output. The JSON owns the specific
footage, captions and times; the Python rendering code remains game-agnostic.

Short: 50k / level0_1p (1,646 lines) and 200k / level0_1p (2,085).
Long: best recorded game for each of 10k, 20k, 50k, 100k, 200k, with starting
states disclosed. Different starts are a showcase, not a controlled comparison.
The shared source clock advances across explicit cuts; the recovery sequence is
continuous. Dynamic counters use cumulative timeline values, including thousands,
rather than the rolling game HUD. Both films show the actual final results.

The reader refuses inconsistent counters, incomplete recordings, backward cuts,
and short selections with different starting states. All footage in this example
plays at 1x. No filler time is added to reach a requested duration.

## Previous narrated format (retained, not the current recommendation)

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
The worker prefers complete cached WeText normalization assets before contacting
ModelScope, so a remote authentication failure cannot disable an already-installed
normalizer. Missing/incomplete caches still use the normal download path.
Older cached `en_tn_tagger.fst` / `en_tn_verbalizer.fst` files are also supported
via WeText's explicit local-file interface. This covers English; Chinese text
still requires the corresponding Chinese assets, never English substitution.
Progress messages distinguish model loading, speech generation, and word checking.
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
