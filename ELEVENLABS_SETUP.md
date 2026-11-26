# Paid ElevenLabs narration: setup and commands

This guide covers the paid **studio** speech path, not the older progression
free-tier audition backend. No ElevenLabs Python SDK is needed. Existing FFmpeg,
ffprobe and the normal project venv do the local audio/video work.

## 1. Set up your account and key

Choose a paid ElevenLabs subscription that includes API access and the voices you
want. In its Developers/API Keys area, create a key with access to Text to Speech,
reading your subscription, and reading voices. Set an account/key credit limit in
the provider dashboard. Do not enable automatic top-ups unless you want them.
The repository never purchases credits or changes your subscription.

Official references: [API keys](https://elevenlabs.io/docs/overview/administration/workspaces/api-keys),
[TTS API](https://elevenlabs.io/docs/api-reference/text-to-speech/convert),
[billing and licensing](https://elevenlabs.io/docs/overview/administration/billing).
Check the current paid plan and voice license before publishing commercially.

Inside WSL:

```bash
cd ~/RetroGames
git pull origin main
source venv/bin/activate
read -rsp 'ElevenLabs API key: ' ELEVENLABS_API_KEY; echo
export ELEVENLABS_API_KEY
export ELEVENLABS_VOICE_ID='YOUR_VOICE_ID'
```

The hidden prompt keeps the key out of shell history. It lasts for this shell and
its child processes; enter it again in a new terminal. Do not paste the key into
chat, studio.json, source files, committed .env files, or command arguments.
Use `unset ELEVENLABS_API_KEY` when finished. A Voice ID is not a secret; copy it
from the voice you selected in ElevenLabs, not the whole voice URL.
Your previously selected host-1 ID was `IJxm2BPzAbIAYPy6kY6Q`; its eligibility in
your future paid account has not been verified. This workflow has one narrator.

The default API model is `eleven_multilingual_v2`, the documented TTS endpoint
default. `--elevenlabs-model MODEL_ID` selects another supported model available
to your account, for example `eleven_v3`. Voice delivery is model/voice dependent;
we do not claim that any setting guarantees a natural performance.

## 2. Play from scratch and get four deliverables

All three local narration models are already installed on this machine. Keep them
installed for `--narration-writer ensemble`; run no competing GPU training/TTS job.
The normal writer still generates titles, descriptions and other copy. `auto`
tries the authenticated ChatGPT/Codex CLI, then local Ollama. The narration itself
uses the three local models regardless of the normal writer selection.

```bash
python studio.py \
  --game TeenageMutantNinjaTurtles-Nes-v0 \
  --players 1 --boot-screen \
  --short-seconds 0 \
  --writer auto \
  --narration-writer ensemble \
  --voice --voice-model elevenlabs \
  --voice-id "$ELEVENLABS_VOICE_ID" \
  --elevenlabs-allow-paid \
  --elevenlabs-max-chars 5000 \
  --overlay-videos \
  --level ""
```

Replace the game ID with one from `python studio.py --list-games`. For two human
players, use `--players 2 --human` and select the correct 2P mode
in the game. Do not pass a checkpoint `--model` for human-only play.

Close the game window when finished. Leave WSL/the laptop awake while it renders,
writes, generates speech and renders narration subtitles. The four deliverables:

| File suffix | Contents |
|---|---|
| `_16x9_clean.mp4` | Full-length YouTube framing, original game audio, no added text/voice |
| `_9x16_clean.mp4` | Full-length portrait framing, original game audio, no added text/voice |
| `_16x9_narrated.mp4` | Landscape, narration, reduced game audio and narration text |
| `_9x16_narrated.mp4` | Portrait, narration, reduced game audio and narration text |

The source MP4 and .bk2 are also retained as recording assets, not extra final cuts.
With ElevenLabs, `--overlay-videos` captions the **spoken narration**, in short
phrases, black on white plates. Phrase timing is proportional within each measured
audio clip, not word-accurate alignment. These are not the separate AI-written
studio commentary captions. Clean files remain untouched.

`--short-seconds 0` is required: both formats share the same full-length timeline.
`--elevenlabs-allow-paid` explicitly allows requests consuming account credits.
`--elevenlabs-max-chars` limits submitted script text, **not money or credits**;
pricing/model/voice multipliers belong to ElevenLabs. Set a provider-side credit
limit as well. No network request is retried automatically.

Automatic ensemble narration uses scanned studio events and raw replay integration
variables. It does not automatically run or validate visual analysis. An unfamiliar
game with sparse integration data can produce generic commentary. This is not a
claim of reliable visual understanding across every game. The more closely reviewed
TMNT test used manually reviewed evidence; automatic runs have a different evidence
quality. Inspect the script and final video before publishing.

## 3. Use an approved script on an existing recording

This bypasses gameplay, caption writing and the narration writers. For the reviewed
TMNT master already produced in your WSL workspace:

```bash
python studio.py \
  --narrate-folder studio_out/20261124-232016_teenage-mutant-ninja-turtles \
  --narration-json progression_out/tmnt_gamer_ensemble_02/master_revised.json \
  --voice --voice-model elevenlabs \
  --voice-id "$ELEVENLABS_VOICE_ID" \
  --elevenlabs-allow-paid \
  --elevenlabs-max-chars 5000 \
  --overlay-videos
```

Keep the ensemble's `manifest.json` beside the master JSON: it maps slot IDs to
source times. The timestamped TXT is for reading, not this importer. Alternatively,
create an approved JSON with explicit times:

```json
{"segments":[{"start":6,"end":15,"text":"Your spoken commentary here."}]}
```

For a small audition, use one such passage against your existing folder and a
separate `--voice-output progression_out/elevenlabs_audition_01`. This still creates
full-length video files, with just that short spoken passage; the individual cached
MP3 is available to listen to without opening the video.

The default output is the recording folder's `elevenlabs_narrated/` subfolder:
two narrated videos, `narration.wav`, `narration_timing.json`, `export.json`, and
`voice_cache/`. The original clean videos supply the other two deliverables.
Use `--voice-output PATH` to select another output directory. Repeating the same
command in the same output location reuses cached speech and replaces only these
generated exports. Use another directory to preserve alternate exports.

## 4. Timing, caching and recovery

- Text + Voice ID + model determine the speech cache key. Changed wording or voice
  costs a new generation; unchanged cached passages are reused.
- The MP3 is saved before local conversion. A local FFmpeg/render failure does not
  require buying the same speech again if you keep the cache and output directory.
- Speech is normalized per clip and measured. Approved time slots are kept fixed.
  An overlong passage stops the timed export instead of clipping or speeding it up.
  Shorten that passage in the JSON, then retry in the same output folder.
- Network timeouts can occur after a request has consumed credits. Inspect your
  provider history before retrying; the code does not silently resubmit it.
- Game audio is reduced to 25% amplitude. Narration is mixed at unity gain with a
  final peak limiter. Videos without game audio are supported.
- A speech failure during a new recording leaves the already-rendered clean videos
  and scripts intact. There is no silent switch to another paid provider.
- The older progression `--voice-model elevenlabs` dialogue backend retains its
  free-account guard. These paid studio commands use the paid functions in `tts.py` instead.

## Validation status

Tested locally: flags, missing-key/authorization checks, free-account rejection,
character limit, ensemble script import, speech-cache reuse, strict timing rejection,
and real FFmpeg exports in landscape/portrait with and without source audio.
API responses were mocked. No paid account/key has been supplied, no real ElevenLabs
generation has been made for these tests, and voice quality/account compatibility
remain to be verified with your first small audition.
