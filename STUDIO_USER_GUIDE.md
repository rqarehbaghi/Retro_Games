# Studio: playing, recording, writing and exporting

Commands below run in WSL from `~/RetroGames` with `source venv/bin/activate`.
Windows source is edited in `G:\GitHub\Retro_Games`; pull updates into WSL with
`git pull origin main`. ROMs, recordings, model weights and private voice assets
are not delivered by Git. Keep your WSL copies; do not delete output folders to
update code.

## Find and play a game

```bash
python studio.py --list-games       # Imported ROMs playable now
python studio.py --list-all-games   # All library definitions, including missing ROMs
```

One human, start at the title screen, record full-length videos without narration:

```bash
python studio.py --game TeenageMutantNinjaTurtles-Nes-v0 \
  --players 1 --boot-screen --short-seconds 0 \
  --no-voice --no-captions --level ""
```

For two humans, use `--players 2 --human`. Select the game's actual
two-player mode at the title screen. `--mode versus` is engine context, not a way
to give a single-player ROM a two-player mode. Some games alternate turns.

**Do not use only `--players 2` to request two humans.** Without `--human` (or legacy `--two-human`),
the second side is the AI/random-opponent path. Passing `--model PATH` invokes an
AI checkpoint; with one player it is AI-only. Single-player without a model is
human-controlled. A previously misleading AI message and missing-button crash
were fixed in `e2e943f`.

Click the game window to give it keyboard focus. P1 supports keyboard/gamepad;
P2 human uses another gamepad or numeric keypad. USB controllers must be attached
to WSL before pygame can see them. Use `python tools/probe_pad.py` for diagnostics.
Close the game window to stop recording and let post-processing finish.

`--boot-screen` uses a saved boot state when present, otherwise cold boot. If cold
boot stays gray, use `python tools/capture_boot_state.py --game GAME_ID`.
`--state NAME` selects a prepared state instead of starting from the title screen.

## What happens after playing

1. **Record:** `play_engine.py` records the emulator session as a .bk2 replay.
2. **Render:** `render_bk2.py` replays it into the source MP4, including game audio.
3. **Scan:** studio replays the recording to derive events from integration values
   and any declared `games.json` definitions. Coverage differs by game.
4. **Write:** `writer.py` writes titles, captions and upload copy. Spoken narration
   is requested only when `--voice` is enabled.
5. **Frame/render:** `overlays.py` produces 16:9 and 9:16 versions. Vertical framing
   fits the original image over a blurred background rather than stretching it.
6. **Speak/mix, if requested:** speech is generated, measured and mixed with reduced
   game audio. ElevenLabs can add the actual narration words as overlays.
7. **Stage:** files and upload copy remain in a dated `studio_out/` folder. Publishing
   is manual; no YouTube, Instagram or TikTok post is automatically uploaded.

## Four finished videos with narration

Follow [ELEVENLABS_SETUP.md](ELEVENLABS_SETUP.md) to enter your paid API key and Voice
ID locally, then use its new-recording command. The important combination is:

```text
--short-seconds 0 --narration-writer ensemble
--voice --voice-model elevenlabs --voice-id YOUR_VOICE_ID
--elevenlabs-allow-paid --overlay-videos
```

This produces clean landscape/portrait plus narrated-and-captioned landscape/portrait.
The raw source/replay are retained too. `--overlay-videos` is optional: without it,
ElevenLabs narration is mixed into video without added narration text.

## Writers versus voices

| Setting | Job |
|---|---|
| `--writer chatgpt` | Writes through the authenticated Codex CLI; pinned, no fallback |
| `--writer ollama --writer-model NAME` | Uses the selected installed local model |
| `--writer auto` | Tries ChatGPT then Ollama |
| `--narration-writer standard` | Uses the normal selected writer for narration |
| `--narration-writer ensemble` | Gemma 12B draft, independent Qwen3.8 27B draft, Gemma 31B master; other copy still uses `--writer` |
| `--voice-model chatterbox` | Existing local studio TTS, not the user's preferred quality |
| `--voice-model elevenlabs` | New explicit paid studio API speech path |

ChatGPT writing does not use an OpenAI API key. Its CLI must be installed and logged
in inside WSL. Ollama must be running with the selected models installed. Choosing
`--no-captions --no-voice` skips those particular writing jobs, but the normal studio
pipeline still writes title/upload copy and checks writer availability.

The ensemble prompt is generic: present-tense first-person gamer commentary with
personality, reactions and a connected story. It does not contain TMNT/Tetris rules.
Automatic studio ensemble mode currently supports human-only sessions, full-length
outputs, and recordings at least 15 seconds long. It uses a bounded number of speech
passages and leaves game audio between them; it is not continuous wall-to-wall talk.
All three local models run sequentially. This laptop has 16 GB VRAM, so the larger
models partially offload to CPU and can take minutes per script. Keep WSL awake.

## Emulator data and visual analysis: actual current status

`data.json` declares named memory variables such as score or health. These are raw
values, not a complete story; unit scales and variable meanings vary. `games.json`
contains additional verified per-game definitions. An imported ROM is not evidence
that every variable or event is accurate for that game.

Automatic studio ensemble mode now exports the .bk2's declared integration changes
through `tools/extract_replay_evidence.py` and combines them with the studio event
scan. It does not invent RAM mappings. Variable suffixes are not assumed to mean
separate players, and raw scores are not assumed to equal HUD scores.

`tools/analyze_gameplay.py` separately samples timestamped images using local vision
models. It saves frames, prompts, observations and uncertainty for review. **That
visual stage is not automatically fused into studio narration.** The first TMNT
test exposed high-confidence mistakes; using those as facts would produce bad scripts.
See [VIDEO_ANALYSIS.md](VIDEO_ANALYSIS.md).

For reviewed evidence and explicit time slots, use `tools/narrate_ensemble.py`, then
`studio.py --narrate-folder ... --narration-json ...`. The paid setup guide gives the
complete command for the owner's saved TMNT master. This avoids rewriting an approved
script. See also [LOCAL_NARRATION_ENSEMBLE.md](LOCAL_NARRATION_ENSEMBLE.md).

## Files to keep

| File | Purpose |
|---|---|
| `.bk2` | Replay; lets us recover integration data later |
| `*_source.mp4` | Native gameplay capture; source for future edits |
| `*_16x9_clean.mp4`, `*_9x16_clean.mp4` | Reusable exports with no added voice/text |
| `*_16x9_narrated.mp4`, `*_9x16_narrated.mp4` | ElevenLabs voice plus optional narration overlays |
| `overlays.json`, `events.csv` | Studio rendering specification and scanned event timeline |
| `ensemble_*/` | Automatic ensemble evidence, original raw integration values, prompts, drafts, master, metrics |
| `voice_elevenlabs/` | Paid audio cache for normal studio runs; retain to avoid regenerating identical speech |
| `narration.wav`, `narration.txt` | Spoken track and placed script for normal studio runs |
| `metadata.json`, `paste.txt`, `UPLOAD_BRIEF.md` | Upload copy and instructions; review before posting |

The separate `--narrate-folder` path writes into `elevenlabs_narrated/` by default;
its audio cache is named `voice_cache/`. It leaves the original studio exports intact.

## Useful flags and reruns

- `--short-seconds 0`: full-length vertical video. Nonzero selects highlights in the
  ordinary studio pipeline; current paid timed narration requires the full timeline.
- `--level ""`: removes the old configured World 1-1 label for unrelated games.
- `--out-dir PATH`: parent for newly dated studio folders.
- `--overlay-videos`: render text; cannot combine with `--no-captions`.
- `--no-voice`: disables speech even if studio.json enables it.
- `--writer-model NAME`: local writing model, not a speech voice.
- `--voice-id ID`: ElevenLabs identity. `--voice-reference WAV` is for local Chatterbox.

Reprocess an existing recording through the ordinary studio writer/render path:

```bash
python studio.py --game GAME_ID --from-mp4 studio_out/RUN/SLUG_source.mp4 \
  --short-seconds 0 --no-voice --level ""
```

This reuses the staged folder and can replace its generated exports/copy. It does
not create a fresh isolated edit folder automatically. To keep alternate versions,
copy the run folder first or use the paid path's `--voice-output` for voice variations.

Edit overlay wording/style without replaying:

```bash
python restyle.py studio_out/RUN --list
python restyle.py studio_out/RUN
```

Restyle operates on `overlays.json`; it does not regenerate paid speech or the new
spoken-text overlays. For those, rerun `--narrate-folder` with the same cache location.

## Other project tools

- `tools/progression/video.py`, `short.py`, `story.py`: trained-checkpoint showcase
  films, separate from human-gameplay studio. Their old ElevenLabs audition backend
  is not the paid studio backend.
- `tools/compare_local_writers.py`: identical-input local writing comparisons.
- `train.py`, `rl/`: training, not required to record human play.
- Voice cloning and local CosyVoice references are not required for ElevenLabs.

Code-path tests are not a guarantee of entertaining writing, accurate events for all
700+ games, natural speech, or successful paid API access. Review a short run first.

### Explicit player controls

- `--players 1 --human`: you play alone.
- `--players 2 --human`: two humans.
- `--players 1 --ai --model CHECKPOINT`: the AI plays alone.
- `--players 2 --ai --model CHECKPOINT`: human versus AI; `--player` selects the AI slot.

`--bot` is an alias for `--ai`. Explicit AI selection requires a checkpoint.
Older commands still work; `--two-human` is a compatibility alias.

## Automatic session resume

Studio saves the emulator position on normal window close or Ctrl+C, before
video processing. Saves live in `studio_out/save_states/<game>/<players>p/`,
ordered by UTC timestamp in their filenames, with a copy named
`session_end.state` beside the recording. The next play resumes the newest
save for that game and player count. No save means power-on. `--fresh` starts
from power-on without deleting old saves; `--boot-screen` is a compatibility
alias. An explicit `--state NAME` overrides automatic resume and cannot be
combined with either fresh-start flag. Abrupt process termination or power
loss cannot save. Use --fresh only for the first chapter, then omit it to
continue the next chapter in a new recording.
