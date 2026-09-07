# Working on this repo

Two halves. **The studio pipeline works and is what gets used.** The training
half is on hold — the agent speedruns without collecting and cannot reliably
clear World 1-1, and the owner deferred it deliberately. Do not restart
training work unless asked.

## The one command

```bash
python studio.py --game SuperMarioBros3-Nes-v0
```

Play until the window closes. Everything after is automatic and lands in one
self-contained folder under `studio_out/`. The run prints TWO paths and
nothing else -- the folder, and `UPLOAD_BRIEF.md` inside it -- because
everything it used to dump on the terminal is in the folder:

| File | What |
|---|---|
| `*_16x9.mp4` | 1920×1080 with captions — YouTube |
| `*_9x16.mp4` | 1080×1920 with captions — TikTok / Reels / Shorts |
| `*_16x9_clean.mp4` | HD, no text, for thumbnails and re-edits |
| `*_narrated.mp4` | only with `--voice` — the captioned cuts with commentary over them |
| `*_source.mp4` | the raw capture everything is rendered from |
| `*.bk2` | the replay — the folder is self-contained because of this |
| `overlays.json` | every word and style rule, editable, re-renderable |
| `narration.wav` | only with `--voice` — the spoken track on its own |
| `UPLOAD_BRIEF.md` | paste into Claude and it walks the upload -- the copy, the rules, and Windows-side paths for the files to attach |
| `paste.txt` | the same copy as three upload forms, to retype by hand |
| `metadata.json` `captions.txt` `narration.txt` `events.csv` | |

Retry the writing without replaying — it renders **into the same folder**:

```bash
python studio.py --game <id> --from-mp4 ./studio_out/<folder>/<slug>_source.mp4
```

Rebuild just the brief for a folder from an earlier run:

```bash
python studio.py --brief ./studio_out/<folder>
```

Re-render after editing wording or style, in seconds, no emulator:

```bash
python restyle.py ./studio_out/<folder> --list
python restyle.py ./studio_out/<folder>
```

## Where things live

- `studio.py` — orchestrates: play → scan events → write → render → speak
- `play_engine.py` — unified pygame interactive gameplay engine (P1/P2 human, AI model, pads, cold boot)
- `render_bk2.py` — unified emulator replay renderer to MP4 with reward shape correction
- `writer.py` — every prompt and all three LLM backends
- `overlays.py` — all rendering; both studio and restyle go through `render_spec`
- `tts.py` — Qwen3-TTS speech and the ducking mux
- `restyle.py` — re-render from an edited `overlays.json`
- `games.json` — verified RAM addresses per game, with the evidence for each
- `train.py` — the deferred RL PPO training pipeline
- `tools/` — RAM discovery, pad testing, and inspection utilities (`probe_pad.py`, `find_game_vars.py`, `audit_ram.py`, etc.)

## Facts that cost real effort to establish

Re-deriving these wastes a session. Breaking them reintroduces bugs the owner
already reported.

**Captions are anchored by event INDEX, never by model timestamps.** The
prompt numbers the timeline `[0] [1] [2]`; the model returns
`{"event": 3, "text": "..."}` and `event_time()` computes the second. Asking a
model when something happened produced captions on the wrong moments — it only
has the numbers in the prompt and it approximates.

**There is no caption lead, and there must not be one.** Every value the event
scan reads changes on the frame the thing happens, `lives` included — it
decrements as Mario dies, not at the end of the death animation. A previous
version subtracted 0.5–3s per event kind on the opposite theory; it put every
caption about a second early and the owner reported it. `caption_offset` in
`studio.json` is the only timing knob and is 0.

**One RAM value, two different events — three cases, all fixed, all easy to
reintroduce:**

- The power byte `0x00ED` is cleared on a hit, on death, AND on level exit.
  `filter_power_noise` discards drops near a death or a clear, and near the end
  of the tape. The clear is now logged at the card grab, and the power byte is
  not wiped until the course finishes unloading ~6s later, so the clear's
  window there is wider (`CLEAR_OUTRO_WINDOW`) than a death's.
- **A course clear is READ from `0x00C4`, not inferred from position.** The
  byte is 0 all level and 255 for the ~5.5s a clear sequence runs. It was
  verified on FOUR recordings: it rose at the exact frame Mario touches the
  end-of-course card, 1.6-2.0s BEFORE the COURSE CLEAR banner is drawn — the
  earliest honest moment. `games.json` records the frame numbers. Inferring a
  clear from a position collapse was measurably WRONG: on those same
  recordings it called a pipe entry a clear and logged nothing for three
  genuine clears. `split_collapses` now uses the flag; the SMB3 ROM's victory
  routine at CPU `$8FE3` is what sets bytes like this.
- The old position heuristic is the FALLBACK only, for a game with no
  `course_clear` address. A collapse means a level ending or a pipe, and the
  level TIMER separates them: it only counts down during play and resets UP
  when a new level starts. `classify_collapses` does this, and it is guesswork
  — see above for how wrong it was on SMB3.
- A power drop to tier 0 is not the same as raccoon → big. `shrink` vs
  `powerdown`.

**Captions are a solid white plate, not a white drop shadow.** A white shadow
was tried and failed on this exact game: World 1-1's sky is near-white, the
shadow vanished into it, and black glyphs were left with nothing separating
them from the background. The plate (`box` on, `box_color` white, `borderw` 0)
is legible over any frame. Change it in `overlays.py` `DEFAULT_STYLE`.

**The closing ASK lands on the moment the course is cleared**, held ~2.5s
longer than a joke (`CLOSING_BONUS`), and is written to be plain and literally
a question — not a pun. It used to appear five seconds before the tape ended,
which put it in the score tally or on the world map, seconds after anything
worth watching had stopped. `writer.closing_time` places it; a run with no
clear falls back to `duration - CLOSING_ROOM`.

**Text is sized off the SHORT edge of the frame**, not the width — for 9:16 the
short edge is the width, for 16:9 it is the height. Sizing off width made one
setting render 1.8× larger in landscape than in vertical.

**Text placement is aspect-aware.** In 9:16 the blurred bands leave room
outside the picture; in 16:9 the picture fills the height and captions must
overlay, above the console status bar (`HUD_TOP`), never through it.

**Press Start 2P is fixed-cell**, about one em per character, so caption LENGTH
decides rendered size far more than `size_div` does. Keep caption lines near 40
characters. `layout_text` wraps to two lines and verifies every word survived —
an earlier version silently dropped the end of a line that nearly fit.

**`fontfile=` paths must be escaped** for `\ : , [ ] ; '` — Ubuntu ships
variable fonts as, literally, `Ubuntu[wdth,wght].ttf`, and `[` `]` `,` are
filtergraph structure.

**The TTS speaker must stay FIXED.** Qwen3-TTS's VoiceDesign model invents a
new voice from the description on every call, so rendering line by line made
every sentence sound like a different person. Use `CustomVoice` with a named
preset (`voice_speaker`) — it keeps one speaker identity and still accepts a
per-line `instruct`, so the delivery varies while the person does not.

**The spoken commentary is OFF by default.** It is the least finished part of the pipeline and the owner has parked it; `--voice` turns it on. Without it no script is even requested, which saves one of the three model calls a run makes. Everything below still applies when it is on.

**The spoken track is one continuous monologue WITH the lines that name a
moment held back to it.** Both extremes failed: pinning every line to an event
sounded like captions read aloud, and pure end-to-end speech drifted out of
sync with the footage. Script entries carry an optional `event` index; anchored
lines wait for their moment while the trivia between them flows continuously.
`tts.space_clips` reports how much silence it had to insert waiting, which is
the signal that the script needs more said between the anchors.

**Ollama runs WITH reasoning on.** `think` was set false to stop Qwen3 putting
its chain of thought in front of the JSON, before `_strip_thinking` existed —
which meant a thinking model was being judged with thinking disabled. The
answer is read from `response`, falling back to `thinking` when the first is
empty.

**Speech is measured, never estimated, and never allowed to outlast the
footage.** Synthesised speech is reliably slower than words-per-minute
arithmetic, so a line that cannot finish before the video ends is dropped.
Room is reserved for the closing ask so it always lands.

**Duck the game audio with a plain gain, not a compressor.** A
`sidechaincompress` keyed off the narration never opened and the music stayed
at full volume. A fixed `volume=0.25` is measurable after the fact — verified
at −12.1 dB against an expected −12.0.

**Never hard-slice text to a character limit.** `text[:40]` put "Which level
should he try to actually fi" on screen. `trim_words` cuts at a space, and the
cap is generous because the renderer already wraps and shrinks to fit.

**Sanitise anything a model returns before it is spoken or drawn.** Backends
without constrained decoding hand back field names alongside values, and the
voice read "tone amused" out loud. `clean_spoken` strips it.

**A USB gamepad needs `--gamepad`, and needs to be attached to WSL first.**
`stable_retro.examples.interactive` — what `--players 1` uses by default —
is KEYBOARD ONLY, so a pad is silently ignored. `--gamepad` routes single
player through the pygame window instead, which reads both. Separately, WSL2
cannot see a USB device at all until `usbipd-win` attaches it from Windows.

## The three writer backends, and what they bill

`--writer` in `studio.json`. **A Claude Pro subscription and the Messages API
are separate products** — Pro does not include API credits, and a Console
organisation starts at a zero balance. A 400 "credit balance is too low" means
exactly that, not a broken key.

| Backend | Bills against | Constrained JSON |
|---|---|---|
| `claude-code` (default) | the Pro/Max **subscription**, via `claude -p` | no — asked for in the prompt |
| `claude` | prepaid API **credits** | yes |
| `ollama` | nothing, runs locally | yes |

Also: an exported `ANTHROPIC_API_KEY` silently overrides an OAuth profile AND
Claude Code's own login. If auth looks wrong, check `env | grep -i anthropic`
first.

## Tuning, in order of how often it is wanted

| Want | Change |
|---|---|
| Tone of everything | `VOICE` in `writer.py` — one system prompt, all three artifacts |
| Caption style specifically | the worked examples in `captions()` — models copy samples harder than they follow adjectives |
| Colour, size, position | `style` in `studio.json`, or per-caption in a staged `overlays.json` |
| The voice's sound | `voice_describe` in `studio.json` — plain words, not a preset |
| Cost | `claude_effort` (`low`…`max`) before `claude_model` |
| One video's wording | edit `overlays.json`, run `restyle.py` |

## House rules

- **Verify against the artefact, not the code.** Nearly every bug here was
  found by rendering a frame and looking, or by measuring audio levels — not by
  reading. Render and check.
- **Never invent a RAM address or a timing constant.** `games.json` records the
  evidence for every address, and `KNOWN_BAD_ADDRESSES` in `train.py` records
  the ones that were tried and disproven. Several bugs came from numbers that
  were guessed and then asserted as fact.
- **A model failure must never cost a recording.** Availability of the writer,
  the fonts and the TTS is checked BEFORE a frame is played; the event scan and
  the speech are non-fatal and run after rendering.
- Model output is untrusted: cap caption length, drop out-of-range event
  indices, re-sort.

## Known open

- Training: the agent under-collects and cannot reliably clear World 1-1.
  Reward weights, sprite observation and a ? block detector are all in; the
  binding constraint is that PPO cannot reinforce behaviour its own rollouts
  never contain. Behaviour cloning from collection demos plus the current
  weights is the untried combination.
- SMB3 is alternating two-player, so `--players 2` records the AI doing
  nothing. Co-op and versus formats need a genuinely simultaneous game.
- `ACTION_TABLE` has no `RIGHT+B+A`, so the agent cannot make a running jump,
  and consecutive jump actions merge into one held press (no release frame).
- Uploading is manual by design: only TikTok supports post-privately-then-
  review with an unaudited client. See `studio.py --print-upload-plan`.
