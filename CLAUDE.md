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
self-contained folder under `studio_out/`:

| File | What |
|---|---|
| `*_16x9.mp4` | 1920×1080 with captions — YouTube |
| `*_9x16.mp4` | 1080×1920 with captions — TikTok / Reels / Shorts |
| `*_16x9_clean.mp4` | HD, no text, for thumbnails and re-edits |
| `*_narrated.mp4` | the two captioned cuts with spoken commentary over them |
| `*_source.mp4` | the raw capture everything is rendered from |
| `*.bk2` | the replay — the folder is self-contained because of this |
| `overlays.json` | every word and style rule, editable, re-renderable |
| `narration.wav` | the spoken track |
| `paste.txt` | the three upload forms, ready to copy |
| `metadata.json` `captions.txt` `narration.txt` `events.csv` | |

Retry the writing without replaying — it renders **into the same folder**:

```bash
python studio.py --game <id> --from-mp4 ./studio_out/<folder>/<slug>_source.mp4
```

Re-render after editing wording or style, in seconds, no emulator:

```bash
python restyle.py ./studio_out/<folder> --list
python restyle.py ./studio_out/<folder>
```

## Where things live

- `studio.py` — orchestrates: play → scan events → write → render → speak
- `writer.py` — every prompt and all three LLM backends
- `overlays.py` — all rendering; both studio and restyle go through `render_spec`
- `tts.py` — Qwen3-TTS speech and the ducking mux
- `restyle.py` — re-render from an edited `overlays.json`
- `list_fonts.py` — sample sheet of every installed font
- `games.json` — verified RAM addresses per game, with the evidence for each
- `train.py` and the `inspect_*` / `find_*` tools — the deferred training half

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
  of the tape, because a run that ENDS at the level end never logs a clear.
- A position collapse means either a level ending or going down a pipe. The
  level TIMER separates them: it only counts down during play and resets UP
  when a new level starts. `classify_collapses` does this.
- A power drop to tier 0 is not the same as raccoon → big. `shrink` vs
  `powerdown`.

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

**The spoken track is fitted to the footage, and that needs both halves.**
Lines are spaced in the script by estimated reading time, which is never exact,
so two of them talk over each other and both become unintelligible. And
synthesised speech is reliably slower than the estimate, so the track ran past
the end of the video. `space_clips` pushes overlapping lines back AND drops any
line that cannot finish before the footage does — pushing alone fixes the first
problem and worsens the second. Room is reserved for the closing ask so it
always lands.

**The commentary is mostly trivia, not play-by-play.** The viewer can see the
screen; what they cannot see is how the game was made, what got cut, what
everyone got stuck on. The prompt asks for facts and jokes about the specific
game and level, with reactions to events as the smaller part — and tells the
model to say only what it is confident is true, because a wrong fact about a
game this audience grew up with is worse than no fact.

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
