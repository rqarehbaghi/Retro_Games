# Checkpoint progression: evaluation suite and video

A standalone tool that shows how the TetrisTime afterstate agent improves over
training, and an evaluation behind it whose claims hold up. Nothing here
changes training, the studio pipeline, or live play.

Two artefacts:

- **The evidence.** Every `tetris_v54` checkpoint played over a fixed suite of
  one-player start states, with all raw per-trajectory results published.
- **The illustration.** A 3x2 video grid where three checkpoints play two of
  those states side by side. It is a picture of the evidence, not the evidence.

## 1. Rules this follows

These exist because the obvious version of this video would be dishonest.

1. **Nothing about the displayed states may depend on checkpoint results.**
   The suite is generated and frozen first; the two displayed states are chosen
   from the ZERO-VALUE CONTROL's results alone, by a rule written down here
   before anything runs.
2. **The video is illustrative; the sweep is the claim.** Single trajectories
   vary enormously in this game -- one checkpoint measured 13 to 817 lines
   across start states -- so no single game proves anything.
3. **No thresholds.** Report median, IQR, min, max, how many trajectories hit
   the placement cap, and the full per-state results. The old "100+ lines"
   figure is RETIRED: it was a diagnostic goal from when the agent cleared a
   handful of lines, not an acceptance threshold, and it must not appear as a
   target, threshold or benchmark.
4. **Equal opportunity.** Every trajectory gets the same PLACEMENT budget, not
   the same time budget: higher levels drop faster and animations cost frames,
   so a time cap would hand different checkpoints different numbers of moves.
5. **One evaluator, one configuration, for all 40 checkpoints.** Frozen before
   the sweep and recorded in the manifest.

## 2. Scope

- Checkpoints: `checkpoints/tetris_v54`, 40 of them, every 5,000 steps from
  5k to 200k. `final.zip` is EXCLUDED when it duplicates the last numbered
  checkpoint -- verified: its SHA-256 equals `ckpt_200000_steps.zip`.
- Plus the **zero-value control**: the same code with an untrained value head.
  It is not a random policy -- candidates are still ranked by immediate reward
  -- so it is labelled `Zero-value control -- untrained value head;
  immediate-reward guidance only`, never "no AI" or "random".
- Game: `TetrisTime-Nes-v0`, ONE player, the agent playing alone.

## 3. The state suite

32 one-player states, generated before any checkpoint is evaluated.

- **One procedure, one variable.** Same mode, same level, empty board, zero
  counters, same player. Only the RNG / piece sequence differs.
- **No trained checkpoint may be used to generate them.** The menu is driven by
  a fixed input script taken from a verified 1-player recording, and the only
  thing varied is idle timing at the menu, which is what seeds the piece
  sequence. Nothing about the agent enters the suite.
- **Independence is verified by behaviour, not by file hash.** Two states can
  hash differently and still play the same pieces. Each state's first 20-50
  piece types are recorded, and states whose piece prefix duplicates another's
  are rejected and regenerated.
- **The manifest is immutable and hashed.** `suite_manifest.json` holds every
  state's hash and piece prefix, the generation command, the ROM hash, the
  resolved training configuration hash, the code commit, the player setup and
  the placement cap. It is written and hashed BEFORE checkpoint evaluation.

## 4. Evaluation

- Deterministic greedy (epsilon 0), one trajectory per (checkpoint, state).
- **Placement cap: 500** (provisional). Justification is fixed compute and
  equal opportunity across checkpoints -- nothing to do with any line target.
  A 10-trajectory timing probe runs first; if a large share of late-checkpoint
  trajectories reach the cap, the sweep PAUSES and the cap is reconsidered,
  because a cap that binds often censors exactly the checkpoints being praised.
- Recorded per trajectory: placements, **delta lines from the state's initial
  value** (a restored state may not start at zero, so the raw counter is not
  the metric), final holes and height, and the end reason, which is one of
  `game_over` or `placement_cap`. The two are never merged.
- The live FPS cap is explicitly disabled and the achieved rate asserted;
  a dummy video driver does not imply uncapped stepping.
- Every checkpoint's SHA-256 is printed and recorded.
- Full sweep: 40 checkpoints + control over 32 states = 1,312 trajectories,
  run with bounded CPU parallelism.

### Caching

A cache entry is keyed on the SHA-256 of: checkpoint, state file, resolved
configuration, code commit, ROM, player setup, deterministic mode, and the
placement cap. Each pair runs in an isolated directory and its result is
written atomically, so an interrupted job can never look complete.

## 5. Choosing the two displayed states

Using the ZERO-VALUE CONTROL's results only, never a checkpoint's:

1. Rank all 32 states by the control's placements, ascending (harder = fewer).
2. **Median state** = rank 16 of 32 (upper median).
3. **Hard state** = the 10th percentile by that ranking, i.e. rank 4 of 32.
4. Ties break on the state's name, ascending, so the choice is reproducible.

The rule is stated on the end card and in the published results.

## 6. The video

- 1920x1080 exactly. 3x2 grid: columns are control / 100k / 200k; rows are the
  median state and the hard state. **100k is the temporal midpoint, fixed
  here in advance**, not the best-looking middle checkpoint.
- Panels show `DELTA LINES` and placements, plus `GAME OVER` or
  `PLACEMENT CAP REACHED` when they stop. Ended panels freeze on their final
  frame rather than going black.
- Uniform 3x playback speed, labelled once on screen.
- No game audio: six simultaneous tracks are noise.
- Labels use the house style -- black text on an opaque white plate, the same
  as captions elsewhere.
- An end card reports, over all 40 checkpoints and all 32 states: median, IQR,
  min, max, the number of capped trajectories, the state count and the
  placement cap, plus the selection rule. Raw per-checkpoint points stay
  visible; any smoothing is drawn over them, never instead of them.
- Full results are published as machine-readable JSON/CSV beside the video.

## 7. Stages

1. Suite generation + piece-prefix verification + hashed manifest.
2. Evaluator with caching, the timing probe, then the full sweep.
3. Grid renderer and end card.

Each stage lands as its own commit, with tests for manifest determinism, cache
invalidation, state selection and tie-breaking, `game_over` versus
`placement_cap` labelling, checkpoint deduplication, and output layout. Artefact
verification is a rendered grid still and a frozen ended/capped panel, inspected
before any full render -- per the house rule, verify against the artefact.

## 8. Known risks

- **The cap may bind for strong checkpoints.** Handled by the probe and the
  pause rule above.
- **A 6-panel grid may still be too small to read.** The still is inspected
  before the full render; the fallback is fewer panels.
- **Suite generation may fail to produce 32 distinct piece sequences** if menu
  timing does not perturb the RNG enough. Then the generator must vary a
  different checkpoint-independent input, and the piece-prefix check is what
  catches it.
- **3x speed** is a deliberate choice for a visualisation. It is not applied to
  match footage anywhere else in the project.
