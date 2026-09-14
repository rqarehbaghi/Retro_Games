# Afterstate correctness fixes — 2026-09-13

The trainer still learns a value network. The fixes address incorrect experience
and controls; they do not establish that long multi-state training now converges.

## Implemented

- Replay stores **observed settled boards**, using simulator-defined encoding
  and observed-reward interfaces. Predicted candidates are used for ranking,
  not treated as executed experience. Actual line rewards use the configured
  counter, not terminal animation pixels.
- Missing-candidate steps send neutral input. Pending placements survive spawn
  delays. A nonterminal replay item without a real successor is rejected.
- Death reward is recorded once on the last observed afterstate. Truncation and
  training-budget exhaustion do not add death rewards. A pending placement can
  finish settling at the collection boundary without starting a new action.
- Macro completion is monitored during rotation, movement, release, and drop.
  Type changes and configured row resets stop controls; a row reset also detects
  consecutive pieces of the same type. Spawn waiting no longer waits for a
  second row reset after a replacement piece has already appeared.
- **Additional measured bug:** vertical I-piece rotations use RAM-column offset
  2, not 3. `tools/measure_piece_origins.py` verified all six types and all four
  rotations against rendered cells. Only type 1 rotations 1/3 differ. The values
  and evidence are in `macro_config.piece_col_offsets` in games.json. Training
  and live-play planning use this configuration.
- Training counts placements separately from neutral waits. Wait-only steps
  do not repeatedly optimize the same replay or advance exploration decay.
- Game-over during board settling is propagated. Automatic level transitions
  mark a discontinuity instead of joining unrelated boards in replay.
- Logs include actual rewards (including terminal reward), prediction mismatch
  counts, waits, and starting-state names. Seeded environment resets also seed
  multi-state selection.

## Generic framework

New games still select their algorithm in games.json:

- Platformers such as Sonic use `algorithm: ppo`, `action_mode: button_stream`,
  pixel observations, and their configured actions/rewards/termination.
- Placement games can use `algorithm: afterstate` and a suitable simulator.
  The shared afterstate runner no longer requires every simulator to have piece
  shapes. Simulators implement `encode_observation` and `observed_reward`;
  environments support neutral `step(None)` when a decision is unavailable.
  Optional `afterstate_ready` and `afterstate_discontinuity` info fields prevent
  learning from transitional observations. Custom environments must implement
  these semantics if they expose delayed observations.

Grid-specific type counts, coordinate normalization, masking dimensions, line
tiers/counter, hole normalization, piece names, lock threshold, and measured
column offsets are configurable. Tetris retains 210 observation entries and
221 value-network inputs. Its reward weights were not retuned.

Sonic is not claimed to be configured or trained by this change: its integration,
ROM, appropriate start states, and verified game configuration are still needed.
The existing PPO path is preserved and has a configuration regression test.

## Validation

- 17 new regression tests pass, covering full runner transitions, missing
  candidates, delays, terminal rewards, truncation/budget boundaries, macro
  completion, origin offsets, a different board/type alphabet, and pixel PPO.
- The existing core suite passes; `git diff --check` passes.
- The measured-origin fix removed the nonterminal mismatches in a repeat of
  the 40-decision control trace that originally exposed the vertical-I error.
- A fresh **1,000-placement CUDA smoke test** used the owner's 32-state pool.
  It completed 60 episode loops with finite losses (last mean about 0.82),
  saved and reloaded a checkpoint, and ran visual placement checks.
  This is a plumbing/correctness test, not a convergence benchmark.
- Earlier intermediate validation outputs remain in
  `checkpoints/afterstate_fix_validation`; final smoke-test outputs are in
  `checkpoints/afterstate_fix_validation_v2`.

Reproduce core checks in the existing WSL venv:

```bash
python tests/test_afterstate.py
python tests/test_core.py
```

The original 32-state command remains valid. Use a **fresh save directory**,
for example `checkpoints/tetris_v32`, and train from scratch rather than resuming
weights learned from corrupted transitions. Old checkpoints remain loadable,
but loading them does not undo their previous training.

## Remaining questions

Straight-drop candidates do not model every path/reachability constraint under
gravity; observed replay now accounts for execution differences, but candidate
ranking can still propose unreachable placements on crowded boards. The value
representation still omits speed/preview context. The update remains sampled
TD policy evaluation with replay, not a greedy optimal-control backup. Neither
normalization nor reward shaping was changed speculatively. Long-run per-state
greedy evaluations are needed before claiming the original performance blocker
is resolved.
