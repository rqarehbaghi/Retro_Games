# Afterstate training review — 2026-09-13

Historical pre-fix findings. See `AFTERSTATE_RL_FIXES.md` for implemented changes,
the additional measured vertical-I origin bug, and current validation limits.

Reviewed CLAUDE.md, AFTERSTATE_RL_HANDOFF.md, train.py, rl/afterstate.py,
rl/env.py, rl/macro.py, the grid simulator, and the TetrisTime configuration.
No training implementation or game configuration was changed.

## Confirmed defects, ordered by relevance

### 1. Replay uses imagined outcomes even when execution disagrees (high)

`rl/afterstate.py:589-597` uses the candidate's predicted reward and feature
vector unconditionally. `next_obs` is used for the next decision, but never
to correct the afterstate being learned. The next reward is therefore computed
from an actual board and attached to a previous predicted board that may never
have existed. This is an inconsistent transition, not merely noisy exploration.

The handoff's 87% execution agreement does not make this safe: the other 13%
still enters replay. In this review's short random-action emulator audit,
level0 had 11 nonidentical predicted/observed boards out of 56 candidate
placements; level8 had 6/52. These counts include terminal observations, which
need separate treatment. Nonterminal discrepancies of 6–8 cells were observed
on level0. This is a short diagnostic sample, not a performance benchmark.

Fix direction: give simulators a generic interface for encoding observed
afterstates and scoring observed transitions. Use actual settled outcomes for
replay. Compute rewards from configured counters/features; do not infer line
clears from terminal animation pixels. Measure prediction disagreement and
handle invalid/transitional observations explicitly.

### 2. A replacement piece still receives the previous action's soft drop (high)

`rl/macro.py:198-218`: when the piece type changes during movement, only the
movement loop exits. The code then sends a release and enters soft-drop anyway.
The synthetic reproduction makes the type change on the first movement frame:

    emitted buttons: LEFT, neutral, DOWN, neutral

DOWN is applied to the replacement piece. Phase 1 also detects only type
changes, so consecutive pieces with the same type are not recognized there.
This breaks the assumption that one decision controls one placement, especially
when gravity locks a piece before a long movement plan finishes.

Fix direction: track placement completion across all macro phases with the
configured spawn/lock signals, stop controls once the selected piece locks,
and return after settling. Keep signals and thresholds in games.json.

### 3. No-candidate transitions erase the previous state and can lose death (high)

`rl/afterstate.py:595-597`: a missing candidate produces `afterstate_feat=None`.
The trainer pushes that as the successor and then overwrites the previous
afterstate with None. If nonterminal, replay sampling replaces None with an
all-zero feature vector (`:74`), and update bootstraps from it (`:174`). That
vector is not an observed successor. Subsequent transitions lose the chain.

If the no-candidate step ends the game, the final penalty push is skipped
because the previous feature has already been cleared. Synthetic reproduction:

    previous=[1], reward=0, next=None, done=True

No configured -6 terminal penalty is stored. The emulator audit encountered
4 and 8 no-candidate steps in the two 60-decision rollouts. It did not classify
each as a transient versus a board with no valid drop; those cases must be
distinguished. Action 0 currently executes a placement, not a neutral wait.

Fix direction: preserve pending credit through transitional frames and never
bootstrap from a fabricated successor. Distinguish terminal/no-legal-action
states from a temporarily unavailable decision using a generic simulator/env
contract.

### 4. Terminal reward does not propagate through the final predecessor (high)

For an ordinary terminal placement, the exact synthetic replay is:

    A -> B: reward=2, done=True
    B -> terminal: reward=-6, done=True

The first target is 2; the done mask prevents it from incorporating B's -6.
This is not double-counting the penalty. It disconnects it from the predecessor.
Furthermore, B may be only a predicted placement if the attempted action failed.

Fix direction: define precisely when an afterstate exists and when death is
observed. Either include the terminal outcome in the final predecessor target,
or retain a valid transition to a terminal-valued afterstate without masking
its value prematurely. Do not mix both conventions.

### 5. Cutoffs are labeled as deaths (medium)

`rl/afterstate.py:617` adds a terminal penalty whenever the inner loop exits,
including the total training budget. It also treats an env truncation as a
terminal transition via `done = terminated or truncated` at `:580`.
Both cases were reproduced with a two-step synthetic environment. A live final
board received -6 despite no game-over. The budget case occurs only at run end,
so it cannot by itself explain rising loss throughout training.

Fix direction: separate true termination, time-limit truncation, and collection
budget exhaustion. Bootstrap through valid truncations; add death only for a
configured terminal outcome.

## Learning-design concerns, not proven causes of this run's degradation

- Replay stores the epsilon-selected next placement. Update evaluates that
  stored successor, rather than recomputing a greedy backup from the current
  decision's candidates. This is a sampled policy-evaluation backup using stale
  behavior, despite the buffer being described as off-policy. It is not a
  Q-learning-style optimal-control backup. It can mix historical exploration
  policies; measure its effect after fixing transition correctness.
- The value representation contains only board geometry. It omits level/speed
  and upcoming-piece context. If these change future reachable moves or piece
  probabilities, identical feature vectors can have different returns. Mixed
  start states expose this more strongly. Any added context should be declared
  in games.json and encoded through a generic interface.
- The claimed telescoping shaping is `Phi(after)-Phi(before)` while gamma is
  0.99. It is not discount-consistent potential shaping. If policy-invariant
  shaping is intended, use `gamma*Phi(after)-Phi(before)` with consistent terminal
  boundary handling. Existing shaping may instead be an intentional objective;
  changing it changes the learned reward and needs an explicit design decision.
- Rising Huber loss alone does not prove value divergence. Neither mixed-state
  average episode length nor cumulative starting score is a clean evaluation.
  Use fixed per-state greedy evaluations, lines gained since reset, value/target
  magnitudes, and failure/mismatch rates.

## Generic-framework violations

The grid simulator hardcodes a seven-entry type encoding (`:165-167`), hole
normalization by 20 (`:114`), and line tiers [0,1,3,6,12] capped at four (`:218`).
GridObservation also hardcodes seven types and coordinate divisors. Macro
completion hardcodes a row threshold of 5 (`rl/macro.py:237`), and the visual
test contains Tetris piece names. These should be declarative parameters or
derived dimensions. The simulator's straight-drop model also does not test
whether the current falling piece can reach a destination before locking.

## Verification and next steps

Reproducer: `tools/audit_afterstate_review.py --live`. It uses the existing
trainer with synthetic env/agent replacements to inspect replay, a synthetic
macro trace, and two short real emulator rollouts. It does not train weights
or save checkpoints. Synthetic rewards are deliberately simple to expose
indexing and masking.

Executed in Ubuntu using `/home/reza/RetroGames/venv/bin/python` against the
Windows checkout mounted at `/mnt/g/GitHub/Retro_Games`. CUDA is available.
The WSL checkout has all 30 rs states; the Windows checkout does not. Compared
afterstate.py, macro.py, and games.json across both: identical ignoring CRLF.
The full 32-state training run was not repeated, and no claim of solved training
or a single dominant cause is justified yet.

Prioritize observed transition correctness, macro boundaries, and terminal/
no-candidate handling before reward retuning or normalization. Then compare
fixed per-state evaluations with a matched baseline. This keeps a learned value
network and retains game-specific rules in games.json.
