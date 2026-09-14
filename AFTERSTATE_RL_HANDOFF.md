# Afterstate RL — handoff brief for a code reviewer

**Update 2026-09-13:** transition/macro correctness fixes and measured piece-origin
corrections are documented in `AFTERSTATE_RL_FIXES.md`. The snapshot below is
historical; long-run multi-state convergence remains to be evaluated.

**Purpose of this file:** a self-contained briefing for a fresh reviewer (human or
LLM) who has the repo but none of the prior conversation. It explains what the
project is, the specific goal, what has already been verified so you don't repeat
it, and the one problem that is currently blocking progress. Read this, then dig
into the files it points at. It is a snapshot as of commit `fac8335`.

---

## 1. What the repo is

Two independent halves share one repo:

- **The studio pipeline** (works, in daily use): play a retro game in an emulator,
  record a `.bk2` replay, and auto-generate captioned/narrated short videos.
  Entry point `studio.py`. **Not relevant to this brief** — don't touch it.
- **The training half** (the subject of this brief): a *generic* RL trainer,
  `train.py`, that can train an agent for any game defined in `games.json`.
  Everything game-shaped (buttons, states, episode-end, observation, reward) is
  read from `games.json`; the Python framework is meant to stay game-agnostic.

The game in question is **`TetrisTime-Nes-v0`** (an NES Tetris variant, 2-player).
The agent plays **player 2**.

## 2. The goal

Train a genuinely strong Tetris-playing bot using **afterstate value-network
reinforcement learning** (Sutton & Barto, Ch. 6), not a hand-coded heuristic.

Hard constraints the owner has repeatedly stated (please respect them):

1. **It must stay an AI / RL agent** — a *learned* value network choosing moves.
   Do not replace the learner with a fixed hand-tuned board-evaluation heuristic.
2. **Anything game-specific goes in `games.json`** — piece shapes, board size,
   reward weights, RAM addresses. If something game-specific is hard-coded in the
   framework, that's a bug to fix, not to extend.
3. **ROMs are copyrighted** — `integrations/**/rom.*` is git-ignored, never commit
   it. (The reviewer needs a legally-obtained ROM placed there to run anything.)

## 3. How the afterstate trainer works

Selected when `games.json` sets `"algorithm": "afterstate"`.

- **Observation** (`rl/env.py` `build_grid_observation`): the board is read from
  the rendered frame into a 20×10 binary grid (the falling piece is masked out
  using its RAM row/col), flattened to 200 cells, plus a 7-way piece-type one-hot
  and 3 piece coords = 210 floats.
- **Simulator** (`rl/simulators/grid_placement.py` `GridPlacementSimulator`): given
  the current board + piece type, it enumerates every legal placement
  (rotation × column), drops the piece, clears lines, and returns for each
  candidate: the resulting afterstate feature vector (221-dim: 200 board cells +
  per-column heights + neighbour height diffs + holes + max-height), an
  `immediate_reward`, and the action id. The piece **shapes come from
  `games.json`** (`afterstate.shapes`, keyed by this game's RAM `piece_type`
  1..6 — measured against the ROM; this is NOT the standard tetromino set and a
  wrong shape set was an early, now-fixed bug).
- **Value net** (`rl/afterstate.py` `AfterstateValueNet`): MLP, 256-wide, 4 layers,
  LayerNorm, outputs a scalar V(afterstate).
- **Action selection** (`AfterstateAgent.select_action`): epsilon-greedy over
  `immediate_reward + gamma * V(afterstate)`.
- **Learning** (`AfterstateAgent.update` + the loop in `train_afterstate`): TD(0)
  with a replay buffer (50k) and a target network synced every 500 steps; Huber
  loss; the transition `(prev_afterstate, reward, afterstate, done)` is pushed to
  the buffer, where `reward` is the chosen candidate's `immediate_reward`. A
  terminal transition with `terminal_penalty` is pushed on top-out.
- **Macro execution** (`rl/macro.py` `MacroPlacementWrapper`): translates the
  chosen (rotation, column) into the actual button taps + soft-drop, one whole
  placement per `env.step`.
- **Exploration**: epsilon anneals from `--ent-coef` to `--ent-coef-final` over
  `--explore-steps` steps (decoupled from `--timesteps`).

### The reward (current, in `games.json` → `TetrisTime-Nes-v0.training.afterstate`)

```
immediate_reward = ( tiered_line_bonus(lines) + (Phi(after) - Phi(before)) ) * reward_scale
Phi(board)       = -( hole_penalty*holes + height_penalty*aggregate_height + bump_penalty*bumpiness )
tiers            = line_scale * [0, 1, 3, 6, 12]   for 0..4 lines
weights          : line_scale 10, hole 4, height 0.4, bump 0.3, reward_scale 0.3, terminal_penalty -20
```

`Phi(after) - Phi(before)` is a **potential difference**, so it telescopes: a
hole/height/bumpiness increase is charged once when created and refunded once a
line clear removes it. `reward_scale 0.3` keeps the value targets small enough to
converge. (There is also an unused generic `survival_reward` knob, default 0.)

## 4. Key files

- `train.py` — CLI + dispatch. `--unittest` runs a visual placement check.
- `rl/afterstate.py` — value net, agent, replay buffer, `train_afterstate`,
  `unittest_afterstate`, and `_verify_simulator` (a self-check that the simulator's
  predicted afterstate matches the emulator before training).
- `rl/simulators/grid_placement.py` — the generic drop simulator + the reward.
- `rl/env.py` — `make_env`, `GenericRetroEnv`, `GridObservation`,
  `build_grid_observation`; per-episode multi-state draw in `reset()`.
- `rl/features.py` — reads the board from pixels (`grid_from_frame`, `mask_piece`),
  computes holes/filled/height/max_height/bumpiness.
- `rl/macro.py` — `MacroPlacementWrapper` (rotation/column → button frames).
- `games.json` — the `TetrisTime-Nes-v0` block: observation, `afterstate` config,
  measured piece shapes, RAM var map. Every number has an evidence note.

## 5. What has ALREADY been verified (do not re-investigate)

Each of these was checked empirically against the running emulator, not by
reading code:

- **Board perception is correct.** The code-read board matches the on-screen
  board cell-for-cell; code holes == env holes == visible holes; score/lines match
  the on-screen HUD to the digit.
- **Piece shapes are correct.** `_verify_simulator` reports 6/6 sampled placements
  match the emulator. (The standard-tetromino assumption was the first bug; fixed.)
- **Macro execution is correct.** Over 60 placements, ~87% were cell-perfect and
  predicted line-clears matched actual clears 7/7. (`--unittest` renders
  before/decision/contact/result screenshots with a MATCH/MISMATCH verdict; on a
  trained model mismatches should be rare — occasional mismatches on an untrained
  model are legitimate: the piece can't reach a buried column.)
- **The reward shape is sound on a single start state.** Trained from the *empty*
  board (`level0_2p`) for 80k steps, the agent reaches survival ~43 placements /
  ~8 lines per episode, holes falling to ~20-30; loss converges ~0.8-1.5. This is
  the healthy baseline.
- **`height` in the training log is AGGREGATE height** (sum of 10 column heights,
  max 200), not a single column — the log now prints `filled` (0..200) and
  `max_height` (0..20) instead to avoid confusion.

## 6. THE BLOCKER — multi-state training destabilises the value net

The owner trains from a *set* of start states for variety:

```
--state level0_2p,level8_2p,rs_01,...,rs_30   (32 states)
```

`level0_2p` is empty; `rs_01..rs_30` are saved **mid-game** positions of
increasing depth (rs_30 starts already at score 2466, 58 filled cells, 27 holes,
aggregate height 85). With this mix, training **diverges**:

- Loss climbs (e.g. ~1.0 → 2.0 by 18k, → ~3.5 by 39k) instead of settling.
- Survival falls over training, and the trained value net ends up playing **worse
  than the near-random early phase** — i.e. `immediate_reward + gamma*V` picks
  worse placements than `immediate_reward` alone once V is trained.

Measured A/B/C/D comparison (18k steps each, identical reward):

| states | AvgLen late | Loss trend |
|---|---|---|
| `level0_2p` only | ~30 | stable ~1.7 |
| all 32 (incl. deep) | ~16 | 1.0 → 2.0 rising |
| all 32 + LR 8e-5 | ~17 | 1.0 → 2.1 rising |
| 12 shallow states | ~22 | noisy ~2.0 |

**Interpretation (working hypothesis, may be wrong — this is what we want a fresh
pair of eyes on):** the deep mid-game states are near-unrecoverable, so their
short episodes flood the replay buffer with large-magnitude, high-variance
"doomed" targets; the value net can't fit them and the whole function degrades
(a deadly-triad-style instability). But note AvgLen is *confounded* — episodes
seeded from a deep well are short by construction, so AvgLen alone can't separate
"bad play" from "hard start"; the **rising loss within a single run** is the
cleaner divergence signal.

## 7. What we tried that did NOT fix it

- Lowering the learning rate (2.5e-4 → 8e-5): loss still rises. Not a fix.
- Curating to the 12 shallower states: helps survival, loss still elevated/noisy.
- (Earlier, on single-state) scaling the reward down cured an *earlier*
  divergence and is already in place at `reward_scale 0.3`.

## 8. Questions for the reviewer

1. Is the afterstate TD target in `AfterstateAgent.update` / the buffer push in
   `train_afterstate` correct — especially the `done`/terminal handling and which
   reward is attached to which transition? Could the terminal push double-count or
   mis-associate a reward at episode end (which fires often with deep states)?
2. Is there a genuine value-divergence cause in the multi-state setting beyond the
   working hypothesis — e.g. target/return scale, replay staleness across very
   different state distributions, or the potential reward interacting badly with
   frequent early termination?
3. Would **return normalisation** (e.g. PopArt / running-std target scaling) or
   value/target clipping be the right principled fix, and is anything about the
   current net/loss making it worse?
4. Anything in the observation, the reward telescoping, or the multi-state
   `reset()` draw (`rl/env.py`) that would make V learned from mixed states
   inconsistent?
5. Is the afterstate approach itself the right tool here, or is there a simpler
   change that makes multi-state training stable while keeping it a learned agent?

## 9. How to run / reproduce

```bash
# healthy baseline (stable):
python train.py --game TetrisTime-Nes-v0 --state level0_2p \
  --timesteps 300000 --explore-steps 120000 --device cuda \
  --save-dir checkpoints/base --save-every 50000

# reproduce the blocker (diverges):
python train.py --game TetrisTime-Nes-v0 \
  --state level0_2p,level8_2p,rs_01,rs_02,...,rs_30 \
  --timesteps 300000 --explore-steps 120000 --device cuda \
  --save-dir checkpoints/multi --save-every 50000

# visually verify decisions + execution of any checkpoint:
python train.py --game TetrisTime-Nes-v0 --state level0_2p --unittest \
  --resume checkpoints/base/ckpt_150000_steps.zip --unittest-samples 8
```

Notes: needs a legally-obtained ROM at `integrations/TetrisTime-Nes-v0/rom.nes`.
The `rs_01..rs_30` start states are generated by `tools/make_start_states.py` and
are currently **untracked in git** (local to the owner's machine).
