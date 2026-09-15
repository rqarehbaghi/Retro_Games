# v36 diagnosis — 2026-09-14

No trainer, reward settings, or checkpoints were changed for this diagnosis.
The current Windows and WSL copies of games.json, afterstate.py, env.py,
macro.py and grid_placement.py were compared and match ignoring line endings.
Historical checkpoint files do not embed their reward configuration, so the
return measurements below explicitly use the current configuration.

## Fixed-state, zero-exploration results

Each entry is one deterministic episode from the named exact save state.
The value-disabled policy ranks by the existing immediate reward only; it is
an ablation to diagnose the learned term, not a proposed replacement agent.

| Start | Policy | Placements | Lines gained | Discounted configured return |
|---|---|---:|---:|---:|
| level0_2p | Value disabled | 39 | 5 | 0.015 |
| level0_2p | v35 / 10k | 22 | 3 | 2.451 |
| level0_2p | v36 / 25k | 23 | 4 | 4.851 |
| level0_2p | v36 / final | 35 | 10 | 14.841 |
| rs_03 | Value disabled | 36 | 8 | 11.548 |
| rs_03 | v35 / 10k | 15 | 1 | -7.604 |
| rs_03 | v36 / 25k | 9 | 0 | -8.135 |
| rs_03 | v36 / final | 25 | 7 | 7.720 |
| rs_12 | Value disabled | 23 | 2 | -11.413 |
| rs_12 | v35 / 10k | 10 | 0 | -11.518 |
| rs_12 | v36 / 25k | 5 | 0 | -7.799 |
| rs_12 | v36 / final | 4 | 0 | -7.048 |

These are counterexamples and diagnostic trajectories, not a representative
average over the full training distribution or a convergence experiment.

## Confirmed objective mismatch

Current reward uses absolute board cost at scale 0.02, reward_scale 0.3,
survival_reward 0, and a scaled terminal penalty of -6. Every placement on an
inherited bad board incurs its absolute cost again. Dying stops those costs.
Thus a longer recovery attempt may score worse even if it clears more lines.

On rs_12, the final policy earns -7.048 for four placements and zero lines,
which is HIGHER than -11.413 for the 23-placement, two-line alternative. The
configured reward therefore ranks these two outcomes opposite to the stated
goal. This is not a proposed learning-rate explanation.

The final rs_12 policy had ZERO predicted/actual board mismatches. Its first
three afterstate value predictions versus realized continuation returns were:

| Predicted V | Realized continuation return |
|---:|---:|
| -6.96 | -6.78 |
| -6.41 | -6.40 |
| -6.03 | -6.00 |

Mean signed prediction error was -0.072. Low TD loss is consistent with fitting
the returns of a poor policy accurately. It does not establish good control.
This does not prove the final policy is globally optimal under the reward.

## Other observations

- The final policy DOES improve lines on the empty-board trajectory: 10 versus
  5 with value disabled and 3 at the resumed v35 checkpoint. The evidence does
  not support describing the whole run as simple numeric value divergence.
- On rs_03, final policy performance remains below the value-disabled baseline,
  even under configured return. Reward mismatch is not a complete explanation
  for every state; action-value generalization and the control backup remain
  candidates for controlled ablation.
- The replay update fits the stored successor chosen by historical behavior;
  it does not recompute the best successor action. Sampled TD policy evaluation
  is not inherently invalid, but its low loss cannot certify action rankings.
- On level0_2p, final policy had 7 nonterminal board mismatches in 35 placements.
  Only two of those mismatches were on line-clearing placements, and their
  predicted/observed line counts agreed. Thus the tested mismatches were not
  exclusively a line-clear timing problem. They are still worth measuring.
- `tools/eval_afterstate.py` is not a matched multi-state evaluator: its states
  argument controls the NUMBER of random resets, not an explicit state schedule.
  Different policies can get different state samples. Its untrained baseline
  also has unseeded random weights. Single-state greedy comparisons are less
  confounded; the new diagnostic uses a separate one-state environment for
  each named state and an exactly zero value baseline.

## Next experiment, before another long training run

Define the task score explicitly (e.g. lines cleared with survival as a stated
secondary goal), then test whether proposed rewards rank known recovery and
early-death trajectories correctly. Keep board guidance separate from the task
objective and handle terminal boundaries consistently. Changing an absolute
cost or terminal penalty without this test can move the same problem elsewhere.

After that, compare learned-control variants on a fixed per-state schedule with
exploration disabled. Report lines gained, placements, actual task score,
return calibration and execution mismatch rate. Use short checkpoint sweeps as
an acceptance gate, then scale training only after measured policy improvement.

Reproducer: tools/diagnose_value_policy.py accepts --game, --integration-dir,
repeated --state and repeated --checkpoint. It only loads models and runs
emulator episodes. Returns exclude the reward already earned when calculating
the continuation value of an afterstate. Terminal -6 is included once; neutral
waits do not advance the placement-discount clock.
