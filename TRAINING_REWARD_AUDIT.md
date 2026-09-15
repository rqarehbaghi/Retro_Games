# Training/reward audit — 2026-09-15

Target: at least100 NEW lines per game, not100 total across16 games. Report
each state, mean, minimum, and number reaching100; total1600 alone is insufficient.
This target is NOT met. No claim that any single fix guarantees it.

## Confirmed problems and corrections

1. **Wrong reward contract.** Production afterstate reward had absolute board
   costs, nonlinear line tiers, and death penalty. Experimental potential runs
   also had survival reward. Neither was literally lines-only. games.json now
   pays exactly the positive delta of the configured line counter. Generic
   environment reward terms were aligned too, so there is no misleading legacy
   landing scaffold in reward descriptions. Score remains a diagnostic counter;
   train_afterstate ignores the environment reward and computes its own observed
   simulator reward. A score increase from dropping did NOT itself reward that
   learner. Tests explicitly increase score without clearing and require zero.

2. **Successful level-boundary placement discarded.** finalize_macro_step skips
   level animation and marks afterstate_discontinuity. The trainer previously
   cleared pending and trajectory.previous without recording the placement.
   Real emulator evidence from level0_2p: step91, line counter29->30, predicted
   clear1, pre-skip counter30, discontinuity=true, previous replay reward0.
   Evidence: checkpoints/experiment_loop/reward_boundary_audit.jsonl.
   Correction: capture counters BEFORE skipping; retain the pending action;
   after readiness, credit its counter delta and bootstrap to the actual next
   playable state. Do not compare board geometry across a level redraw and do
   not invent a terminal boundary. Simulator API adds a generic optional hook.
   Regression checks reward7 and observed successor3 rather than lost transition.

3. **New experimental candidate update lost executed death feedback.** This
   was introduced in my opt-in candidate-control experiment, NOT the cause of
   historical sampled-mode v37 failure. It inserted the candidate set and cleared
   control_previous before execution; the later terminal branch then had no
   predecessor to update. Correction: retain the replay item/selected index until
   the action resolves; replace the executed candidate's predicted reward/state
   with actual reward/state, or mark it terminal. Online selection AND target
   evaluation mask terminal continuations to zero. Tests verify a hypothetical
   V=1000 contributes zero following observed death. Unexecuted alternatives
   remain model predictions, so unreachable-action optimism is still a risk.

4. **Evaluation cap inadequate for the requested target.** Default was200
   placements;100 lines generally requires more than250 tetromino placements
   from an empty board. New default1000, report caps explicitly. This cap did
   not cause the low results of episodes that terminated before it.

## Other audited paths / limits

- Sampled TD associates reward with the prior observed afterstate, as required
  when V excludes the reward for creating its own state. Stored next board is
  observed, not imagined. Existing terminal and delayed-readiness tests retained.
- The Z exclusion is separately verified/fixed in games.json; type7 is playable.
- Resume restores weights/optimizer, not replay/RNG/target history; it is not an
  exact continuation. Reducing exploration does not repair learned rankings.
- Candidate control's simulated alternatives are an approximation. Reward fixes
  do not establish their physical reachability or calibrated long-horizon value.
- gamma0.99 still discounts future lines. Lines-only immediate reward is not an
  undiscounted total-lines objective; changing gamma is a separate experiment.
- Sparse lines-only learning is not automatically easy. Do not reintroduce
  hidden drop/board rewards to make loss or episode return look better.
- Old learned checkpoints have different objectives. Fresh runs required for
  clean comparison; never silently reuse old weights under the new contract.

The old reward candidate-control extension is preserved separately and is not
evidence for these new corrections. Next experiment must use the corrected code
and explicit lines-only configuration, with per-game100-line success reported.
