# Autonomous Tetris experiment loop

User authorization: run WSL training/evaluation, capture metrics, revise and repeat;
resume after account cooldown. Hourly heartbeat: tetris-training-experiment-loop.
Do not buy/redeem credits. No parallel GPU jobs. Keep unrelated edits/old models.
Standing user instruction (2026-09-15): immediately commit and push repository
changes to origin (https://github.com/rqarehbaghi/Retro_Games.git) as they are made.
This includes future experiment/code changes, not just the initial check-in.
Keep ROMs, checkpoints, generated outputs and unrelated private data out of Git.
Do not leave completed source changes uncommitted/unpushed; report push failures.

## Locations

- Editable code: G:/GitHub/Retro_Games (WSL /mnt/g/GitHub/Retro_Games).
- Python: /home/reza/RetroGames/venv/bin/python.
- ROM/states, read-only: /home/reza/RetroGames/integrations.
- Baseline: /home/reza/RetroGames/checkpoints/tetris_v36/final.zip.
- Results and immutable per-run games.json: checkpoints/experiment_loop/.
- Production reward remains unchanged. Verified type-7 configuration correction
  was applied to Windows games.json on 2026-09-14; WSL checkout not yet synced.

## Diagnosis and acceptance

Read TETRIS_V36_DIAGNOSIS.md. Repeated absolute board cost can reward early death.
Existing tools/eval_afterstate.py is not a matched multi-state benchmark.
Use an explicit per-state schedule, zero exploration, real lines gained and
placements. Reward totals across different objectives are NOT comparable.

First gate: reward tests must prefer recovery to early death on known examples.
Then compare frozen checkpoints with v36 on level0_2p and rs_01..rs_15.
Use rs_16..rs_30 as held-out evaluation, never silently train on them.
Provisional promotion target: >=50% improvement in mean lines on the fixed
training-state suite, with no substantial broad survival regression; confirm
across two training seeds and held-out states before calling it robust.
This is an initial acceptance gate, not proof of strong Tetris. Also track empty
board progress toward 30+ lines and persistent execution/prediction mismatches.

## Experiment procedure

1. Inspect processes and manifests first. Never duplicate an active run.
2. Preserve source/config snapshots, seed, command and baseline evaluation.
3. Train bounded stages, evaluate each, change one hypothesis at a time.
4. Compare actual task metrics. Do not promote a model because loss fell.
5. Persist failures and next decisions here. Keep game rules in games.json
   snapshots; Python changes must work for other configured games.
6. If usage unavailable, leave durable state and use the heartbeat to retry.
7. Pause heartbeat only after robust improvement is verified or user stops work.

## Active experiment

LATEST 2026-09-15 02:37 UTC: candidate-control10k completed. Fixed16-state
lines/placements: learned5k78/634,10k90/669; same-seed sampled10k43/512,
small-replay10k54/538; control330/1134. Empty-board learned10k5lines, so not
strong play and not promoted. There is an early comparative learning advantage.
Next active manifest experiments/tetris_control_s17_extend20k.json; output
checkpoints/experiment_loop/seven_piece_control_s17_extend20k/.
20k additional placements with unchanged reward/backup, epsilon0.03, resume10k.
This is approximate continuation: replay/target/RNG reset, not an uninterrupted
30k comparison. Evaluate additional10k/20k (cumulative20k/30k), compare with prior
sampled20k50lines/30k75lines and value-disabled330. If promising, next confirm
with a fresh uninterrupted run and second seed before drawing robust conclusions.

LATEST 2026-09-15: user stopped all jobs and explicitly reiterated that the
objective is to MAKE THE NETWORK TRAIN BETTER. Confirmed no train processes.
New experiment experiments/tetris_control_s17.json ->
checkpoints/experiment_loop/seven_piece_control_s17_10k/. Check status/process.
Optional generic backup=greedy_candidates in rl/control.py compares simulator
placements for the REAL observed next piece (no invented next-piece distribution).
Online net selects reward+gamma*V, target net evaluates the chosen candidate.
The previous sampled observed-transition mode remains the production default.
This tests control-target formulation, not an epsilon tweak. The model-based
mode can be optimistic about unreachable/terminal placements; measured simulator
errors remain a risk, and this is NOT promoted as a proven fix. It does not
replace the learned network with a heuristic. 28 afterstate +3 control tests pass.
Fresh seed17, same potential task reward, state suite, replay50000, epsilon
schedule10000 as previous seven-piece run. Evaluate5k/10k; compare10k against
sampled10k43 lines (small-buffer10k54), plus current value-disabled330 lines.
All state evaluations fixed; no claims based on loss. Keep WSL clean/untouched.

Recent CPU diagnosis of the user's v37/70k on rs12: learned7placements/0lines,
return-9.881 versus value-disabled45/6, return0.322. Same production absolute
objective; poor learned ranking is not explained by reward preference on this
trajectory. Evidence: checkpoints/experiment_loop/v37_70k_rs12_calibration.jsonl.

CHECK-IN 2026-09-14: user requested committed changes and a clean WSL checkout
ready for git pull. WSL older tracked edits and standalone env.py were preserved
in named stash codex-before-pull-2026-09-14. Local assets remain in place under
Git exclusions (ROMs, recordings, checkpoints, rs save states). Do not auto-pop
that stash: its older training edits overlap the newly committed fixes.
Windows is the source of the check-in; WSL remains at the previous revision until
the user pulls. Future experiment work must preserve this clean-checkout intent.

LATEST 2026-09-14 21:xx UTC: first corrected seven-piece30k run COMPLETED.
16-state totals lines/placements: control330/1134, corrected v36 55/321,
learned10k43/512,20k50/554,30k75/624. Empty learned30k49placements/5lines.
Execution errors mostly1-3 per episode, neutral waits essentially gone. Missing
Z was a real bug but learned decision quality remains far below control.

New active manifest experiments/tetris_seven_piece_recent_s17.json, output
checkpoints/experiment_loop/seven_piece_recent_s17_30k/. Check status/process.
Fresh same seed17, same30k duration and epsilon schedule, same reward and states;
ONLY experimental difference is replay_capacity2048 vs50000. Hypothesis: sampled
afterstate policy-evaluation targets from the entire exploratory history lag the
current greedy policy. This is a hypothesis test, not an established diagnosis.
Added generic optional afterstate.replay_capacity (default50000 preserved),
reject capacity<batch_size.25 tests pass, including recent replay retention.
Next compare learned10k/20k/30k with matching checkpoints from the50k-buffer run,
not merely v36. If still far below control, test target formulation/representation
or reward ablations rather than repeatedly extending the same weak learner.

LATEST 2026-09-14 20:xx UTC: seven-piece fresh training launched, manifest
experiments/tetris_seven_piece_s17.json, output
checkpoints/experiment_loop/seven_piece_potential_s17_30k/. Check its status and
process before launching anything else. 30k fresh placements, seed17, potential
objective unchanged, epsilon0.2->0.03 over10k, eval10k/20k/30k on all16 states.
Corrected full-set baseline and control are evaluated under the same seven-piece
configuration. Historical six-piece scores must be labeled separately.

VERIFIED FIX EFFECT (all episodes terminal, no caps), control placements/lines:
empty91/30, rs03 103/29, rs12 45/6; formerly39/5,36/8,23/2.
Wait calls now2,4,0 vs808,1095,413; mismatches1,3,2 vs6,5,4.
Unchanged v36 under corrected config:43/10,30/6,5/0. Thus missing piece is a
proven execution/configuration bug, but old learned policy still underperforms.
24 regression tests pass, including generic config shape/encoding/control
consistency across all configured grid games. No new game-specific Python.
Source is Windows-mounted; WSL production checkout still needs explicit sync
before recommending its normal train.py command to the user.

UPDATE 2026-09-14 20:08 UTC: TYPE 7 IS CONFIRMED PLAYABLE Z. Last hour's
frame/RAM trace (unknown_piece_trace/) showed type7 descend with neutral controls
from frame704 row4 to frame955 row12 and onward. Screenshots visibly show Z.
type7_rotations/ contains pixel measurements and screenshots for ALL4 rotations:
0/2 [[0,0],[0,1],[1,1],[1,2]], 1/3 [[0,1],[1,0],[1,1],[2,0]], col_offset3.
Corrected Windows games.json: shape7, nameZ, valid types1..7, observation and
simulator piece_types8. This does not change learned feature_dim221, so existing
afterstate checkpoints still load. Previous six-piece claim was FALSE.
Cooldown interrupted verification at19:xx. Resumed now;23 regression tests pass.
Calibration command failed because the corrected control reached a LEVEL
TRANSITION (unsupported by calibration tool); type7_corrected_baseline.jsonl is
empty, preserve it as failed attempt. A transition-aware fixed evaluation is
now running, output type7_corrected_eval.json (control then v36, states empty,
rs03,rs12, cap200). Check process/output before starting more work.

UPDATE 2026-09-14 18:07 UTC: extension COMPLETED, no active training worker.
Do not extend again before investigating the systematic extra pieces below.

Full 16-state evaluation totals (lines / placements): v36 53/237;
value-disabled 70/468; learned cumulative20k 45/346, cumulative30k 51/365,
cumulative50k 68/429. Empty-board learned50k: 46 placements / 8 lines.
This beats v36 aggregate lines by 28%, NOT the provisional50% gate, and still
does not beat value-disabled. No production promotion.

Calibration measured using the exact experiment games.json (new diagnostic
--config/--output options): calibration.jsonl under the extension output.
Learned mean V minus actual continuation return: empty -1.825, rs03 -0.966,
rs12 +1.374. Not uniform exploding values. Learned overrides greedy reward
33/46,21/36,12/15 times respectively; measured lines8,6,1 vs control5,8,2.

IMPORTANT NEW EVIDENCE: board_mismatches.jsonl contains full before/after
observations and predicted features for learned and value-disabled empty runs.
Both have mismatches at placements12,18,23,28,31, despite DIFFERENT actions.
For learned policy, differences are ONLY extra observed cells:
- placement12: (16,4),(16,5),(17,5),(17,6)
- placement18: (14,4),(14,5),(15,5),(15,6)
- placement23: (11,4),(11,5),(12,5),(12,6)
- placement28: TWO copies, rows8-9 and10-11, same columns4-6
- placement31: rows6-7, same pattern
- placement39: TWO copies, rows3-4 and5-6, same pattern
All are the SAME Z-like four-cell shape dropped at the center, on top of the
predicted board. This is stronger evidence than aggregate mismatch rate and
looks like additional uncontrolled pieces, NOT ordinary destination error.
Current config says type0/7 are transient and there is no Z shape; this must
now be checked against FRAME/RAM traces rather than assumed true. Working
hypothesis: a real piece is excluded from valid types and neutral waiting lets
it drop uncontrolled. Could also be an external game event; not yet proven.

NEXT PRIORITY: trace emulator frames/type/row/column through the first mismatch
from level0_2p, including every neutral wait and macro-settling phase; capture
screenshots. Determine whether excluded type0/7 is a playable Z and whether
macro spawn detection skips it. Never invent shapes/addresses: use measured
pixels/RAM and put any verified type/shape/config changes in games.json. Then
rerun fixed-state control and learned evaluations before more training. The
diagnostic process finished successfully; no GPU job is running as of this update.

UPDATE 2026-09-14 16:07 UTC: first 10k experiment completed. Next manifest is
experiments/tetris_potential_s17_extend40k.json, output
checkpoints/experiment_loop/potential_task_s17_extend40k/. Check that output's
status.json/process first on the next wakeup. The details below describe its
completed predecessor.

First fixed diagnostic results (placements / lines):

| Policy | level0_2p | rs_03 | rs_12 |
|---|---|---|---|
| v36 | 35 / 10 | 25 / 7 | 4 / 0 |
| value disabled, either config | 39 / 5 | 36 / 8 | 23 / 2 |
| potential 2.5k | 25 / 3 | 17 / 2 | 7 / 0 |
| potential 5k | 28 / 2 | 17 / 0 | 17 / 1 |
| potential 10k | 29 / 6 | 28 / 1 | 20 / 1 |

No promotion: total lines 8 vs v36's 17 and value-disabled 15. Difficult-state
survival recovered relative to v36 but learned policy still underperforms the
value-disabled control. Low loss is not success. All these episodes terminated,
none hit evaluation caps. Prediction mismatches persist (10k: 4,4,3).

Next decision: hold objective/code fixed and train 40k additional placements to
test training duration before rejecting this configuration. Approximate resume
from first run final.zip; replay is rebuilt, target resets to online weights,
RNG restarts at seed17, epsilon stays at prior final 0.03. This is NOT an exact
uninterrupted 50k run. Checkpoint names count additional steps (10k means 20k
cumulative exposure). Evaluate 10k/20k/40k additional-step checkpoints on all16
training states against v36 and both value-disabled controls after training.
Use original baseline config snapshot, not potentially changing production JSON.
If learned policy still underperforms, do not simply extend again: investigate
backup/value representation and actual execution errors with targeted evidence.

- Manifest: experiments/tetris_potential_s17.json
- Worker: tools/run_rl_experiment.py, WSL PID 2190576 at launch (verify process
  command/start time; do not assume a stale PID still belongs to this worker).
- Output: checkpoints/experiment_loop/potential_task_s17_10k/
- Stage: 10,000 fresh placements; snapshots at 2,500/5,000/7,500/10,000.
- Hypothesis: linear line reward plus positive placement reward, with discounted
  potential guidance and zero absorbing-terminal potential, removes the repeated
  inherited-board cost incentive. No v36 weight resume across this objective.
- 23 regression tests passed, including telescoping discounted return and an
  early-death/recovery reward comparison. Production games.json is unchanged.
- The worker saves status.json, source/config snapshots, train.log, checkpoints,
  evaluation.log, and results.json. After training it compares value-disabled
  production reward, v36, value-disabled experiment reward, and experiment
  checkpoints on level0_2p, rs_03, rs_12. These three states are a diagnostic gate,
  NOT final promotion evidence. Expand to the full 16-state suite if promising.
- Command: WSL Python tools/run_rl_experiment.py experiments/tetris_potential_s17.json
  with cwd /mnt/g/GitHub/Retro_Games. Integration files remain in the WSL checkout.

## Next continuation

Check status.json, train.log tail and WSL process list. If active, let it finish;
do useful analysis but do not duplicate the GPU job. If complete, compare actual
lines/placements and decide the next experiment. If failed, read its traceback
and fix the specific failure. Do not rerun into an existing output folder.
For interrupted-process recovery, create a new manifest/output and explicitly
resume its latest checkpoint using train_options.resume; preserve the original
config snapshot and account for completed steps. Replay/target/annealing state
are not fully checkpointed, so record this as an approximate recovery, not an
exact continuation. Never delete previous experiment results to reuse a name.

The hourly heartbeat is active and requests continued work in this same task.
It cannot bypass account limits or start local work while the host is unavailable.
