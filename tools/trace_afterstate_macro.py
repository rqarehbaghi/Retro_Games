"""Print RAM control traces for mismatched placements; no training."""
import argparse
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.env import make_env, TrainingSpec
from rl.simulators import get_simulator
from rl.afterstate import AfterstateAgent, _find_macro_wrapper

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--game", required=True)
p.add_argument("--state", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=40)
args = p.parse_args()
spec = TrainingSpec(args.game, {"state": args.state})
sim = get_simulator(spec.afterstate_config["simulator"], config=spec.afterstate_config)
agent = AfterstateAgent(sim.feature_dim)
agent.load(args.checkpoint)
env = make_env(args.game, {"state": args.state})
macro = _find_macro_wrapper(env)
base = env.unwrapped
raw = base.step_raw_frame
trace = []
def coords():
    return [macro._read_var(v) for v in (macro.piece_type_var, macro.rot_var, macro.col_var, macro.row_var)]
def tick(buttons):
    result = raw(buttons)
    trace.append([buttons, coords()])
    return result
base.step_raw_frame = tick
obs, info = env.reset()
for i in range(args.steps):
    cands = sim.get_candidates(obs)
    if not cands:
        obs, _, term, trunc, info = env.step(None)
    else:
        action, pred, _ = agent.select_action(cands)
        start = coords()
        trace.clear()
        obs, _, term, trunc, info = env.step(action)
        diff = np.count_nonzero(pred[:sim.rows * sim.cols] != obs[:sim.rows * sim.cols])
        if diff and not term:
            print("mismatch", i, "target", divmod(action, sim.cols), "start", start, "cells", diff, "trace", trace, flush=True)
    if term or trunc:
        obs, info = env.reset()
env.close()
