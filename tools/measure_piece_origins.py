"""Measure normalized piece origins relative to RAM on clear upper boards."""
import argparse
import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.env import TrainingSpec, make_env
from rl.simulators import get_simulator
from rl.afterstate import _find_macro_wrapper
from rl.features import grid_from_frame
from rl.macro import MacroPlanGenerator

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--game", required=True)
p.add_argument("--state", required=True)
args = p.parse_args()
spec = TrainingSpec(args.game, {"state": args.state})
env = make_env(args.game, {"state": args.state})
base, macro = env.unwrapped, _find_macro_wrapper(env)
em = base.env.unwrapped.em
sim = get_simulator(spec.afterstate_config["simulator"], config=spec.afterstate_config)
obs, info = env.reset()
measured = {}
for _ in range(100):
    candidates = sim.get_candidates(obs)
    if not candidates:
        obs, _, term, trunc, info = env.step(None)
    else:
        kind = macro._read_var(macro.piece_type_var)
        board = (obs[:sim.rows * sim.cols].reshape(sim.rows, sim.cols) > .5)
        if kind not in measured and not board[:sim.rows // 2].any():
            snapshot = em.get_state()
            results = []
            for rot, shape in enumerate(sim.shapes[kind]):
                em.set_state(snapshot)
                raw_col = macro._read_var(macro.col_var)
                current_col = raw_col - macro.col_offset
                plan = MacroPlanGenerator.plan(rot * sim.cols + current_col, macro.cfg,
                                              macro._read_var(macro.rot_var), current_col)
                for buttons in plan["frames"]:
                    frame, _, _, _, _ = base.step_raw_frame(buttons)
                # Wait until the whole piece is visible in the clear upper half.
                for _ in range(macro.max_settle_frames):
                    frame, _, term, trunc, _ = base.step_raw_frame([macro.commit_button])
                    if term or trunc or macro._read_var(macro.row_var) >= spec.grid["row_offset"] + shape.shape[0] + 1:
                        break
                frame, _, _, _, _ = base.step_raw_frame([])
                full = grid_from_frame(frame, spec.grid).astype(bool)
                added = full & ~board
                points = np.argwhere(added)
                if len(points) != int(shape.sum()):
                    raise RuntimeError((kind, rot, "ambiguous cells", points.tolist()))
                low = points.min(axis=0)
                normalized = np.zeros(shape.shape, dtype=np.uint8)
                for row, col in points - low:
                    normalized[row, col] = 1
                if not np.array_equal(normalized, shape):
                    raise RuntimeError((kind, rot, "shape differs"))
                results.append({"col_offset": int(macro._read_var(macro.col_var) - low[1]),
                                "row_offset": int(macro._read_var(macro.row_var) - low[0])})
            em.set_state(snapshot)
            measured[kind] = results
            print(json.dumps({kind: results}), flush=True)
        chosen = max(candidates, key=lambda c: c["immediate_reward"])
        obs, _, term, trunc, info = env.step(chosen["action"])
    if len(measured) == len(sim.shapes):
        break
    if term or trunc:
        obs, info = env.reset()
env.close()
