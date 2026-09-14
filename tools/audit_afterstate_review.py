"""Run the regression suite and optionally a short two-state emulator audit."""
import gc
import importlib.util
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rl.env import TrainingSpec, make_env
from rl.simulators import get_simulator
from rl.vars import GameVars


def live():
    spec = TrainingSpec("TetrisTime-Nes-v0")
    sim = get_simulator("grid_placement", config=spec.afterstate_config)
    vars = GameVars(spec.game, entry=spec.entry)
    rng = np.random.default_rng(7)
    for state in ("level0_2p", "level8_2p"):
        env = make_env(spec.game, overrides={"state": state})
        mismatches = terminal = count = no_candidates = 0
        examples = []
        obs, info = env.reset()
        for _ in range(60):
            candidates = sim.get_candidates(obs, info=info, vars=vars, spec=spec)
            if not candidates:
                no_candidates += 1
                obs, _, term, trunc, info = env.step(None)
                terminal += int(term)
                if term or trunc:
                    obs, info = env.reset()
                continue
            c = candidates[int(rng.integers(len(candidates)))]
            obs, _, term, trunc, info = env.step(c["action"])
            diff = int(np.count_nonzero(c["afterstate"][:200] != obs[:200]))
            count += 1
            mismatches += int(diff > 0)
            terminal += int(term)
            if diff > 4 and len(examples) < 3:
                examples.append({"step": count, "diff_cells": diff, "terminal": term,
                                 "predicted_reward": c["immediate_reward"]})
            if term or trunc:
                obs, info = env.reset()
        env.close()
        del env
        import gc
        gc.collect()
        print(json.dumps({"state": state, "placements": count, "nonidentical_boards": mismatches,
                          "terminals": terminal, "no_candidates": no_candidates, "examples": examples}))



if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("afterstate_tests", os.path.join(ROOT, "tests", "test_afterstate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if not result.wasSuccessful():
        sys.exit(1)
    if "--live" in sys.argv:
        live()
