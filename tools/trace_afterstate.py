#!/usr/bin/env python3
"""Print the greedy placement sequence the TRAINING path produces.

Live play emits the same lines under PLAY_DEBUG=1, so the two traces can be
diffed decision by decision. That is the only way to tell a live-play EXECUTION
bug (the buttons do not do what was chosen) from a live-play DECISION bug (a
different placement is chosen because the candidates or features differ) -- and
three fixes were spent guessing between those two before this existed.

  python tools/trace_afterstate.py --game TetrisTime-Nes-v0 --state level0_2p_1 \
      --model checkpoints/tetris_v52/ckpt_78000_steps.zip
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.env import make_env, TrainingSpec
from rl.vars import GameVars
from rl.simulators import get_simulator
from rl.afterstate import AfterstateAgent

FEATS = ("holes", "filled", "height", "max_height", "bumpiness")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", required=True)
    p.add_argument("--state")
    p.add_argument("--model", required=True)
    p.add_argument("--player", type=int, default=2)
    p.add_argument("--max-placements", type=int, default=2000)
    a = p.parse_args()

    overrides = {"player": a.player}
    if a.state:
        overrides["state"] = a.state
    spec = TrainingSpec(a.game, overrides)
    sim_name = (spec.afterstate_config.get("simulator")
                or spec.features_name or spec.game)
    sim = get_simulator(sim_name, config=spec.afterstate_config)
    agent = AfterstateAgent(
        input_dim=sim.feature_dim, device="cpu",
        model_type=spec.afterstate_config.get("model_type", "mlp"))
    agent.load(a.model)
    gvars = GameVars(spec.game, entry=spec.entry)

    env = make_env(spec)
    obs, info = env.reset()
    pending, n, frame = False, 0, 0
    while n < a.max_placements:
        ram = getattr(env.unwrapped, "ram", None)
        action = None
        if not pending:
            cands = sim.get_candidates(obs=obs, ram=ram, info=info,
                                       vars=gvars, spec=spec)
            if cands:
                action, _f, _i = agent.select_action(cands, epsilon=0.0)
                pending = True
                n += 1
                print("[trn] frame %d DECIDE action=%s feats=%s"
                      % (frame, action,
                         {k: float(info.get(k, 0) or 0) for k in FEATS}))
        obs, _r, term, trunc, info = env.step(action)
        frame += 1
        if info.get("afterstate_ready", True):
            pending = False
        if term or trunc:
            print("[trn] EPISODE END after %d placements at frame %d" % (n, frame))
            break
    env.close()


if __name__ == "__main__":
    main()
