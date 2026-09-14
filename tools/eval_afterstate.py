#!/usr/bin/env python3
"""Deterministic greedy evaluation of afterstate checkpoints.

Training logs mix exploration, varying start states and inherited scores, so they
cannot say whether a checkpoint actually plays better. This runs a FIXED state at
epsilon 0 and reports survival, lines gained since reset, and holes -- the numbers
that matter -- for an untrained network and for each checkpoint, so runs are
comparable like for like.

  python tools/eval_afterstate.py --game TetrisTime-Nes-v0 --state level0_2p \
      --checkpoints checkpoints/foo/ckpt_10000_steps.zip checkpoints/foo/final.zip

Note: with epsilon 0 and one fixed state the emulator is deterministic, so each
state contributes ONE trajectory. Pass several --state values for a real average.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.env import make_env, TrainingSpec
from rl.vars import GameVars
from rl.simulators import get_simulator
from rl.afterstate import AfterstateAgent


def evaluate(env, sim, vars, spec, agent, states, episodes_per_state, lines_var):
    lens, lines, holes = [], [], []
    for _ in range(max(1, episodes_per_state) * max(1, len(states))):
        obs, info = env.reset()
        ln = 0
        l0 = info.get(lines_var, 0) or 0
        pending = False
        guard = 0
        while guard < 20000:
            guard += 1
            ram = getattr(env.unwrapped, "ram", None) if hasattr(env, "unwrapped") else None
            cands = sim.get_candidates(obs=obs, ram=ram, info=info, vars=vars, spec=spec)
            action = None
            if not pending and cands:
                if agent is None:
                    action = cands[np.random.randint(len(cands))]["action"]
                else:
                    action, _feat, _imm = agent.select_action(cands, epsilon=0.0)
                pending = True
            obs, _r, term, trunc, info = env.step(action)
            if action is not None:
                ln += 1
            if info.get("afterstate_ready", True):
                pending = False
            if term or trunc:
                break
        lens.append(ln)
        lines.append((info.get(lines_var, 0) or 0) - l0)
        holes.append(info.get("holes", 0) or 0)
    return np.mean(lens), np.mean(lines), np.mean(holes), len(lens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", required=True)
    p.add_argument("--state", action="append", required=True,
                   help="Start state; repeat the flag to average over several.")
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--episodes", type=int, default=1,
                   help="Episodes per state (>1 only helps if the game is not deterministic).")
    p.add_argument("--include-random", action="store_true",
                   help="Also report random placement as a floor.")
    args = p.parse_args()

    states = ",".join(args.state)
    spec = TrainingSpec(args.game, {"state": states})
    sim = get_simulator(spec.afterstate_config.get("simulator"), config=spec.afterstate_config)
    if sim is None:
        sys.exit("No afterstate simulator for %s" % args.game)
    env = make_env(args.game, overrides={"state": states})
    vars = GameVars(spec.game, entry=spec.entry)
    lines_var = getattr(sim, "lines_var", None) or "lines"

    print("state(s): %s | epsilon 0 (greedy)" % states)
    print("%-46s %9s %9s %9s" % ("policy", "survival", "lines/ep", "holes"))
    rows = []
    if args.include_random:
        np.random.seed(0)
        rows.append(("random placement (floor)",
                     evaluate(env, sim, vars, spec, None, args.state, args.episodes, lines_var)))
    rows.append(("UNTRAINED value net (baseline)",
                 evaluate(env, sim, vars, spec,
                          AfterstateAgent(input_dim=sim.feature_dim, device="cpu"),
                          args.state, args.episodes, lines_var)))
    for ck in args.checkpoints:
        if not os.path.exists(ck):
            print("  (missing: %s)" % ck)
            continue
        a = AfterstateAgent(input_dim=sim.feature_dim, device="cpu")
        a.load(ck)
        rows.append((ck, evaluate(env, sim, vars, spec, a, args.state, args.episodes, lines_var)))
    for label, (sv, li, ho, n) in rows:
        print("%-46s %9.1f %9.2f %9.1f   (n=%d)" % (label[-46:], sv, li, ho, n))
    env.close()
    print("\nAcceptance test: a trained checkpoint should BEAT the untrained baseline.")


if __name__ == "__main__":
    main()
