#!/usr/bin/env python3
"""
Make a SPREAD of start states, so training is not one memorised game.

A save state restores the emulator exactly, including whatever seeds the game's
own randomness. Train from one and the agent can memorise that one game instead
of learning to play: measured on TetrisTime, 8 episodes from level0_2p produced
the identical 76 pieces, a RANDOM policy produced the same 76, and so did up to
300 no-op frames at reset. The policy that scored +217 with 16 lines there
scored -2 with 1 line on a different state. That is not a Tetris player.

This plays forward from a base state with random actions and saves the emulator
along the way, giving states that differ in BOTH the board and how far into the
game they are. Combined with a second base state, which carries a genuinely
different sequence, the agent has to read the board rather than recite a script.

    python tools/make_start_states.py --game TetrisTime-Nes-v0 \
        --from level0_2p --count 12 --min-steps 2 --max-steps 45

Then train on all of them -- the trainer draws one per episode:

    python train.py --game TetrisTime-Nes-v0 --state "$(python tools/make_start_states.py --game TetrisTime-Nes-v0 --list)"

Nothing here is game-specific: it drives whatever action space the game's
training block declares, so a macro-placement game advances by placements and a
button game by button presses.
"""
import argparse
import glob
import gzip
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import custom_integrations                                       # noqa: E402
from rl.env import make_env                                      # noqa: E402


def integration_dir(game):
    return os.path.join(custom_integrations.INTEGRATIONS_DIR, game)


def existing(game, prefix):
    d = integration_dir(game)
    return sorted(os.path.basename(p)[:-6]
                  for p in glob.glob(os.path.join(d, prefix + "*.state")))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True)
    p.add_argument("--from", dest="base", default=None,
                   help="Base save state to play forward from (default: the "
                        "game's own training start state)")
    p.add_argument("--count", type=int, default=12, help="How many to write")
    p.add_argument("--min-steps", type=int, default=2,
                   help="Fewest decisions to play before saving")
    p.add_argument("--max-steps", type=int, default=45,
                   help="Most decisions to play before saving. Keep this below "
                        "a typical episode length or most states start nearly "
                        "dead.")
    p.add_argument("--prefix", default="rs",
                   help="Filename prefix (default: rs, for random start)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list", action="store_true",
                   help="Print the comma-separated state names that already "
                        "exist for --prefix, ready to paste into --state")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be written without writing it")
    args = p.parse_args()

    custom_integrations.register()
    if args.list:
        names = existing(args.game, args.prefix)
        base = args.base or ""
        print(",".join(([base] if base else []) + names))
        return

    out_dir = integration_dir(args.game)
    if not os.path.isdir(out_dir):
        sys.exit("No integration directory at %s" % out_dir)

    over = {"state": args.base} if args.base else {}
    env = make_env(args.game, over)
    # Walk down to the emulator itself. gym's .unwrapped stops at
    # GenericRetroEnv, which is a plain Env rather than a Wrapper, so the retro
    # env is one more .env below that.
    node, em = env, None
    for _ in range(10):
        if node is None:
            break
        cand = getattr(node, "unwrapped", node)
        if hasattr(cand, "em"):
            em = cand.em
            break
        node = getattr(node, "env", None)
    if em is None:
        sys.exit("Could not reach the emulator through the wrapper stack.")
    rng = np.random.default_rng(args.seed)

    written = []
    for i in range(args.count):
        env.reset()
        want = int(rng.integers(args.min_steps, args.max_steps + 1))
        done = 0
        for _ in range(want):
            _o, _r, term, trunc, _i = env.step(int(rng.integers(env.action_space.n)))
            done += 1
            if term or trunc:
                break
        if done < args.min_steps:
            print("  skipped one: the episode ended after %d decisions" % done)
            continue
        name = "%s_%02d" % (args.prefix, len(written) + 1)
        path = os.path.join(out_dir, name + ".state")
        if not args.dry_run:
            with gzip.open(path, "wb") as fh:
                fh.write(em.get_state())
        written.append(name)
        print("  %-8s  %2d decisions played%s" % (name, done,
              "  (dry run, not written)" if args.dry_run else ""))
    env.close()

    print("")
    print("wrote %d states to %s" % (len(written), out_dir))
    if written:
        allnames = ([args.base] if args.base else []) + written
        print("train on all of them with:")
        print("  --state %s" % ",".join(allnames))


if __name__ == "__main__":
    main()
