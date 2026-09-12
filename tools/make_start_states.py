#!/usr/bin/env python3
"""
Save a spread of LIVE start states for training variety.

Training from ONE save state memorises one game: the piece sequence is seeded
by the state, so every episode is identical and the agent learns a script, not
skill. This walks forward from one or more base states with random play and
saves the emulator at a range of depths, giving states that differ in:

  - LEVEL          -- one per base state (level0_2p is level 0, level8_2p is 8)
  - BOARD FILL     -- shallow walks are near-empty, deep walks are mid-played
  - PIECE SEQUENCE -- a state saved at depth D faces sequence[D..], so different
                      depths face different upcoming pieces

Every saved state is VERIFIED live before it is kept: a state that is already
game over trains the agent to top out on its first move, which is the bug the
first version of this tool shipped. Each candidate is reloaded, stepped once,
and deleted if it ends immediately.

    python tools/make_start_states.py --game TetrisTime-Nes-v0 \
        --from level0_2p,level8_2p --count 20

Then train across all of them (the trainer draws one per episode when --state is
a comma list):

    python tools/make_start_states.py --game TetrisTime-Nes-v0 \
        --from level0_2p,level8_2p --list        # prints the --state string

Nothing here is game-specific: it drives whatever action space the game's
training block declares and reads the player/game-over from games.json.
"""
import argparse
import gc
import glob
import gzip
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import custom_integrations                                       # noqa: E402
from rl.env import TrainingSpec, make_env                        # noqa: E402


def integration_dir(game):
    return os.path.join(custom_integrations.INTEGRATIONS_DIR, game)


def base_env_of(env):
    """The GenericRetroEnv under the wrapper stack (it carries `vars`, `_ram`,
    and `.env` = the retro env whose unwrapped side has load_state and em).

    gym.Wrapper forwards unknown attributes to the env it wraps, so a hasattr
    walk stops at the first WRAPPER (it "has" _ram by forwarding), not at the
    real base. .unwrapped is the property that returns the true base Env."""
    return env.unwrapped


def emulator_of(base_env):
    return base_env.env.unwrapped.em


def read_var(base_env, name, default=None):
    v = base_env.vars.read(name, base_env._ram(), {}, default)
    return default if v is None else v


def existing(game, prefix):
    d = integration_dir(game)
    return sorted(os.path.basename(p)[:-6]
                  for p in glob.glob(os.path.join(d, prefix + "*.state")))


def survival(env, rng, tries=3, cap=400):
    """Median decisions a random policy lasts here -- the depth budget."""
    out = []
    for _ in range(tries):
        env.reset()
        k = 0
        while k < cap:
            _o, _r, term, trunc, _i = env.step(int(rng.integers(env.action_space.n)))
            k += 1
            if term or trunc:
                break
        out.append(k)
    return int(np.median(out))


def walk(env, depth, rng):
    """Play `depth` random decisions from a fresh reset. Return True if the
    game is still going (safe to save), False if it ended on the way."""
    env.reset()
    for _ in range(depth):
        _o, _r, term, trunc, _i = env.step(int(rng.integers(env.action_space.n)))
        if term or trunc:
            return False
    return True


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True)
    p.add_argument("--from", dest="base", default=None,
                   help="Comma-separated base states to walk forward from. Each "
                        "contributes its own level. Default: the game's training "
                        "start state.")
    p.add_argument("--count", type=int, default=20, help="Total states to make")
    p.add_argument("--prefix", default="rs", help="Filename prefix (default rs)")
    p.add_argument("--min-depth", type=int, default=2,
                   help="Fewest decisions before the shallowest (near-empty) save")
    p.add_argument("--depth-cap-frac", type=float, default=0.7,
                   help="Deepest walk as a fraction of how long a random policy "
                        "survives, kept below 1.0 so nothing is saved already "
                        "topped out.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list", action="store_true",
                   help="Print the comma-separated state names (base states plus "
                        "any already generated) for --state, and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be written without writing it")
    args = p.parse_args()

    custom_integrations.register()
    bases = [s.strip() for s in (args.base or "").split(",") if s.strip()]
    if not bases:
        bases = [TrainingSpec(args.game).state]

    if args.list:
        print(",".join(bases + existing(args.game, args.prefix)))
        return

    out_dir = integration_dir(args.game)
    if not os.path.isdir(out_dir):
        sys.exit("No integration directory at %s" % out_dir)

    spec = TrainingSpec(args.game)
    go_var = "game_over_p%d" % spec.player
    lvl_var = "level_p%d" % spec.player
    rng = np.random.default_rng(args.seed)

    # Split the count across the base states as evenly as possible.
    per = [args.count // len(bases)] * len(bases)
    for i in range(args.count - sum(per)):
        per[i] += 1

    written = []       # list of (name, base)
    idx = 0
    for base, n in zip(bases, per):
        if n <= 0:
            continue
        env = make_env(args.game, {"state": base})
        benv = base_env_of(env)
        em = emulator_of(benv)
        env.reset()
        level = read_var(benv, lvl_var, "?")
        surv = survival(env, rng)
        cap = max(args.min_depth + 1, int(surv * args.depth_cap_frac))
        # Depths spread evenly from shallow (near-empty) to deep (mid-played),
        # so the batch covers a range of fills instead of clustering.
        depths = list(np.linspace(args.min_depth, cap, num=n).round().astype(int))
        print("base %-11s level=%s  random survives ~%d, walking %d..%d over %d states"
              % (base, level, surv, min(depths), max(depths), n))
        made = attempts = 0
        while made < n and attempts < n * 6:
            attempts += 1
            depth = int(depths[made]) if made < len(depths) \
                else int(rng.integers(args.min_depth, cap + 1))
            if not walk(env, depth, rng):
                continue                      # topped out early -- try again
            idx += 1
            name = "%s_%02d" % (args.prefix, idx)
            if not args.dry_run:
                with gzip.open(os.path.join(out_dir, name + ".state"), "wb") as fh:
                    fh.write(em.get_state())
            written.append((name, base))
            made += 1
            print("  %-8s base %-11s depth %3d%s"
                  % (name, base, depth, "  (dry run)" if args.dry_run else ""))
        env.close()
        del env, benv, em
        gc.collect()

    if args.dry_run or not written:
        print("\n%d states would be written." % len(written))
        return

    # VERIFY every saved state is a live game. One emulator, switching states
    # with load_state -- the exact check training makes on reset. A dead state
    # (already game over) is deleted so it can never poison a run.
    print("\nverifying every state is a LIVE game:")
    venv = make_env(args.game, {"state": bases[0]})
    vbenv = base_env_of(venv)
    dead = []
    for name, base in written:
        vbenv.env.unwrapped.load_state(name)
        venv.reset()
        _o, _r, term, trunc, info = venv.step(0)
        alive = read_var(vbenv, go_var, None)
        if term or trunc or alive == 0:
            dead.append(name)
            print("  %-8s base %-11s DEAD -- ends on its first move" % (name, base))
        else:
            print("  %-8s base %-11s level %s  holes %s  height %s  OK"
                  % (name, base, read_var(vbenv, lvl_var, "?"),
                     info.get("holes"), info.get("height")))
    venv.close()

    for name in dead:
        os.remove(os.path.join(out_dir, name + ".state"))
    kept = [w for w in written if w[0] not in dead]
    if dead:
        print("  removed %d dead state(s)" % len(dead))

    print("\nwrote %d live states to %s" % (len(kept), out_dir))
    print("train on all of them (base states included) with:")
    print("  --state %s" % ",".join(bases + [n for n, _ in kept]))


if __name__ == "__main__":
    main()
