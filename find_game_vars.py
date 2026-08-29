#!/usr/bin/env python3
"""
Find RAM addresses for game variables the integration doesn't publish, by
matching their BEHAVIOUR in one of your own recordings.

Motivated by three gaps found while auditing SuperMarioBros3-Nes-v0:

  COINS   -- not exposed at all, so coin collection is invisible to shaping
             except through the score it also grants.
  P-METER -- SMB3's speed/power meter. It fills while running at full tilt and,
             once full, is what lets raccoon Mario take off. Nothing in the
             pipeline can see it, which is precisely why "explore the sky" is
             currently unreachable: there is no signal for the state that
             precedes flight.
  TIMER   -- the published `time` reads a constant 299 and then jumps to 43, so
             it is NOT the level timer. Whatever it is, it isn't usable.

Each is found by its signature over a demo rather than by guessing an address:

  --find coins   values in a small non-negative range that step UP by one and
                 essentially never fall (except a reset at 100 / new level)
  --find meter   a small bounded enum (0..~8) that rises AND falls repeatedly
                 and spends much of its time at the bottom -- a meter filling
                 and draining as you sprint and stop
  --find timer   decreases far more often than it increases, in slow steady
                 steps, over a wide range

Usage:
    python find_game_vars.py --demo ./human_demos_v2/<file>.bk2 --find coins
    python find_game_vars.py --demo ./human_demos_v2/<file>.bk2 --find meter
    python find_game_vars.py --demo ./human_demos_v2/<file>.bk2 --find timer
    python find_game_vars.py --demo ./human_demos_v2/<file>.bk2 --watch 0x0ABC

Record a demo that exercises the thing you're looking for: collect coins
deliberately for --find coins, and do long full-speed sprints (and stops) for
--find meter. Then confirm the winner with --watch and audit_ram.py.
"""
import argparse

import numpy as np
import stable_retro as retro


def replay(bk2_path):
    """Replay a recording, returning (ram_matrix, infos)."""
    movie = retro.Movie(bk2_path)
    movie.step()
    env = retro.make(
        game=movie.get_game(), state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players, render_mode="rgb_array",
    )
    env.initial_state = movie.get_state()
    env.reset()
    rams, infos = [], []
    while movie.step():
        keys = [movie.get_key(i, 0) for i in range(env.num_buttons)]
        _obs, _rew, terminated, truncated, info = env.step(keys)
        rams.append(env.get_ram().copy())
        infos.append(info)
        if terminated or truncated:
            break
    env.close()
    if not rams:
        raise SystemExit("No frames replayed -- is that a valid .bk2?")
    return np.array(rams, dtype=np.int32), infos


def score_coins(col, n):
    """A coin counter: small values, steps of +1, almost never decreases."""
    d = np.diff(col)
    ups = int(np.count_nonzero(d > 0))
    downs = int(np.count_nonzero(d < 0))
    if ups < 3 or col.max() > 99 or col.min() < 0:
        return None
    step_ones = int(np.count_nonzero(d == 1))
    if step_ones < ups * 0.8:      # increments must be single coins
        return None
    if downs > 3:                  # a reset at 100 or a new level, not more
        return None
    return ups * 10 - downs * 5


def score_meter(col, n):
    """A P-meter: bounded small enum, rises AND falls a lot, idles at 0."""
    d = np.diff(col)
    ups = int(np.count_nonzero(d > 0))
    downs = int(np.count_nonzero(d < 0))
    if col.max() > 8 or col.max() < 3:
        return None
    if ups < 5 or downs < 5:
        return None
    # Filling and draining should be roughly balanced over a whole demo.
    balance = min(ups, downs) / max(ups, downs)
    if balance < 0.4:
        return None
    bottom_frac = float(np.count_nonzero(col == col.min())) / n
    if bottom_frac < 0.15:         # a meter spends real time empty
        return None
    return ups + downs + balance * 50


def score_timer(col, n):
    """A countdown: decreases far more than it increases, over a wide range."""
    d = np.diff(col)
    ups = int(np.count_nonzero(d > 0))
    downs = int(np.count_nonzero(d < 0))
    if downs < 10 or downs < ups * 3:
        return None
    if col.max() - col.min() < 5:
        return None
    return downs * 10 - ups * 5


SCORERS = {"coins": score_coins, "meter": score_meter, "timer": score_timer}

HINTS = {
    "coins": "Steps up by 1 each coin. Cross-check a printed frame against the\n"
             "  coin counter on screen in the video.",
    "meter": "SMB3's P-meter is usually 0..7. The winner should climb while you\n"
             "  sprint and fall when you stop -- confirm with --watch, then it can\n"
             "  be rewarded to make sustained running (and therefore flight)\n"
             "  learnable.",
    "timer": "Should tick down steadily during play. The published `time` key\n"
             "  does NOT (it sits at 299 then jumps to 43), so it is not this.",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", required=True, help="A .bk2 recording that exercises the variable")
    p.add_argument("--find", choices=sorted(SCORERS), help="Which signature to search for")
    p.add_argument("--watch", default=None, help="Comma-separated addresses to print over time instead of searching")
    p.add_argument("--every", type=int, default=60, help="Sample interval for --watch (default: %(default)s)")
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    if not args.find and not args.watch:
        raise SystemExit("Pass --find {coins,meter,timer} or --watch 0xADDR")

    print(f"Replaying {args.demo} ...")
    ram, infos = replay(args.demo)
    n = ram.shape[0]
    print(f"{n} frames, RAM size {ram.shape[1]} bytes.\n")

    if args.watch:
        addrs = [int(a, 0) for a in args.watch.split(",")]
        head = f"  {'FRAME':>6}  {'VIDEO':>9}  {'score':>7}"
        for a in addrs:
            head += f"  {('0x%04X' % a):>8}"
        print(head)
        for frame in range(0, n, args.every):
            secs = frame / 60.0988
            line = (f"  {frame:6d}  {int(secs//60):02d}:{secs%60:06.3f}"
                    f"  {str(infos[frame].get('score')):>7}")
            for a in addrs:
                line += f"  {int(ram[frame, a]):>8}"
            print(line)
        print("\nA correct variable moves when the thing it measures moves --")
        print("check a few of these frames against the video.")
        return

    scorer = SCORERS[args.find]
    rows = []
    for addr in range(ram.shape[1]):
        col = ram[:, addr]
        s = scorer(col, n)
        if s is not None:
            d = np.diff(col)
            rows.append((s, addr, int(col.min()), int(col.max()),
                         int(np.count_nonzero(d > 0)), int(np.count_nonzero(d < 0))))
    rows.sort(reverse=True)

    print(f"Candidates for '{args.find}' ({len(rows)} matched the signature):\n")
    print(f"  {'ADDR':>8}  {'DEC':>6}  {'RANGE':>12}  {'UPS':>5}  {'DOWNS':>5}")
    for _s, addr, lo, hi, ups, downs in rows[:args.top]:
        print(f"  0x{addr:04X}  {addr:6d}  {lo:5d}..{hi:<5d}  {ups:5d}  {downs:5d}")
    if not rows:
        print("  (none matched -- record a demo that exercises this variable more:")
        print("   collect plenty of coins, or sprint at full speed and stop repeatedly)")
        return

    print(f"\n{HINTS[args.find]}")
    print(f"\nConfirm the top pick against the video:")
    print(f"    python find_game_vars.py --demo {args.demo} --watch 0x{rows[0][1]:04X}")


if __name__ == "__main__":
    main()
