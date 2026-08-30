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
    buttons = env.unwrapped.buttons
    rams, infos, presses = [], [], []
    while movie.step():
        keys = [movie.get_key(i, 0) for i in range(env.num_buttons)]
        _obs, _rew, terminated, truncated, info = env.step(keys)
        rams.append(env.get_ram().copy())
        infos.append(info)
        presses.append(keys)
        if terminated or truncated:
            break
    env.close()
    if not rams:
        raise SystemExit("No frames replayed -- is that a valid .bk2?")
    return (np.array(rams, dtype=np.int32), infos,
            np.array(presses, dtype=bool), buttons)


def sprint_mask(presses, buttons, run_frames=16):
    """Frames where the player was SUSTAINING a sprint: run button held together
    with a direction, for at least run_frames consecutively.

    A P-meter is defined by what causes it, so searching for what it correlates
    with beats searching for its shape -- especially when the display ('6 arrows
    plus a P') suggests the state may be split across more than one variable."""
    idx = {b: i for i, b in enumerate(buttons) if b}
    run = presses[:, idx["B"]] if "B" in idx else np.zeros(len(presses), bool)
    move = np.zeros(len(presses), bool)
    for d in ("LEFT", "RIGHT"):
        if d in idx:
            move |= presses[:, idx[d]]
    holding = run & move
    # Require a sustained hold: the meter fills only after running a while.
    sustained = np.zeros_like(holding)
    streak = 0
    for i, h in enumerate(holding):
        streak = streak + 1 if h else 0
        if streak >= run_frames:
            sustained[i] = True
    return sustained


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


def score_meter_correlated(col, sprinting):
    """Rank a byte by how well it behaves like a meter DRIVEN BY SPRINTING:
    rising while the sprint is sustained, falling once it stops."""
    d = np.diff(col)
    sprint_at = sprinting[1:]
    ups = d > 0
    downs = d < 0
    n_up, n_down = int(ups.sum()), int(downs.sum())
    if n_up < 3 or n_down < 3:
        return None
    if col.max() - col.min() < 2 or col.max() > 255:
        return None
    up_while_sprint = float((ups & sprint_at).sum()) / n_up
    down_while_idle = float((downs & ~sprint_at).sum()) / n_down
    # Both must be better than chance to be interesting at all.
    if up_while_sprint < 0.55 or down_while_idle < 0.55:
        return None
    return (up_while_sprint + down_while_idle) * 100 + min(n_up, n_down) * 0.1


def score_meter(col, n):
    """A P-meter: bounded small enum that rises AND falls, idling at the bottom.

    Deliberately permissive. A stricter version returned nothing on a real demo,
    which is the wrong failure: a meter that never fills (no sustained sprint in
    the recording) still shows its signature at low values, and hard filters
    hide that. Rank instead of reject."""
    d = np.diff(col)
    ups = int(np.count_nonzero(d > 0))
    downs = int(np.count_nonzero(d < 0))
    if col.max() > 16 or col.max() < 1:
        return None
    if ups < 3 or downs < 3:
        return None
    balance = min(ups, downs) / max(ups, downs)
    bottom_frac = float(np.count_nonzero(col == col.min())) / n
    # Reward balance and bottom-dwelling rather than requiring them.
    return ups + downs + balance * 60 + bottom_frac * 40


def score_timer(col, n):
    """A countdown byte.

    Handles BOTH encodings, because a strict "mostly decreases" test found
    nothing on a real demo: SMB3 keeps the timer as separate BCD DIGITS, and a
    single digit wraps 0 -> 9 constantly, so its ups and downs are nearly equal.
    A whole-value timer still shows the plain decreasing signature."""
    d = np.diff(col)
    ups = int(np.count_nonzero(d > 0))
    downs = int(np.count_nonzero(d < 0))
    if downs < 5:
        return None
    # Plain countdown: decreases dominate.
    if downs >= ups * 3 and col.max() - col.min() >= 5:
        return downs * 10 - ups * 5
    # BCD digit: 0..9, steps of -1 with periodic +9 wraps.
    if col.max() <= 9 and col.min() >= 0:
        minus_one = int(np.count_nonzero(d == -1))
        wraps = int(np.count_nonzero(d == 9))
        if minus_one >= 5 and minus_one >= wraps:
            return minus_one * 8 + wraps
    return None


def find_bcd_timer(ram, n, top):
    """Look for adjacent digit bytes that TOGETHER count down.

    SMB3 shows a 3-digit timer; if it is stored as one digit per byte, no single
    address looks like a countdown but the combination does."""
    results = []
    size = ram.shape[1]
    for addr in range(size - 2):
        trio = ram[:, addr:addr + 3]
        if trio.max() > 9 or trio.min() < 0:
            continue
        value = trio[:, 0] * 100 + trio[:, 1] * 10 + trio[:, 2]
        d = np.diff(value)
        downs = int(np.count_nonzero(d < 0))
        ups = int(np.count_nonzero(d > 0))
        if downs < 10 or downs < ups * 3:
            continue
        if value.max() - value.min() < 20:
            continue
        results.append((downs * 10 - ups * 5, addr, int(value.min()), int(value.max()), ups, downs))
    results.sort(reverse=True)
    if results:
        print("\n3-digit BCD timer candidates (addr = the HUNDREDS digit;")
        print("value = ram[addr]*100 + ram[addr+1]*10 + ram[addr+2]):\n")
        print(f"  {'ADDR':>8}  {'DEC':>6}  {'RANGE':>12}  {'UPS':>5}  {'DOWNS':>5}")
        for _s, addr, lo, hi, ups, downs in results[:top]:
            print(f"  0x{addr:04X}  {addr:6d}  {lo:5d}..{hi:<5d}  {ups:5d}  {downs:5d}")
    else:
        print("\n(no 3-digit BCD timer found either)")


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
    p.add_argument("--compare", default=None, help="Two addresses (e.g. 0x25A2,0x2167) to diff frame by frame. Use when a search returns several candidates that look equally good: identical everywhere means one is a copy of the other (either works); any divergence tells you which is the real variable and which is a display mirror.")
    p.add_argument("--every", type=int, default=60, help="Sample interval for --watch (default: %(default)s)")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--run-frames", type=int, default=16, help="Consecutive frames of run+direction that count as a sustained sprint for --find meter (default: %(default)s)")
    args = p.parse_args()

    if not args.find and not args.watch and not args.compare:
        raise SystemExit("Pass --find {coins,meter,timer}, --watch 0xADDR, or --compare 0xA,0xB")

    print(f"Replaying {args.demo} ...")
    ram, infos, presses, buttons = replay(args.demo)
    n = ram.shape[0]
    print(f"{n} frames, RAM size {ram.shape[1]} bytes.\n")

    if args.compare:
        a, b = [int(x, 0) for x in args.compare.split(",")]
        ca, cb = ram[:, a], ram[:, b]
        diff_idx = np.where(ca != cb)[0]
        print(f"Comparing 0x{a:04X} and 0x{b:04X} over {n} frames:\n")
        if len(diff_idx) == 0:
            print("  IDENTICAL at every frame.")
            print("  One is a copy of the other (typically the live counter and the")
            print("  value the HUD renders from). Either works -- prefer the LOWER")
            print("  address, which is more often the primary. The distinction only")
            print("  matters at edge cases this demo never reached: crossing 100")
            print("  coins (1-Up + reset) and level transitions.")
        else:
            print(f"  DIVERGE on {len(diff_idx)} of {n} frames. First few:")
            print(f"    {'FRAME':>7}  {'VIDEO':>9}  {('0x%04X' % a):>8}  {('0x%04X' % b):>8}")
            for i in diff_idx[:15]:
                secs = i / 60.0988
                print(f"    {i:7d}  {int(secs//60):02d}:{secs%60:06.3f}  {int(ca[i]):8d}  {int(cb[i]):8d}")
            print("\n  Check these frames on the video: whichever matches the on-screen")
            print("  value is the real variable; the other lags or is a scratch copy.")
        return

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

    rows = []
    if args.find == "meter":
        # Search by CAUSE rather than shape: a P-meter is whatever fills while
        # you sprint and drains when you stop. Shape alone was too weak -- the
        # display ("6 arrows plus a P") hints the state may span more than one
        # variable, and a demo without long sprints shows no shape at all.
        sprinting = sprint_mask(presses, buttons, run_frames=args.run_frames)
        frac = float(sprinting.sum()) / max(1, len(sprinting))
        print(f"Sustained sprint detected on {sprinting.sum()} of {len(sprinting)} "
              f"frames ({frac*100:.1f}%).")
        if sprinting.sum() < 60:
            print("  That is very little sprinting -- record a demo with several")
            print("  LONG flat-out runs (hold B plus a direction) for a clean result.\n")
        else:
            print()
        for addr in range(ram.shape[1]):
            sc = score_meter_correlated(ram[:, addr], sprinting)
            if sc is not None:
                col = ram[:, addr]
                d = np.diff(col)
                rows.append((sc, addr, int(col.min()), int(col.max()),
                             int(np.count_nonzero(d > 0)), int(np.count_nonzero(d < 0))))
        rows.sort(reverse=True)
        print(f"Candidates for 'meter' ({len(rows)} rise with sprinting and fall without):\n")
        print(f"  {'ADDR':>8}  {'DEC':>6}  {'RANGE':>12}  {'UPS':>5}  {'DOWNS':>5}")
        for _sc, addr, lo, hi, ups, downs in rows[:args.top]:
            print(f"  0x{addr:04X}  {addr:6d}  {lo:5d}..{hi:<5d}  {ups:5d}  {downs:5d}")
        if rows:
            print(f"\n{HINTS['meter']}")
            print(f"\nConfirm against the video:")
            print(f"    python find_game_vars.py --demo {args.demo} --watch 0x{rows[0][1]:04X}")
        else:
            print("  (none matched)")
            print("\n  Record a demo with several LONG full-speed sprints (hold B and")
            print("  a direction until the meter fills) separated by full stops.")
        return

    scorer = SCORERS[args.find]
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
    if args.find == "timer":
        # A 3-digit timer stored one digit per byte shows nothing per-address.
        find_bcd_timer(ram, n, args.top)

    if not rows:
        print("  (none matched this signature)")
        if args.find == "meter":
            print("\n  A P-meter only shows its signature if the recording actually")
            print("  fills it. Record a demo with several LONG full-speed sprints")
            print("  (hold B and run flat out until the meter fills) and stops in")
            print("  between, then search again.")
        elif args.find == "timer":
            print("\n  If the BCD search above also found nothing, the timer may be")
            print("  packed (two digits per byte) or simply not run in this demo.")
        else:
            print("\n  Record a demo that exercises this variable more.")
        return

    print(f"\n{HINTS[args.find]}")
    print(f"\nConfirm the top pick against the video:")
    print(f"    python find_game_vars.py --demo {args.demo} --watch 0x{rows[0][1]:04X}")


if __name__ == "__main__":
    main()
