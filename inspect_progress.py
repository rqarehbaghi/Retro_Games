#!/usr/bin/env python3
"""
Determine what the 'hpos' info variable actually measures, and find a better
one if it's the wrong thing.

This matters enormously for reward shaping. There are two possibilities:

  ABSOLUTE level position -- climbs steadily the whole level (possibly wrapping
    0..255 once per screen). This is what you want to reward: it tracks real
    progress toward the flag.

  ON-SCREEN x position -- Mario walks right until he hits the scroll threshold
    near mid-screen, then the LEVEL scrolls while Mario stays put, so the value
    rises to ~110-130 and then FLATLINES. Rewarding this pays only for the first
    couple of seconds of a level and nothing afterwards -- so the agent has no
    incentive to actually advance, and jittering in place becomes the best
    available strategy.

The test: hold RIGHT+B and watch the value. Plateau => on-screen. Keeps climbing
(or sawtooths through wraps) => absolute.

It also scans ALL of RAM for bytes that rise monotonically while running right,
which is how you find a true level-position variable if hpos isn't one. A
2-byte little-endian pair of such addresses is the usual encoding for a level
that's longer than 255 units.

Usage:
    python inspect_progress.py --game SuperMarioBros3-Nes-v0
"""
import argparse
import json
import os

import numpy as np
import stable_retro as retro


def load_known_vars(game, config="games.json"):
    """Every variable we've located for this game, from games.json, so the dump
    below shows them side by side instead of one at a time."""
    path = config if os.path.isabs(config) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), config)
    try:
        with open(path) as fh:
            return ((json.load(fh).get("games") or {}).get(game) or {}).get("variables") or {}
    except Exception:
        return {}


def read_var(ram_row, info, spec):
    """One variable's value for a frame, per its games.json source type."""
    src = spec.get("source")
    if src == "info":
        return info.get(spec.get("key"))
    if src == "info16":
        # Low byte from an info key, high byte from RAM (SMB3 level position).
        low = info.get(spec.get("key"))
        if low is None:
            return None
        return int(low) + (int(ram_row[int(str(spec["address_high"]), 0)]) << 8)
    if src in ("ram", "ram16", "ram_bcd3"):
        a = int(str(spec["address"]), 0)
        if src == "ram":
            return int(ram_row[a])
        if src == "ram16":
            hi = int(str(spec["address_high"]), 0)
            return int(ram_row[a]) + (int(ram_row[hi]) << 8)
        return int(ram_row[a]) * 100 + int(ram_row[a + 1]) * 10 + int(ram_row[a + 2])
    return None


def find_high_byte(ram, hpos, top=10):
    """Find the page/high byte that pairs with a wrapping low byte.

    A level position wider than 255 is stored as two bytes: the low byte sweeps
    0..255 and wraps, and a high byte ticks up by one on each wrap. So look for
    addresses that increment exactly where the low byte rolls over. This is the
    check that distinguishes "the value is garbage" from "the value is half of a
    16-bit number" -- they look identical until you line up the wraps.
    """
    h = np.asarray([v if v is not None else -1 for v in hpos], dtype=np.int32)
    d = np.diff(h)
    wrap_down = np.where(d < -100)[0]   # 241 -> 3 style rollovers
    wrap_up = np.where(d > 100)[0]      # backtracking across the boundary
    if len(wrap_down) == 0:
        print("\n(no wraps in this recording -- can't pair a high byte. Record a")
        print("demo that travels far enough for the low byte to roll over.)")
        return
    print(f"\nLow byte wrapped {len(wrap_down)} time(s) forward, {len(wrap_up)} backward.")
    print("Looking for a byte that ticks UP on each forward wrap (and down on each")
    print("backward one) -- that is the high byte of a 16-bit position.\n")
    rows = []
    for addr in range(ram.shape[1]):
        col = ram[:, addr]
        cd = np.diff(col)
        hits = int(np.count_nonzero(cd[wrap_down] == 1))
        back = int(np.count_nonzero(cd[wrap_up] == -1)) if len(wrap_up) else 0
        if hits == 0:
            continue
        # A page counter changes ONLY at wraps; penalise anything busier.
        total = int(np.count_nonzero(cd != 0))
        noise = total - hits - back
        score = hits * 100 + back * 50 - noise
        rows.append((score, addr, hits, len(wrap_down), back, len(wrap_up), total,
                     int(col.min()), int(col.max())))
    rows.sort(reverse=True)
    print(f"  {'ADDR':>8}  {'DEC':>6}  {'ON WRAP':>10}  {'ON BACK':>9}  "
          f"{'TOTAL CHG':>9}  {'RANGE':>10}")
    for _s, addr, hits, nw, back, nu, total, lo, hi in rows[:top]:
        print(f"  0x{addr:04X}  {addr:6d}  {hits:4d}/{nw:<5d}  {back:4d}/{nu:<4d}  "
              f"{total:9d}  {lo:4d}..{hi:<4d}")
    if rows:
        best = rows[0]
        print(f"\nBest pairing: 0x{best[1]:04X}. If ON WRAP matches the wrap count and")
        print("TOTAL CHG is close to it, that byte changes only at rollovers -- the")
        print("signature of a page counter. Then level position is:")
        print(f"    position = hpos + (ram[0x{best[1]:04X}] << 8)")
        print(f"and training uses: --progress-address <hpos addr> "
              f"--progress-address-high 0x{best[1]:04X}")


def analyse(rams, hpos_series, args):
    """Rank RAM bytes by how much they behave like accumulated level position.

    Scoring rewards steady climbing and PENALIZES resets to zero, because that
    is exactly what separated a real position counter from 0x00CF, which looked
    monotonic while holding RIGHT but drops back to 0 whenever the player stops
    or reverses -- a per-frame delta, not a position. Also considers adjacent
    16-bit little-endian pairs, the usual encoding for levels longer than 255.
    """
    vals = [v for v in hpos_series if v is not None]
    if vals:
        print(f"\nhpos: min={min(vals)} max={max(vals)} first={vals[0]} last={vals[-1]}")
        # Report what hpos DOES here rather than asserting what it is -- an
        # earlier hardcoded claim ("caps at 144, so it is on-screen x") was
        # drawn from a run where Mario was stuck against the first obstacle,
        # and this very output disproved it.
        cap = max(vals)
        at_cap = sum(1 for v in vals if v >= cap - 2)
        arr = np.asarray(vals, dtype=np.int32)
        d = np.diff(arr)
        wraps = int(np.count_nonzero(d < -100))
        print("\nWHAT hpos DOES IN THIS RECORDING")
        print(f"  spans {min(vals)}..{cap}; sat at/near its maximum on "
              f"{at_cap} of {len(vals)} frames ({at_cap*100.0/len(vals):.0f}%)")
        print(f"  rolled over (dropped by more than 100 in one frame) {wraps} time(s)")
        if wraps >= 2:
            print("  => Rollovers mean this is the LOW BYTE of a wider position: an")
            print("     on-screen coordinate cannot wrap (it clamps at the screen edge")
            print("     or the camera scrolls instead). Pair it with the page counter")
            print("     reported below to recover true level position.")
        elif at_cap > len(vals) * 0.3:
            print("  => Pinned at its maximum for much of the run. Either it is an")
            print("     on-screen coordinate capped by the scroll threshold, or the")
            print("     player was simply blocked -- check the video before deciding;")
            print("     confusing those two cost this project several training runs.")
        else:
            print("  => Moves freely without wrapping: usable as-is if the level fits")
            print("     within its range.")

    ram = np.array(rams, dtype=np.int32)
    n = ram.shape[0]
    print(f"\nProgress candidates over {n} frames of real play.")
    print("Scored on: climbs a lot, rarely collapses, and does NOT keep")
    print("returning to zero.\n")
    print(f"  {'ADDR':>8}  {'DEC':>6}  {'RANGE':>12}  {'UPS':>5}  {'BIGDROPS':>8}  {'ZERO%':>6}")

    rows = []
    for addr in range(ram.shape[1]):
        col = ram[:, addr]
        d = np.diff(col)
        ups = int(np.count_nonzero(d > 0))
        downs = int(np.count_nonzero(d < 0))
        big_drops = int(np.count_nonzero(d < -8))
        zero_frac = float(np.count_nonzero(col == 0)) / n
        if ups < 10 or ups < downs:
            continue
        if zero_frac > 0.15:      # positions don't sit at zero
            continue
        if big_drops > n * 0.02:  # nor collapse repeatedly
            continue
        score = ups - big_drops * 10 - zero_frac * 200
        rows.append((score, addr, int(col.min()), int(col.max()), ups, big_drops, zero_frac))

    rows.sort(reverse=True)
    for score, addr, lo, hi, ups, big_drops, zero_frac in rows[:20]:
        print(f"  0x{addr:04X}  {addr:6d}  {lo:5d}..{hi:<5d}  {ups:5d}  {big_drops:8d}  {zero_frac*100:5.1f}%")
    if not rows:
        print("  (nothing qualified -- every byte either sat at zero or collapsed")
        print("   repeatedly. Try a longer demo with sustained forward progress.)")

    print("\nVerify the top pick before training on it:")
    print(f"    python inspect_progress.py --game {args.game} --demo {args.demo} --watch 0xADDR")
    print("It must rise as you advance and NOT reset when you stop or back up.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True)
    p.add_argument("--state", default=None)
    p.add_argument("--frames", type=int, default=900, help="Frames to run right (default: %(default)s)")
    p.add_argument("--every", type=int, default=30, help="Sample interval (default: %(default)s)")
    p.add_argument("--demo", default=None, help="Scan a .bk2 HUMAN RECORDING instead of a scripted run. Strongly preferred: holding RIGHT exercises a narrow slice of states, and an address that looks like level position under that test can turn out to be a per-frame delta that resets to zero whenever the player stops or backs up (exactly what happened with 0x00CF). Real play covers stopping, reversing, jumping and dying.")
    p.add_argument("--watch", default=None, help="Comma-separated RAM addresses to print alongside hpos, e.g. 0x053C,0x053D. Use this to CONFIRM a candidate really is level progress: it should climb steadily while hpos flatlines.")
    args = p.parse_args()
    watch = [int(a, 0) for a in args.watch.split(",")] if args.watch else []

    if args.demo:
        movie = retro.Movie(args.demo)
        movie.step()
        env = retro.make(game=movie.get_game(), state=None,
                         use_restricted_actions=retro.Actions.ALL,
                         players=movie.players, render_mode="rgb_array")
        env.initial_state = movie.get_state()
        env.reset()
        print(f"Scanning human recording {args.demo}\n")
        # Every variable located so far, side by side -- one column each, with
        # its source in the legend, so a value can be judged in context instead
        # of in isolation.
        known = load_known_vars(movie.get_game())
        cols = [(n, sp) for n, sp in known.items()
                if sp.get("source") not in (None, "unknown")]
        print("COLUMNS")
        for name, sp in cols:
            src = (f"info['{sp['key']}']" if sp["source"] == "info"
                   else f"{sp['source']} @ {sp['address']}")
            flag = "" if sp.get("verified") else "   (UNVERIFIED)"
            print(f"  {name:<10} {src}{flag}")
        for a in watch:
            print(f"  {('0x%04X' % a):<10} extra --watch address")
        print()

        rams, hpos_series = [], []
        header = f"  {'FRAME':>6}  {'VIDEO':>9}"
        for name, _sp in cols:
            header += f"  {name[:9]:>9}"
        for a in watch:
            header += f"  {('0x%04X' % a):>8}"
        print(header)
        frame = 0
        while movie.step():
            keys = [movie.get_key(i, 0) for i in range(env.num_buttons)]
            _obs, _rew, terminated, truncated, info = env.step(keys)
            ram_row = env.get_ram().copy()
            rams.append(ram_row)
            hpos_series.append(info.get("hpos"))
            if frame % args.every == 0:
                secs = frame / 60.0988
                line = f"  {frame:6d}  {int(secs//60):02d}:{secs%60:06.3f}"
                for name, sp in cols:
                    v = read_var(ram_row, info, sp)
                    line += f"  {('-' if v is None else v):>9}"
                for a in watch:
                    line += f"  {int(ram_row[a]):>8}"
                print(line)
            frame += 1
            if terminated or truncated:
                break
        env.close()
        # hpos sweeping 0..255 and rolling over is the signature of a 16-bit
        # position's LOW byte, not of a broken value -- so look for its partner.
        find_high_byte(np.array(rams, dtype=np.int32), hpos_series)
        analyse(rams, hpos_series, args)
        return

    env = retro.make(game=args.game, state=args.state or retro.State.DEFAULT, render_mode="rgb_array")
    buttons = env.unwrapped.buttons
    run_right = np.array([b in ("RIGHT", "B") for b in buttons], dtype=bool)

    _obs, info = env.reset()
    print(f"Holding RIGHT+B for {args.frames} frames.\n")
    header = f"  {'FRAME':>6}  {'hpos':>6}  {'time':>5}  {'score':>6}"
    for a in watch:
        header += f"  {('0x%04X' % a):>8}"
    print(header)

    rams = []
    hpos_series = []
    prev_lives = None
    for frame in range(args.frames):
        _obs, _rew, terminated, truncated, info = env.step(run_right)
        # Stop at the FIRST death. Post-death frames are the map screen, and
        # map animation counters climb there -- including them once produced a
        # false progress candidate (0x053C, a map counter frozen during actual
        # play) because two-thirds of the scan window was map, not level.
        lives = info.get("lives")
        if prev_lives is not None and lives is not None and lives < prev_lives:
            print(f"  (died at frame {frame} -- scan stops here; map frames would poison the candidates)")
            break
        prev_lives = lives
        rams.append(env.get_ram().copy())
        hpos_series.append(info.get("hpos"))
        if frame % args.every == 0:
            line = f"  {frame:6d}  {str(info.get('hpos')):>6}  {str(info.get('time')):>5}  {str(info.get('score')):>6}"
            for a in watch:
                line += f"  {int(rams[-1][a]):>8}"
            print(line)
        if terminated or truncated:
            print(f"  (episode ended at frame {frame})")
            break

    vals = [v for v in hpos_series if v is not None]
    print(f"\nhpos: min={min(vals)} max={max(vals)} first={vals[0]} last={vals[-1]}")
    tail = vals[len(vals) // 2:]
    if max(tail) - min(tail) <= 4:
        print("  => hpos FLATLINES in the second half: it is almost certainly the")
        print("     ON-SCREEN x position, NOT level progress. Rewarding it pays")
        print("     only until Mario reaches the scroll threshold. Use one of the")
        print("     candidate addresses below as the real progress variable.")
    else:
        print("  => hpos keeps changing: plausibly real level position.")

    # Find RAM bytes that mostly increase while running right -- the signature
    # of a level-position counter (a high byte increments a handful of times, a
    # low byte sawtooths through many wraps).
    ram = np.array(rams, dtype=np.int16)
    print("\nRAM bytes that rise consistently while running right (progress candidates):")
    print(f"  {'ADDR':>8}  {'DEC':>5}  {'START':>5}  {'END':>5}  {'UPS':>5}  {'DOWNS':>5}")
    rows = []
    for addr in range(ram.shape[1]):
        col = ram[:, addr]
        d = np.diff(col)
        ups = int(np.count_nonzero(d > 0))
        downs = int(np.count_nonzero(d < 0))
        if ups < 3 or ups < downs * 3:
            continue
        rows.append((ups, addr, int(col[0]), int(col[-1]), ups, downs))
    rows.sort(reverse=True)
    for _, addr, start, end, ups, downs in rows[:20]:
        print(f"  0x{addr:04X}  {addr:5d}  {start:5d}  {end:5d}  {ups:5d}  {downs:5d}")

    env.close()
    print(
        "\nWhat to look for (the scan now covers IN-LEVEL frames only): a HIGH"
        "\nbyte that steps up a few times and never down (a screen/page counter"
        "\n-- the best progress signal), or a pair of adjacent addresses forming"
        "\na 16-bit position."
        "\nALWAYS verify a candidate with --watch before training on it: it"
        "\nmust climb steadily the whole run while you hold RIGHT, not just in"
        "\none phase. Paste the table to Claude to pick together."
    )


if __name__ == "__main__":
    main()
