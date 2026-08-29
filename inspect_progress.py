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

import numpy as np
import stable_retro as retro


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True)
    p.add_argument("--state", default=None)
    p.add_argument("--frames", type=int, default=900, help="Frames to run right (default: %(default)s)")
    p.add_argument("--every", type=int, default=30, help="Sample interval (default: %(default)s)")
    p.add_argument("--watch", default=None, help="Comma-separated RAM addresses to print alongside hpos, e.g. 0x053C,0x053D. Use this to CONFIRM a candidate really is level progress: it should climb steadily while hpos flatlines.")
    args = p.parse_args()
    watch = [int(a, 0) for a in args.watch.split(",")] if args.watch else []

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
