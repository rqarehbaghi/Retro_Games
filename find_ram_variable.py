#!/usr/bin/env python3
"""
Find the RAM address of a game variable the integration doesn't expose --
e.g. Mario's power-up state (small / big / fire / raccoon / tanooki).

Why this exists: reward shaping can only use values the stable-retro
integration publishes in `info`. SuperMarioBros3-Nes-v0 publishes only
hpos/lives/score/time, so "reward getting a fire flower" is impossible until
the power-state byte is exposed. Rather than trusting a RAM address copied off
the internet (a wrong one silently shapes on garbage), this finds it from your
own recording.

How it works: replays a .bk2 frame by frame, snapshots all of RAM each frame,
then scores every address on how much it behaves like a discrete state
variable -- few distinct values, all small, and changing rarely rather than
every frame. A power-up state is a textbook match: it sits at 0 for thousands
of frames, jumps to 1 when you grab a mushroom, and stays there.

Usage:
    # Record yourself GRABBING POWER-UPS first (that's the signal it needs):
    python play_and_record.py --game SuperMarioBros3-Nes-v0 \\
        --human --record-dir ./human_demos

    # Then find candidate addresses:
    python find_ram_variable.py --demo ./human_demos/<your>.bk2

    # Narrow it down: if you know you got a mushroom ~12 seconds in (frame
    # ~720 at 60fps), show what changed near there:
    python find_ram_variable.py --demo ./human_demos/<your>.bk2 --around 720

    # Once you have a candidate, watch it live to confirm:
    python find_ram_variable.py --demo ./human_demos/<your>.bk2 --watch 0x0016
"""
import argparse
from collections import Counter

import numpy as np
import stable_retro as retro


def replay_ram(bk2_path, max_frames=200000):
    """Replay a .bk2 and return (ram_matrix, infos). ram_matrix is
    (n_frames, ram_size) uint8."""
    movie = retro.Movie(bk2_path)
    movie.step()

    env = retro.make(
        game=movie.get_game(),
        state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
        render_mode="rgb_array",
    )
    env.initial_state = movie.get_state()
    env.reset()

    frames = []
    infos = []
    while movie.step() and len(frames) < max_frames:
        keys = [movie.get_key(i, 0) for i in range(env.num_buttons)]
        _obs, _rew, terminated, truncated, info = env.step(keys)
        frames.append(env.get_ram().copy())
        infos.append(info)
        if terminated or truncated:
            break
    env.close()

    if not frames:
        raise SystemExit("No frames replayed -- is that a valid .bk2?")
    return np.array(frames, dtype=np.uint8), infos


def score_addresses(ram, max_distinct=8, max_value=16, min_changes=1, max_changes=60):
    """Rank addresses that look like a discrete state variable: few distinct
    small values, and changing occasionally rather than constantly."""
    n_frames = ram.shape[0]
    results = []
    for addr in range(ram.shape[1]):
        col = ram[:, addr]
        changes = int(np.count_nonzero(np.diff(col.astype(np.int16))))
        if not (min_changes <= changes <= max_changes):
            continue
        distinct = np.unique(col)
        if len(distinct) > max_distinct or distinct.max() > max_value:
            continue
        # Prefer variables that spend long stretches in one value (a real
        # state) over ones that flicker.
        counts = Counter(col.tolist())
        dominant_frac = counts.most_common(1)[0][1] / n_frames
        results.append({
            "addr": addr,
            "changes": changes,
            "values": distinct.tolist(),
            "dominant_frac": dominant_frac,
        })
    # Fewest changes first, then most "settled".
    results.sort(key=lambda r: (r["changes"], -r["dominant_frac"]))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", required=True, help="A .bk2 recording in which you DID collect power-ups")
    p.add_argument("--top", type=int, default=25, help="How many candidates to print (default: %(default)s)")
    p.add_argument("--around", type=int, default=None, help="Show addresses that changed within --window frames of this frame number (use when you roughly know when you grabbed the power-up)")
    p.add_argument("--window", type=int, default=90, help="Half-width in frames for --around (default: %(default)s = 1.5s)")
    p.add_argument("--watch", default=None, help="Print the full value timeline for one address, e.g. 0x0016 or 22")
    p.add_argument("--max-distinct", type=int, default=8, help="Max distinct values for a candidate (default: %(default)s)")
    p.add_argument("--max-changes", type=int, default=60, help="Max transitions for a candidate (default: %(default)s)")
    args = p.parse_args()

    print(f"Replaying {args.demo} ...")
    ram, infos = replay_ram(args.demo)
    print(f"Replayed {ram.shape[0]} frames, RAM size {ram.shape[1]} bytes.\n")

    if args.watch is not None:
        addr = int(args.watch, 0)
        col = ram[:, addr]
        print(f"Timeline for address {addr} (0x{addr:04X}) -- showing every change:")
        prev = None
        for frame, val in enumerate(col):
            if val != prev:
                secs = frame / 60.0
                print(f"  frame {frame:6d} ({secs:6.1f}s): {prev} -> {val}")
                prev = val
        print(f"\nDistinct values seen: {sorted(set(col.tolist()))}")
        return

    if args.around is not None:
        lo = max(0, args.around - args.window)
        hi = min(ram.shape[0] - 1, args.around + args.window)
        print(f"Addresses that changed between frames {lo} and {hi} "
              f"({lo/60.0:.1f}s - {hi/60.0:.1f}s):\n")
        segment = ram[lo:hi + 1].astype(np.int16)
        changed = np.where(np.count_nonzero(np.diff(segment, axis=0), axis=0) > 0)[0]
        for addr in changed:
            col = ram[:, addr]
            total_changes = int(np.count_nonzero(np.diff(col.astype(np.int16))))
            if total_changes > args.max_changes:
                continue  # too noisy to be a state variable
            before, after = ram[lo, addr], ram[hi, addr]
            print(f"  0x{addr:04X} ({addr:5d}): {before} -> {after}   "
                  f"[{total_changes} changes total, values {sorted(set(col.tolist()))[:8]}]")
        print("\nLook for one whose value stepped up and STAYED (e.g. 0 -> 1) "
              "at the moment you grabbed the power-up.")
        return

    cands = score_addresses(ram, max_distinct=args.max_distinct, max_changes=args.max_changes)
    print(f"{len(cands)} candidate state-like addresses. Top {args.top}:\n")
    print(f"  {'ADDR':>8}  {'DEC':>5}  {'CHANGES':>7}  {'SETTLED':>7}  VALUES")
    for c in cands[:args.top]:
        print(f"  0x{c['addr']:04X}  {c['addr']:5d}  {c['changes']:7d}  "
              f"{c['dominant_frac']*100:6.1f}%  {c['values']}")

    print(
        "\nWhat to look for: a power-up state is usually a byte with values like\n"
        "[0, 1, 2, ...] that changes only a handful of times and sits in one\n"
        "value most of the run. Confirm a candidate with:\n"
        f"    python find_ram_variable.py --demo {args.demo} --watch 0xADDR\n"
        "and check the transitions line up with when you actually grabbed\n"
        "mushrooms/flowers/leaves in that recording."
    )


if __name__ == "__main__":
    main()
