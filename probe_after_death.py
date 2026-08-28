#!/usr/bin/env python3
"""
Find the input SEQUENCE that gets the game off the post-death world map and
back into a level.

An earlier version of this probe tested single buttons in isolation and found
nothing -- but that test was wrong. On the SMB3 map you must first WALK along
the path (tapping a direction moves Mario one node per tap) and only then press
A to enter the level you're standing on. A held button, or A pressed while not
on a level tile, does nothing. So single-button probing could never succeed
regardless of whether the map is navigable.

This version:
  * settles much longer, so the death animation and map load actually finish,
    and prints how `info` evolves so you can see when the map appears;
  * saves PNG screenshots so you can look at the real screen;
  * tests multi-step SEQUENCES (tap RIGHT twice, then A, ...), where each tap
    is a press/release pair like a human pressing a d-pad;
  * restores an identical emulator snapshot before every sequence, so all
    candidates are tested from the exact same death state.

"Back in a level" is detected by the level timer counting DOWN, which only
happens inside a level.

Usage:
    python probe_after_death.py --game SuperMarioBros3-Nes-v0
    python probe_after_death.py --game SuperMarioBros3-Nes-v0 --settle 900 --shots ./probe_shots
"""
import argparse
import os

import numpy as np
import stable_retro as retro

try:
    import cv2
except ImportError:
    cv2 = None

# Each sequence is a list of (button, taps). A tap = press then release, which
# is how map movement actually registers -- one node per tap.
SEQUENCES = [
    ("A", [("A", 1)]),
    ("A x3", [("A", 3)]),
    ("START", [("START", 1)]),
    ("RIGHT then A", [("RIGHT", 1), ("A", 1)]),
    ("RIGHT x2 then A", [("RIGHT", 2), ("A", 1)]),
    ("RIGHT x3 then A", [("RIGHT", 3), ("A", 1)]),
    ("UP then A", [("UP", 1), ("A", 1)]),
    ("UP x2 then A", [("UP", 2), ("A", 1)]),
    ("LEFT then A", [("LEFT", 1), ("A", 1)]),
    ("DOWN then A", [("DOWN", 1), ("A", 1)]),
    ("RIGHT, UP, then A", [("RIGHT", 1), ("UP", 1), ("A", 1)]),
    ("UP, RIGHT, then A", [("UP", 1), ("RIGHT", 1), ("A", 1)]),
    ("RIGHT x2, UP, then A", [("RIGHT", 2), ("UP", 1), ("A", 1)]),
    ("START then A", [("START", 1), ("A", 1)]),
]


def step_frames(env, action, n):
    times = []
    for _ in range(n):
        _obs, _rew, terminated, truncated, info = env.step(action)
        times.append(info.get("time"))
        if terminated or truncated:
            break
    return times, info


def tap(env, buttons, button, times_to_tap, press=10, release=12):
    """Press/release a button `times_to_tap` times -- how a human works a menu
    or walks the map. Holding does not advance node-by-node movement."""
    arr = np.array([b == button for b in buttons], dtype=bool)
    off = np.zeros(len(buttons), dtype=bool)
    seen = []
    for _ in range(times_to_tap):
        t1, info = step_frames(env, arr, press)
        t2, info = step_frames(env, off, release)
        seen += t1 + t2
    return seen, info


def timer_ticked(times):
    """Did we get back into a level? Two signals, and the SECOND one matters
    most: the timer counting DOWN means we're in a level, but on level ENTRY
    the timer first jumps back UP to its full value and hasn't started counting
    yet. An earlier version checked only for a decrease and therefore scored a
    genuinely working RIGHT/UP/A sequence as a failure -- the giveaway was
    time going 43 -> 299 with hpos -> 24 (the level start) while the row still
    printed '-'. Treat a significant upward reset as entry too."""
    vals = [t for t in times if t is not None]
    ticked_down = any(b < a for a, b in zip(vals, vals[1:]))
    reset_up = any(b > a for a, b in zip(vals, vals[1:]))
    return ticked_down or reset_up


def save_shot(env, path):
    if cv2 is None:
        return False
    frame = env.unwrapped.get_screen() if hasattr(env.unwrapped, "get_screen") else None
    if frame is None:
        return False
    cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True)
    p.add_argument("--state", default=None)
    p.add_argument("--settle", type=int, default=900, help="Frames to let the death animation and map load finish before snapshotting (default: %(default)s = 15s)")
    p.add_argument("--after", type=int, default=420, help="Frames to wait after each sequence before judging it (default: %(default)s)")
    p.add_argument("--max-frames", type=int, default=20000)
    p.add_argument("--shots", default="./probe_shots", help="Directory for PNG screenshots (default: %(default)s)")
    args = p.parse_args()

    os.makedirs(args.shots, exist_ok=True)
    env = retro.make(game=args.game, state=args.state or retro.State.DEFAULT, render_mode="rgb_array")
    buttons = env.unwrapped.buttons
    print(f"Buttons: {buttons}\n")

    _obs, info = env.reset()
    prev_lives = info.get("lives")

    run_right = np.array([b in ("RIGHT", "B") for b in buttons], dtype=bool)
    died = False
    for frame in range(args.max_frames):
        _obs, _rew, terminated, truncated, info = env.step(run_right)
        lives = info.get("lives")
        if prev_lives is not None and lives is not None and lives < prev_lives:
            print(f"Died at frame {frame}: lives {prev_lives} -> {lives}")
            died = True
            break
        prev_lives = lives
        if terminated or truncated:
            _obs, info = env.reset()
            prev_lives = info.get("lives")
    if not died:
        raise SystemExit("Never died -- raise --max-frames.")

    # Watch the transition so we can see WHEN (and whether) the map appears.
    print(f"\nSettling {args.settle} frames, sampling info every 120:")
    off = np.zeros(len(buttons), dtype=bool)
    for chunk in range(0, args.settle, 120):
        _t, info = step_frames(env, off, 120)
        shot = os.path.join(args.shots, f"settle_{chunk + 120:04d}.png")
        saved = save_shot(env, shot)
        print(f"  +{chunk + 120:4d} frames: hpos={info.get('hpos')} time={info.get('time')} "
              f"lives={info.get('lives')} score={info.get('score')}"
              + (f"  [{shot}]" if saved else ""))

    snapshot = env.unwrapped.em.get_state()
    print(f"\nSnapshot taken. Screenshots in {args.shots}/ -- OPEN THEM to see what screen this is.\n")

    print(f"Testing sequences (each from the identical snapshot, {args.after} frames to judge):\n")
    winners = []
    for label, seq in SEQUENCES:
        env.unwrapped.em.set_state(snapshot)
        seen = []
        for button, count in seq:
            t, info = tap(env, buttons, button, count)
            seen += t
        t, info = step_frames(env, off, args.after)
        seen += t
        got_in = timer_ticked(seen)
        if got_in:
            winners.append(label)
        mark = "IN LEVEL <<<" if got_in else "-"
        print(f"  {label:<26} time={str(info.get('time')):>5} hpos={str(info.get('hpos')):>5}   {mark}")
        if got_in:
            save_shot(env, os.path.join(args.shots, f"win_{label.replace(' ', '_').replace(',', '')}.png"))

    env.close()
    print()
    if winners:
        print("Sequences that reached a level:")
        for w in winners:
            print(f"  {w}")
        print("\nTell Claude which one to build the navigator around.")
    else:
        print(
            "Still nothing. Next step: open the settle_*.png screenshots and\n"
            "describe what's on screen -- world map, game-over, a card/inventory\n"
            "screen, or a frozen death frame. That tells us what the real\n"
            "sequence needs to be (or whether the map is reachable at all)."
        )


if __name__ == "__main__":
    main()
