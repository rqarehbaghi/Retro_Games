#!/usr/bin/env python3
"""
Find out which button actually gets the game off the post-death screen.

play_and_record.py's --on-death continue uses a scripted navigator to re-enter
a level after Mario dies. Its button pattern was a guess (tap A, sometimes
RIGHT) and it does not work for SuperMarioBros3-Nes-v0: the navigator burns its
whole budget without the level timer ever restarting. Rather than guess again,
this measures it.

Method: play until a life is lost, let the transition settle, then SNAPSHOT the
emulator. For each candidate input, restore that exact snapshot, hold/tap the
input for a few seconds, and check whether the in-level timer starts counting
down again (the reliable "we are in a level" signal). Restoring between trials
means every candidate is tested from the identical death state.

Note this probes buttons the training ACTION_TABLE doesn't even contain --
notably START -- which is one likely reason the navigator cannot escape: if the
screen needs START, the agent's action space physically cannot press it.

Usage:
    python probe_after_death.py --game SuperMarioBros3-Nes-v0
    python probe_after_death.py --game SuperMarioBros3-Nes-v0 --settle 240
"""
import argparse

import numpy as np
import stable_retro as retro

# Single buttons plus a couple of combos worth trying on a map screen.
CANDIDATES = [
    ["A"], ["B"], ["START"], ["SELECT"],
    ["RIGHT"], ["LEFT"], ["UP"], ["DOWN"],
    ["RIGHT", "A"], ["START", "A"],
]


def press(env, buttons, combo, frames, tap=False):
    """Step `frames` frames holding `combo` (or tapping it if tap=True).
    Returns the list of `time` values observed."""
    arr = np.array([b in combo for b in buttons], dtype=bool)
    off = np.zeros(len(buttons), dtype=bool)
    times = []
    for i in range(frames):
        # Tapping matters for menu screens that latch on button-down and would
        # otherwise ignore a held button.
        action = arr if (not tap or (i // 8) % 2 == 0) else off
        _obs, _rew, terminated, truncated, info = env.step(action)
        times.append(info.get("time"))
        if terminated or truncated:
            break
    return times


def timer_ticked(times):
    """Did the level timer count DOWN during this window? That only happens
    inside a level, so it's our 'escaped the menu' signal."""
    vals = [t for t in times if t is not None]
    return any(b < a for a, b in zip(vals, vals[1:]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True)
    p.add_argument("--state", default=None)
    p.add_argument("--settle", type=int, default=180, help="Frames to let the death transition finish before snapshotting (default: %(default)s)")
    p.add_argument("--trial", type=int, default=300, help="Frames to hold/tap each candidate (default: %(default)s = 5s)")
    p.add_argument("--max-frames", type=int, default=20000, help="Give up looking for a death after this many frames")
    args = p.parse_args()

    env = retro.make(game=args.game, state=args.state or retro.State.DEFAULT, render_mode="rgb_array")
    buttons = env.unwrapped.buttons
    print(f"Buttons: {buttons}\n")

    _obs, info = env.reset()
    prev_lives = info.get("lives")
    print(f"Starting lives: {prev_lives}")

    # Run right until something kills us -- fastest reliable way to reach the
    # post-death screen without needing a trained model.
    run_right = np.array([b in ("RIGHT", "B") for b in buttons], dtype=bool)
    died_at = None
    for frame in range(args.max_frames):
        _obs, _rew, terminated, truncated, info = env.step(run_right)
        lives = info.get("lives")
        if prev_lives is not None and lives is not None and lives < prev_lives:
            died_at = frame
            print(f"Died at frame {frame}: lives {prev_lives} -> {lives}")
            break
        prev_lives = lives
        if terminated or truncated:
            _obs, info = env.reset()
            prev_lives = info.get("lives")

    if died_at is None:
        raise SystemExit("Never died -- try raising --max-frames.")

    print(f"Letting the transition settle for {args.settle} frames ...")
    settle_times = press(env, buttons, [], args.settle)
    print(f"  info after settling: {dict(list(info.items())[:6])}")
    print(f"  timer during settle: {settle_times[0]} -> {settle_times[-1]} "
          f"(ticking: {timer_ticked(settle_times)})\n")

    snapshot = env.unwrapped.em.get_state()

    print(f"Probing each input for {args.trial} frames from the identical death state:\n")
    print(f"  {'INPUT':<16} {'HELD':>12}   {'TAPPED':>12}")
    winners = []
    for combo in CANDIDATES:
        results = {}
        for tap in (False, True):
            env.unwrapped.em.set_state(snapshot)
            times = press(env, buttons, combo, args.trial, tap=tap)
            results[tap] = timer_ticked(times)
        label = "+".join(combo)
        held = "IN LEVEL" if results[False] else "-"
        tapped = "IN LEVEL" if results[True] else "-"
        print(f"  {label:<16} {held:>12}   {tapped:>12}")
        if results[False] or results[True]:
            winners.append((label, results[False], results[True]))

    env.close()

    print()
    if winners:
        print("Inputs that got back into a level:")
        for label, held, tapped in winners:
            how = "held" if held else "tapped"
            print(f"  {label}  ({how})")
        print("\nTell Claude which one worked and the navigator will be rebuilt around it.")
    else:
        print(
            "Nothing got back into a level. Likely meanings:\n"
            "  - The screen needs an input combination not probed here.\n"
            "  - This save state starts INSIDE a level, so the post-death world\n"
            "    map may not be in a navigable state at all -- in which case\n"
            "    --on-death stop (single-life clips) is the right approach, and\n"
            "    continuous play would need restarting the level instead.\n"
            "  - It may be a game-over screen rather than a map.\n"
            "Watch the recorded MP4 around the death to see which."
        )


if __name__ == "__main__":
    main()
