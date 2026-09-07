#!/usr/bin/env python3
"""
Find out what each button on a pad actually IS, and whether it auto-fires.

Two questions this answers, both of which cost a wrong guess otherwise:

  1. WHICH RAW INDEX is the button labelled X / Y / A / B / START / SELECT?
     SDL ships a per-GUID "GameController" profile that claims to know, and for
     the pads here it is WRONG -- it reports a=b0 b=b1 x=b3 y=b4 and skips b2
     entirely, so the physical B button came out unbound. Raw indices are the
     ground truth; this prints them as you press.

  2. IS X A TURBO A, and Y A TURBO B?  NES has only two action buttons, so
     SNES-shaped pads sold for it conventionally wire the two spare buttons as
     auto-fire duplicates. Whether YOUR pad does that in hardware is measurable:
     hold the button down and watch the signal. A plain button reports one
     press and then holds steady; a turbo button toggles many times a second.
     This measures that rate, so the answer is read off the hardware rather
     than assumed.

Usage:
    python probe_pad.py               # watch every pad
    python probe_pad.py --pad 0       # just the first one
    python probe_pad.py --seconds 30  # how long to watch (default 60)

Press each button in turn, HOLDING each one for about two seconds so the
auto-fire measurement has something to measure. Ctrl+C to stop early.
"""
import argparse
import time
from collections import deque

import pygame

# A button toggling faster than this while held is auto-fire, not a human.
# A person mashing manages maybe 6-8 presses a second at the very best; pad
# turbo circuits usually run 10-30Hz. 5 sits below any turbo and above any
# thumb, and the measured rate is printed anyway so the call is checkable.
TURBO_HZ = 5.0
WINDOW_S = 1.0          # rolling window the rate is measured over
POLL_HZ = 250           # sample fast enough to actually see a 30Hz toggle


def describe_pad(joy):
    return (f"{joy.get_name()} -- {joy.get_numbuttons()} buttons, "
            f"{joy.get_numhats()} hat(s), {joy.get_numaxes()} axes")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pad", type=int, default=None,
                        help="Only watch this pad index (default: all of them)")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="How long to watch before exiting (default: %(default)s)")
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if not count:
        raise SystemExit(
            "No pad visible to SDL.\n"
            "  Under WSL a USB pad has to be attached first, from an admin\n"
            "  PowerShell:  usbipd list / usbipd bind --busid <id> /\n"
            "               usbipd attach --wsl --busid <id>\n"
            "  Then check /dev/input/js* is readable (check_gamepad.py).")

    indices = [args.pad] if args.pad is not None else list(range(count))
    pads = []
    for i in indices:
        joy = pygame.joystick.Joystick(i)
        joy.init()
        pads.append(joy)
        print(f"Pad {i}: {describe_pad(joy)}")

    # A window is not needed to poll a joystick, but SDL will not deliver
    # device events without a pumped event queue, so pump explicitly below.
    print("\nPress buttons. HOLD each for ~2s to measure auto-fire.")
    print("Ctrl+C to stop.\n")
    print(f"{'pad':>3}  {'button':>6}  {'state':>5}  {'toggles/s':>9}  verdict")
    print("-" * 52)

    # per (pad, button): timestamps of recent state changes, and last state
    edges = {}
    last = {}
    reported = {}
    deadline = time.time() + args.seconds
    try:
        while time.time() < deadline:
            pygame.event.pump()
            now = time.time()
            for pi, joy in zip(indices, pads):
                for b in range(joy.get_numbuttons()):
                    key = (pi, b)
                    state = bool(joy.get_button(b))
                    prev = last.get(key)
                    if prev is None:
                        last[key] = state
                        continue
                    if state != prev:
                        edges.setdefault(key, deque()).append(now)
                        last[key] = state
                    dq = edges.get(key)
                    if not dq:
                        continue
                    while dq and now - dq[0] > WINDOW_S:
                        dq.popleft()
                    # Two edges make one on/off cycle.
                    hz = (len(dq) / 2.0) / WINDOW_S
                    turbo = hz >= TURBO_HZ
                    # Print on a fresh press, and keep updating while turbo.
                    fresh = state and not prev
                    if fresh or (turbo and now - reported.get(key, 0) > 0.4):
                        reported[key] = now
                        verdict = (f"AUTO-FIRE (turbo) at {hz:.0f}Hz" if turbo
                                   else "plain button")
                        print(f"{pi:>3}  b{b:<5}  {'down' if state else 'up':>5}  "
                              f"{hz:>9.1f}  {verdict}")
            time.sleep(1.0 / POLL_HZ)
    except KeyboardInterrupt:
        print("\nstopped.")

    print("\nWhat to do with this:")
    print("  - The b<N> printed when you press a button IS its raw index.")
    print("  - PAD_BUTTONS / PAD_SYSTEM_BUTTONS in play_human_vs_ai.py map")
    print("    those indices to NES buttons. Current mapping:")
    try:
        from play_human_vs_ai import PAD_BUTTONS, PAD_SYSTEM_BUTTONS
        for index, name in sorted({**PAD_BUTTONS, **PAD_SYSTEM_BUTTONS}.items()):
            print(f"      b{index} -> {name}")
    except Exception as exc:                                      # noqa: BLE001
        print(f"      (could not import the tables: {exc})")
    print("  - If a button reported AUTO-FIRE, it is a turbo duplicate: map it")
    print("    to the same NES button as the one it repeats (X with A, Y with B).")


if __name__ == "__main__":
    main()
