#!/usr/bin/env python3
"""
The single, complete gamepad diagnostic. Run it top to bottom and read down to
the first FAIL.

It works in three phases, because a pad that "does nothing in the game" can
fail at three very different layers and they look identical from the outside:

  PHASE 0  OS / WSL permissions -- can this process even READ the device nodes?
           This is checked BEFORE pygame starts, because SDL reads the
           /dev/input/event* nodes and silently enumerates NOTHING when it
           cannot -- which reads as "SDL sees no pad" when the real fault is a
           file permission. This is the layer that actually bit this project:
           the event nodes came up crw-rw---- (group only), unreadable, and the
           controllers went dead while the keyboard kept working.

  PHASE 1  SDL / pygame -- does the joystick subsystem enumerate the pad once
           permissions are clean?

  PHASE 2  Live probe -- press buttons and read off the RAW INDEX of each, the
           d-pad hat, the analogue axes, and whether a button auto-fires
           (turbo). This is what tells you how to fill PAD_BUTTONS /
           PAD_SYSTEM_BUTTONS in play_engine.py.

Usage:
    python tools/probe_pad.py                # full check, then live probe
    python tools/probe_pad.py --no-probe     # phases 0 and 1 only, then exit
    python tools/probe_pad.py --pad 0        # live-probe only this pad
    python tools/probe_pad.py --seconds 30   # how long to watch (default 60)

Under WSL, controllers must be attached from an admin PowerShell first:
    usbipd list
    usbipd bind   --busid <id>          (once per device)
    usbipd attach --wsl --busid <id>    (every replug and every wsl --shutdown)
"""
import argparse
import glob
import os
import platform
import subprocess
import time
from collections import deque

# A button toggling faster than this while held is auto-fire, not a human. A
# person mashing manages maybe 6-8 presses a second at the very best; pad turbo
# circuits usually run 10-30Hz. 5 sits below any turbo and above any thumb, and
# the measured rate is printed anyway so the call is always checkable.
TURBO_HZ = 5.0
WINDOW_S = 1.0          # rolling window the toggle rate is measured over
POLL_HZ = 250           # sample fast enough to actually see a 30Hz toggle


def ok(label, detail=""):
    print("  [OK]   %-32s %s" % (label, detail))


def warn(label, detail=""):
    print("  [WARN] %-32s %s" % (label, detail))


def bad(label, detail=""):
    print("  [FAIL] %-32s %s" % (label, detail))


# ---------------------------------------------------------------- phase 0 --
def is_wsl():
    try:
        with open("/proc/version") as fh:
            v = fh.read().lower()
        return "microsoft" in v or "wsl" in v
    except OSError:
        return False


def phase0_permissions():
    """OS-level audit BEFORE pygame is touched. Returns (has_nodes, readable)."""
    print("PHASE 0 -- OS / WSL permissions (can this process READ the pads?)")

    if platform.system() != "Linux":
        print("  Not Linux (%s) -- SDL talks to the OS driver directly here, so"
              % platform.system())
        print("  the /dev/input permission checks below do not apply. Skipping to SDL.")
        return None, None

    if is_wsl():
        ok("environment", "WSL -- pads must be attached with usbipd (see header)")
    else:
        ok("environment", "native Linux")

    if not os.path.isdir("/dev/input"):
        bad("/dev/input", "directory does not exist")
        print("        The kernel exposes no input layer at all. Under WSL2 the")
        print("        stock kernel usually has it; if not, the device is simply")
        print("        not attached. Attach it (see the header), then re-run.")
        return False, False

    events = sorted(glob.glob("/dev/input/event*"))
    joys = sorted(glob.glob("/dev/input/js*"))
    if joys:
        ok("joystick nodes", " ".join(joys))
    if events:
        ok("event nodes", " ".join(events))
    if not events and not joys:
        bad("no device nodes", "/dev/input has no event*/js* nodes")
        print("        Nothing is attached to WSL. From an ADMIN PowerShell:")
        print("          usbipd list")
        print("          usbipd bind   --busid <id>          (once)")
        print("          usbipd attach --wsl --busid <id>    (every replug)")
        return False, False

    # SDL enumerates joysticks from the EVENT nodes, so their readability is the
    # one that actually decides whether a pad works. js* readability alone is
    # NOT enough -- that distinction is exactly what hid the bug here.
    all_nodes = events + joys
    unreadable = [n for n in all_nodes if not os.access(n, os.R_OK)]
    bad_events = [n for n in events if not os.access(n, os.R_OK)]

    # group membership, for the durable fix
    try:
        import grp
        input_gid = grp.getgrnam("input").gr_gid
        in_input_group = input_gid in os.getgroups()
    except (ImportError, KeyError):
        in_input_group = None

    if unreadable:
        bad("not readable", " ".join(unreadable[:4]))
        if bad_events:
            print("        ^ THESE are the ones SDL reads. Unreadable event nodes")
            print("          are why a pad enumerates in the OS but goes dead in")
            print("          SDL. Fix now (lost on replug / wsl --shutdown):")
            print("            sudo chmod a+r /dev/input/event* /dev/input/js*")
        if in_input_group is False:
            print("        Durable fix -- join the 'input' group, then restart WSL:")
            print("            sudo usermod -aG input $USER")
            print("            (from Windows)  wsl --shutdown   then reopen")
        else:
            print("        You ARE in the 'input' group, yet the nodes are not")
            print("        group-readable -- so a udev rule is the durable fix:")
            print("            echo 'KERNEL==\"event*|js*\", MODE=\"0664\", GROUP=\"input\"' \\")
            print("              | sudo tee /etc/udev/rules.d/99-input.rules")
            print("            sudo udevadm control --reload && sudo udevadm trigger")
        return True, False

    ok("permissions", "all %d node(s) readable by this user" % len(all_nodes))
    if in_input_group is False:
        warn("input group", "not a member (nodes readable anyway, but joining is")
        print("        the durable fix if they ever revert): sudo usermod -aG input $USER")
    return True, True


# ---------------------------------------------------------------- phase 1 --
def phase1_sdl():
    """Bring up pygame's joystick subsystem. Returns a list of Joystick, or []."""
    print("\nPHASE 1 -- SDL / pygame enumeration")
    try:
        import pygame
    except ImportError:
        bad("pygame", "not installed -- pip install pygame")
        return []
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if not count:
        bad("SDL joysticks", "pygame enumerates 0")
        print("        If Phase 0 was clean, SDL can read the nodes but does not")
        print("        recognise this pad. Things to try:")
        print("          - a mode switch on the pad (many retro pads have an")
        print("            XInput/DInput/Switch toggle held at power-on)")
        print("          - SDL_JOYSTICK_DEVICE=/dev/input/js0 python tools/probe_pad.py")
        print("          - confirm the physical USB end is actually plugged in")
        return []
    pads = []
    for i in range(count):
        joy = pygame.joystick.Joystick(i)
        joy.init()
        pads.append(joy)
        ok("SDL joystick %d" % i,
           "%s -- %d buttons, %d hat(s), %d axes"
           % (joy.get_name(), joy.get_numbuttons(), joy.get_numhats(),
              joy.get_numaxes()))
    return pads


# ---------------------------------------------------------------- phase 2 --
def phase2_live(pads, indices, seconds):
    """Live probe: raw index, hat, axes, and auto-fire rate as buttons are held."""
    import pygame
    print("\nPHASE 2 -- live probe")
    print("Press each button, HOLDING each for ~2s so auto-fire can be measured.")
    print("Move the d-pad and the stick too. Ctrl+C to stop.\n")
    print("  %3s  %6s  %5s  %9s  %s" % ("pad", "button", "state", "toggles/s", "verdict"))
    print("  " + "-" * 50)

    edges, last, reported = {}, {}, {}
    axis_seen = set()
    deadline = time.time() + seconds
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
                    hz = (len(dq) / 2.0) / WINDOW_S          # two edges = one cycle
                    turbo = hz >= TURBO_HZ
                    fresh = state and not prev
                    if fresh or (turbo and now - reported.get(key, 0) > 0.4):
                        reported[key] = now
                        verdict = ("AUTO-FIRE (turbo) at %.0fHz" % hz if turbo
                                   else "plain button")
                        print("  %3d  b%-5d  %5s  %9.1f  %s"
                              % (pi, b, "down" if state else "up", hz, verdict))
                # hat + axes, reported on change so movement is visible
                hat = joy.get_hat(0) if joy.get_numhats() else (0, 0)
                if hat != (0, 0) and reported.get((pi, "hat")) != hat:
                    reported[(pi, "hat")] = hat
                    print("  %3d  hat     %5s  %9s  d-pad %s" % (pi, "-", "-", hat))
                for a in range(min(4, joy.get_numaxes())):
                    v = joy.get_axis(a)
                    if abs(v) > 0.5 and (pi, a) not in axis_seen:
                        axis_seen.add((pi, a))
                        print("  %3d  axis%-2d  %5s  %9s  stick %+.2f" % (pi, a, "-", "-", v))
                    elif abs(v) <= 0.3:
                        axis_seen.discard((pi, a))
            time.sleep(1.0 / POLL_HZ)
    except KeyboardInterrupt:
        print("\nstopped.")

    print("\nWhat to do with this:")
    print("  - The b<N> printed when you press a button IS its raw index.")
    print("  - PAD_BUTTONS / PAD_SYSTEM_BUTTONS in play_engine.py map those")
    print("    indices to NES buttons. Current mapping:")
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from play_engine import PAD_BUTTONS, PAD_SYSTEM_BUTTONS
        for index, name in sorted({**PAD_BUTTONS, **PAD_SYSTEM_BUTTONS}.items()):
            print("      b%-2d -> %s" % (index, name))
    except Exception as exc:                                      # noqa: BLE001
        print("      (could not import the tables: %s)" % exc)
    print("  - A button that reported AUTO-FIRE is a turbo duplicate: map it to")
    print("    the same NES button as the one it repeats (X with A, Y with B).")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pad", type=int, default=None,
                        help="Live-probe only this pad index (default: all)")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="How long the live probe watches (default: %(default)s)")
    parser.add_argument("--no-probe", action="store_true",
                        help="Run the permission + SDL checks only, then exit")
    args = parser.parse_args()

    has_nodes, readable = phase0_permissions()
    if readable is False:
        print("\nStop at Phase 0: fix the permission above, then re-run. SDL cannot")
        print("enumerate a pad whose event node it cannot read.")
        return 1

    pads = phase1_sdl()
    if not pads:
        return 1
    if args.no_probe:
        print("\nPads are reachable. Play with:  python studio.py --game <id>")
        return 0

    indices = [args.pad] if args.pad is not None else list(range(len(pads)))
    pads = [pads[i] for i in indices] if args.pad is not None else pads
    phase2_live(pads, indices, args.seconds)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
