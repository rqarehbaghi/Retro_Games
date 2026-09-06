#!/usr/bin/env python3
"""
Find out why a gamepad is not reaching the game, one layer at a time.

A pad has to clear four separate hurdles before the emulator sees it, and each
one fails silently:

    1. USB    the device is attached to this machine at all. Under WSL2 that
              means usbipd-win has handed it over from Windows.
    2. KERNEL Linux exposes it as an input device -- /dev/input/js* or
              event*. WSL2's stock kernel does not always include the joystick
              drivers, and this is the hurdle that catches people, because
              lsusb shows the device while nothing appears under /dev/input.
    3. PERMS  your user can read that node. A device owned by root with no
              group access is invisible to SDL without a word of complaint.
    4. SDL    pygame enumerates it. If 1-3 pass and this does not, the pad is
              an unusual one rather than the setup being wrong.

Run it and read down to the first FAIL.

    python check_gamepad.py
    python check_gamepad.py --watch     # then press buttons to see them arrive
"""
import argparse
import glob
import os
import subprocess
import sys


def ok(label, detail=""):
    print("  PASS  %-34s %s" % (label, detail))


def bad(label, detail=""):
    print("  FAIL  %-34s %s" % (label, detail))


def step_usb():
    print("1. USB -- is the device attached to this machine?")
    try:
        out = subprocess.run(["lsusb"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except FileNotFoundError:
        print("        (lsusb not installed: sudo apt install usbutils)")
        return None
    except Exception as exc:
        print("        (could not run lsusb: %s)" % exc)
        return None
    likely = [l for l in out.splitlines()
              if any(w in l.lower() for w in
                     ("game", "pad", "joystick", "controller", "8bitdo",
                      "xbox", "sony", "nintendo", "logitech"))]
    if likely:
        ok("a controller is on the bus", likely[0].split(": ", 1)[-1])
    else:
        bad("no obvious controller in lsusb")
        print("        Attach it from an ADMIN PowerShell on Windows:")
        print("          usbipd list")
        print("          usbipd bind   --busid <id>     (once)")
        print("          usbipd attach --wsl --busid <id>   (every replug)")
        print("        Devices seen right now:")
        for line in out.splitlines():
            print("          " + line)
    return bool(likely)


def step_kernel():
    print("\n2. KERNEL -- is there an input device node?")
    js = sorted(glob.glob("/dev/input/js*"))
    ev = sorted(glob.glob("/dev/input/event*"))
    if js:
        ok("joystick node", " ".join(js))
    if ev:
        ok("event node(s)", "%d found" % len(ev))
    if not js and not ev:
        bad("nothing under /dev/input")
        print("        The device can be on the USB bus and still have no input")
        print("        node: WSL2's stock kernel often omits the joydev and")
        print("        hid drivers. lsusb passing while this fails is the")
        print("        signature. Options, in order of effort:")
        print("          - run the game on Windows rather than WSL")
        print("          - build a WSL kernel with CONFIG_INPUT_JOYDEV=y")
        print("          - use the keyboard (no --gamepad)")
    return bool(js or ev)


def step_perms():
    print("\n3. PERMISSIONS -- can this user read it?")
    nodes = sorted(glob.glob("/dev/input/js*")) + sorted(glob.glob("/dev/input/event*"))
    if not nodes:
        print("        (skipped, no nodes)")
        return None
    unreadable = [n for n in nodes if not os.access(n, os.R_OK)]
    if unreadable:
        bad("not readable", unreadable[0])
        print("        sudo usermod -aG input $USER   then log out and back in")
        print("        or, to test right now:  sudo chmod a+r %s" % unreadable[0])
        return False
    ok("readable", "%d node(s)" % len(nodes))
    return True


def step_sdl(watch=False):
    print("\n4. SDL -- does pygame enumerate it?")
    try:
        import pygame
    except ImportError:
        bad("pygame not installed", "pip install pygame")
        return False
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if not count:
        bad("pygame sees no joystick")
        print("        If 1-3 passed, SDL is not recognising this particular")
        print("        pad. Try a different mode on it -- many retro pads have")
        print("        an XInput/DInput/Switch toggle held at power-on.")
        return False
    pads = []
    for i in range(count):
        pad = pygame.joystick.Joystick(i)
        pad.init()
        pads.append(pad)
        ok("pygame joystick %d" % i,
           "%s -- %d buttons, %d hat(s), %d axes"
           % (pad.get_name(), pad.get_numbuttons(), pad.get_numhats(),
              pad.get_numaxes()))
    if watch:
        print("\nPress buttons and move the stick. Ctrl+C to stop.")
        pygame.display.set_mode((240, 80))
        try:
            while True:
                pygame.event.pump()
                for pad in pads:
                    held = [b for b in range(pad.get_numbuttons()) if pad.get_button(b)]
                    hat = pad.get_hat(0) if pad.get_numhats() else (0, 0)
                    axes = [round(pad.get_axis(a), 2) for a in range(min(2, pad.get_numaxes()))]
                    if held or hat != (0, 0) or any(abs(a) > 0.5 for a in axes):
                        print("    buttons %-16s hat %-8s axes %s"
                              % (held, hat, axes))
                pygame.time.wait(120)
        except KeyboardInterrupt:
            print("\nstopped.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", action="store_true",
                        help="After the checks, print button presses live so the mapping can be read off")
    args = parser.parse_args()

    print("Checking the path from USB socket to emulator.\n")
    step_usb()
    node = step_kernel()
    step_perms()
    seen = step_sdl(args.watch)

    print()
    if seen:
        print("The pad is reachable. Play with it:")
        print("  python studio.py --game <id> --gamepad")
        print("Buttons coming out wrong? --watch shows the numbers, and")
        print("PAD_BUTTONS in play_human_vs_ai.py maps them.")
    elif node:
        print("Linux can see an input device but SDL cannot use it.")
    else:
        print("Nothing usable. Read up from the first FAIL above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
