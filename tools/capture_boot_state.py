#!/usr/bin/env python3
"""Capture a known-good boot state for a game and save it into its integration.

Cold boot (State.NONE) starts a game from power-on RAM, which is NOT
deterministic: some ROMs (TetrisTime-Nes-v0 is one) intermittently boot to a
blank, PPU-disabled GRAY screen -- the game logic runs but no video is drawn.
Standard stable-retro integrations avoid this by shipping a .state; a custom
integration made by hand does not have one.

This cold-boots the game, steps a few seconds so the title screen renders,
and -- only if the frame is genuinely non-blank (pixel std above a floor) --
saves the emulator state to integrations/<game>/<name>.state (gzip, the format
retro.make(state="<name>") loads). One cold boot per process (stable-retro
allows one emulator per process); re-run until it reports SAVED. A generated
state is ROM-derived, so it is kept local and gitignored like the ROM.

    python tools/capture_boot_state.py --game TetrisTime-Nes-v0
    # repeat until it prints SAVED (each run is a fresh power-on)
"""
import argparse
import gzip
import os
import sys

import numpy as np
import stable_retro as retro

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import custom_integrations
    custom_integrations.register()
except Exception:                                                # noqa: BLE001
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True, help="Game id, e.g. TetrisTime-Nes-v0")
    p.add_argument("--name", default="boot", help="State name -> <name>.state (default: %(default)s)")
    p.add_argument("--frames", type=int, default=240, help="No-op frames to step so the title renders before capture (default: %(default)s)")
    p.add_argument("--min-std", type=float, default=20.0, help="Reject the boot if the frame's pixel std is below this -- that is the blank gray boot (default: %(default)s)")
    p.add_argument("--out", default=None, help="Output path (default: integrations/<game>/<name>.state)")
    args = p.parse_args()

    out = args.out or os.path.join(custom_integrations.INTEGRATIONS_DIR,
                                   args.game, f"{args.name}.state")

    env = retro.make(game=args.game, state=retro.State.NONE,
                     use_restricted_actions=retro.Actions.ALL,
                     render_mode="rgb_array")
    # Same cold-boot fixups play_engine uses so reset() succeeds with no state.
    if getattr(env.unwrapped, "statename", None) is None:
        env.unwrapped.statename = args.name
    if not getattr(env.unwrapped, "initial_state", None):
        env.unwrapped.initial_state = env.unwrapped.em.get_state()

    obs, _info = env.reset()
    noop = [False] * env.num_buttons
    for _ in range(args.frames):
        obs, _r, term, trunc, _i = env.step(noop)
        if term or trunc:
            break

    std = float(np.std(obs))
    mean = float(np.mean(obs))
    if std < args.min_std:
        env.close()
        print(f"BLANK boot (mean={mean:.1f} std={std:.1f} < {args.min_std}). "
              f"This is the gray power-on. Re-run -- next power-on differs.")
        sys.exit(1)

    state_bytes = env.unwrapped.em.get_state()
    env.close()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with gzip.open(out, "wb") as f:
        f.write(state_bytes)
    print(f"SAVED {out}  (boot frame mean={mean:.1f} std={std:.1f}, "
          f"{len(state_bytes)} state bytes). "
          f"play now boots from this every run instead of gambling on power-on.")


if __name__ == "__main__":
    main()
