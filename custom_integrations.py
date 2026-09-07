#!/usr/bin/env python3
"""
Make the repo's own game integrations visible to stable-retro.

stable-retro only ships integrations for games it has upstream, and never the
ROMs. A game it has no integration for at all -- NES Tetris, for one -- cannot
be imported OR played until an integration exists: a folder with rom.sha,
data.json, scenario.json and metadata.json describing the ROM and its RAM.

Those folders live in ./integrations/<Game>-<System>-v0/ in this repo, tracked
in git (the ROM itself is NOT -- you drop your own rom.<ext> in beside them).
retro.data.add_custom_integration() registers that directory AND flips custom
integrations into the DEFAULT search, so an ordinary retro.make(game) finds
them with no per-call changes. It has to be called once per PROCESS, so every
entry point that touches stable-retro calls register() -- studio.py and
play_engine share a process, but render_bk2 runs as a subprocess and needs its
own call.
"""
import os

INTEGRATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "integrations")

_registered = False


def register():
    """Register ./integrations with stable-retro. Idempotent and non-fatal."""
    global _registered
    if _registered or not os.path.isdir(INTEGRATIONS_DIR):
        return
    try:
        import stable_retro as retro
        retro.data.add_custom_integration(INTEGRATIONS_DIR)
        _registered = True
    except Exception as exc:                                      # noqa: BLE001
        print(f"(custom integrations not registered: {exc})")


if __name__ == "__main__":
    register()
    import stable_retro as retro
    here = [g for g in retro.data.list_games()
            if os.path.isdir(os.path.join(INTEGRATIONS_DIR, g))]
    print("Custom integrations registered from:", INTEGRATIONS_DIR)
    for g in sorted(here):
        rom = any(f.startswith("rom.") and not f.endswith(".sha")
                  for f in os.listdir(os.path.join(INTEGRATIONS_DIR, g)))
        print("  %-24s %s" % (g, "ROM present" if rom else "NO ROM -- drop rom.<ext> in"))
    if not here:
        print("  (none found)")
