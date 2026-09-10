#!/usr/bin/env python3
"""
Read a game's verified values out of games.json, whatever shape they take.

games.json is the single source of truth for addresses: every entry there was
checked frame-by-frame against the screen, and its `variables` block already
uses several different encodings because different games store things
differently. This turns any of them into a number, so nothing above needs to
know which game it is looking at.

Sources currently in use, all handled here:

    info      the integration publishes it in env.step()'s info dict
    info16    low byte from info, high byte from RAM (SMB3's level position:
              hpos wraps at 256 and a page byte counts the wraps)
    ram       one plain byte
    ram_bcd3  three BCD bytes, hundreds/tens/units (SMB3's timer)
    tiles     a run of display digits, one per byte, byte = digit + offset,
              most significant first (TetrisTime's whole HUD)
    unknown   deliberately not found yet -- reads as None so the shaping that
              needs it stays inert instead of rewarding garbage

Adding a game means adding its addresses to games.json, not editing code.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_JSON = os.path.join(ROOT, "games.json")


_CONFIG_PATH = [None]


def set_config_path(path):
    """Point every lookup at a different config file (train.py --config)."""
    _CONFIG_PATH[0] = path


def load_entry(game, path=None):
    """The whole games.json record for one game ({} if it has none)."""
    path = path or _CONFIG_PATH[0] or GAMES_JSON
    try:
        with open(path, encoding="utf-8") as fh:
            return (json.load(fh).get("games") or {}).get(game) or {}
    except Exception as exc:                                     # noqa: BLE001
        print("Could not read %s: %s" % (path, exc))
        return {}


def _as_int(value):
    if value is None:
        return None
    return int(str(value), 0) if isinstance(value, str) else int(value)


class GameVars:
    """Every named value for one game, readable from (ram, info).

    Variables may live under several sections -- `variables` for a
    single-player game, `two_player` where each player has their own copy --
    and later sections win, so a two-player entry can shadow a generic one.
    """

    def __init__(self, game, entry=None, sections=("variables", "two_player")):
        self.game = game
        self.entry = entry if entry is not None else load_entry(game)
        self.spec = {}
        for section in sections:
            for name, s in (self.entry.get(section) or {}).items():
                if isinstance(s, dict):
                    self.spec[name] = s

    def __contains__(self, name):
        return name in self.spec

    def names(self):
        return sorted(self.spec)

    def address(self, name):
        s = self.spec.get(name) or {}
        if s.get("source") in (None, "unknown"):
            return None
        return _as_int(s.get("address"))

    def read(self, name, ram=None, info=None, default=None):
        """One variable as a number, or `default` when it cannot be read.

        Never raises on a missing address or a bad tile: an unreadable value
        must leave its reward term inert rather than feed the agent noise.
        """
        s = self.spec.get(name)
        if not s:
            return default
        source = s.get("source")
        info = info or {}

        if source == "info":
            return info.get(s.get("key", name), default)

        if source == "info16":
            low = info.get(s.get("key", "screen_x"))
            high_addr = _as_int(s.get("address_high"))
            if low is None or high_addr is None or ram is None:
                return default
            return int(low) + (int(ram[high_addr]) << 8)

        base = _as_int(s.get("address"))
        if base is None or ram is None:
            return default

        if source == "ram":
            return int(ram[base])

        if source == "ram_bcd3":
            try:
                return (int(ram[base]) * 100 + int(ram[base + 1]) * 10
                        + int(ram[base + 2]))
            except Exception:                                    # noqa: BLE001
                return default

        if source == "tiles":
            off = int(s.get("digit_offset", 48))
            digits = []
            for i in range(int(s.get("length", 1))):
                d = int(ram[base + i]) - off
                # A blank or garbage tile reads as 0 rather than poisoning the
                # whole number; these HUDs pad with '0' tiles anyway.
                digits.append(d if 0 <= d <= 9 else 0)
            return int("".join(str(d) for d in digits))

        return default

    def matches(self, name, ram=None, info=None, equals=None, default=False):
        """Whether a variable currently equals a value -- the shape used for
        'is this player alive' and 'is an animation playing'."""
        value = self.read(name, ram, info, default=None)
        if value is None:
            return default
        return int(value) == int(equals)
