#!/usr/bin/env python3
"""
The repo's first tests. Pure logic only -- no emulator, no ROM, no GPU.

    python tests/test_core.py

Every case here is a bug that was actually shipped, or a rule that was actually
relied on. They run in about a second and need nothing installed beyond numpy,
so there is no excuse for not running them before a push.

What is deliberately NOT here: anything needing a ROM. Those are copyrighted and
not in the repo, so an emulator test would fail for everyone who clones it. The
measurement work that needs a running game -- hole counting against a real
board, the piece mask, settle timing -- is verified by rendering frames and
looking, which is the house rule and cannot be automated away.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FAILURES = []


def check(label, got, want):
    ok = got == want
    print("  %-58s %s" % (label, "ok" if ok else "FAIL"))
    if not ok:
        print("      got  %r" % (got,))
        print("      want %r" % (want,))
        FAILURES.append(label)


# ------------------------------------------------------------ reward terms ---
def test_reward_terms():
    from rl.env import RewardModel
    print("reward term kinds")

    class FakeVars:
        def __init__(self):
            self.v = {}

        def read(self, name, ram=None, info=None, default=None):
            return self.v.get(name, default)

    fv, feats = FakeVars(), {}

    def hook(vars, ram, info, player, frame=None, spec=None):
        return dict(feats)

    def run(terms, steps):
        rm = RewardModel(terms, fv, hook, 2)
        fv.v.update(steps[0][0]); feats.clear(); feats.update(steps[0][1])
        rm.reset(None, {})
        out = []
        for vals, f, term in steps[1:]:
            fv.v.update(vals); feats.clear(); feats.update(f)
            r, _ = rm.step(None, {}, term)
            out.append(round(r, 4))
        return out

    check("delta pays a rise",
          run([{"kind": "delta", "var": "x", "scale": 1.0}],
              [({"x": 0}, {}, False), ({"x": 5}, {}, False)]), [5.0])
    check("delta ignores a fall by default",
          run([{"kind": "delta", "var": "x", "scale": 1.0}],
              [({"x": 5}, {}, False), ({"x": 0}, {}, False)]), [0.0])
    check("delta with positive_only false charges a fall",
          run([{"kind": "delta", "var": "x", "scale": 1.0, "positive_only": False}],
              [({"x": 5}, {}, False), ({"x": 0}, {}, False)]), [-5.0])
    check("delta_tiered pays by SIZE of the jump, not per unit",
          run([{"kind": "delta_tiered", "var": "x", "tiers": [0, 10, 30, 60]}],
              [({"x": 0}, {}, False), ({"x": 2}, {}, False)]), [30.0])
    check("delta_tiered clamps a jump past the table",
          run([{"kind": "delta_tiered", "var": "x", "tiers": [0, 10, 30, 60]}],
              [({"x": 0}, {}, False), ({"x": 9}, {}, False)]), [60.0])
    check("event pays once, the step it BECOMES the value",
          run([{"kind": "event", "var": "x", "equals": 7, "scale": 100.0}],
              [({"x": 0}, {}, False), ({"x": 7}, {}, False), ({"x": 7}, {}, False)]),
          [100.0, 0.0])
    check("decrease_event pays when a counter drops",
          run([{"kind": "decrease_event", "var": "x", "scale": 3.0}],
              [({"x": 9}, {}, False), ({"x": 2}, {}, False), ({"x": 4}, {}, False)]),
          [3.0, 0.0])
    check("feature_delta pays the change",
          run([{"kind": "feature_delta", "name": "h", "scale": -0.5}],
              [({}, {"h": 2}, False), ({}, {"h": 6}, False)]), [-2.0])
    check("feature_level pays the level every step",
          run([{"kind": "feature_level", "name": "h", "scale": -0.5}],
              [({}, {"h": 4}, False), ({}, {"h": 4}, False), ({}, {"h": 6}, False)]),
          [-2.0, -3.0])
    check("terminal pays only on the ending step",
          run([{"kind": "terminal", "scale": -20.0}],
              [({}, {}, False), ({}, {}, False), ({}, {}, True)]), [0.0, -20.0])
    # The game-over frame is an ANIMATION, not a board the agent played.
    # Diffing across it once paid +53.7 for dying against a configured -20.
    check("feature terms are skipped on the terminating step",
          run([{"kind": "feature_delta", "name": "h", "scale": -1.0},
               {"kind": "terminal", "scale": -5.0}],
              [({}, {"h": 0}, True and False), ({}, {"h": 100}, True)]), [-5.0])
    check("an unreadable variable leaves its term inert",
          run([{"kind": "delta", "var": "missing", "scale": 1.0},
               {"kind": "step", "scale": 0.5}],
              [({}, {}, False), ({}, {}, False)]), [0.5])


# --------------------------------------------------------- player plumbing ---
def test_player_substitution():
    from rl.env import _fill_player, _for_player
    print("per-player config")
    check("{player} is replaced through nested structures",
          _fill_player({"a": "game_over_p{player}",
                        "b": [{"var": "lines_p{player}"}]}, 1),
          {"a": "game_over_p1", "b": [{"var": "lines_p1"}]})
    check("config without {player} is untouched",
          _fill_player({"a": "score", "n": 5}, 2), {"a": "score", "n": 5})
    check("per_player merges over the shared values",
          _for_player({"y": 56, "cell": 8,
                       "per_player": {"1": {"x": 9}, "2": {"x": 153}}}, 2),
          {"y": 56, "cell": 8, "x": 153})
    check("per_player for the other player",
          _for_player({"y": 56, "per_player": {"1": {"x": 9}, "2": {"x": 153}}}, 1),
          {"y": 56, "x": 9})
    check("a block with no per_player is returned as-is",
          _for_player({"x": 1}, 2), {"x": 1})


# ------------------------------------------------------- macro placement ---
def test_macro_plan():
    from rl.macro import MacroPlanGenerator as M
    print("macro placement planning")
    cfg = {"rotations": 4, "columns": 10, "tap_hold": 2, "tap_release": 2}

    def taps_of(plan, button):
        return sum(1 for f in plan["frames"] if f == [button]) // 2

    # Measured: one tap of A runs the counter 0 -> 3 -> 2 -> 1 -> 0, so it
    # DECREMENTS. Asking for rotation 1 from 0 needs THREE taps, not one.
    for cur in (0, 1, 2, 3):
        for want in (0, 1, 2, 3):
            p = M.plan(want * 10 + 4, cfg, current_rot=cur, current_col=4)
            n = taps_of(p, "A")
            check("rotation %d -> %d lands on target" % (cur, want),
                  (cur - n) % 4, want)
    p = M.plan(0 * 10 + 0, cfg, current_rot=0, current_col=4)
    check("moving left 4 columns taps LEFT four times",
          taps_of(p, "LEFT"), 4)
    check("and never taps RIGHT", taps_of(p, "RIGHT"), 0)
    p = M.plan(0 * 10 + 9, cfg, current_rot=0, current_col=4)
    check("moving right 5 columns taps RIGHT five times",
          taps_of(p, "RIGHT"), 5)
    p = M.plan(0 * 10 + 4, cfg, current_rot=0, current_col=4)
    check("already in place plans no frames at all", p["frames"], [])


# ------------------------------------------------------------ board maths ---
def test_board_features():
    from rl.features import board_features
    print("board features")
    g = np.zeros((20, 10), np.uint8)
    check("an empty board has no holes and no height",
          (board_features(g)["holes"], board_features(g)["height"]), (0.0, 0.0))
    g = np.zeros((20, 10), np.uint8)
    g[19, :] = 1
    check("one full bottom row is height 10, no holes",
          (board_features(g)["holes"], board_features(g)["height"]), (0.0, 10.0))
    g = np.zeros((20, 10), np.uint8)
    g[18, 0] = 1                      # a block with a gap underneath
    check("a covered gap is exactly one hole",
          board_features(g)["holes"], 1.0)
    g = np.zeros((20, 10), np.uint8)
    g[15, 3] = 1
    check("a block four rows up covers four holes",
          board_features(g)["holes"], 4.0)
    g = np.zeros((20, 10), np.uint8)
    g[19, 0] = 1
    g[19, 1] = 1
    check("open space ABOVE the stack is not a hole",
          board_features(g)["holes"], 0.0)
    g = np.zeros((20, 10), np.uint8)
    g[19, 0] = 1; g[17, 1] = 1
    # heights are 1, 3, then eight zeros, so the summed steps are 2 + 3.
    # Note what that means: the drop from the stack to an EMPTY column counts
    # as bumpiness, so a tall stack beside untouched columns scores as jagged.
    # It is the standard definition and the weight on it is tiny (-0.02, worth
    # -0.00 a placement measured), but it is not "roughness of the surface".
    check("bumpiness sums every step, including the drop to empty columns",
          board_features(g)["bumpiness"], 5.0)
    g = np.zeros((20, 10), np.uint8)
    g[19, :] = 1
    check("a flat board has no bumpiness", board_features(g)["bumpiness"], 0.0)


# ------------------------------------------------------------- recordings ---
def test_find_new_bk2():
    import tempfile, time
    from recording import find_new_bk2
    print("recording discovery")
    with tempfile.TemporaryDirectory() as d:
        old = os.path.join(d, "old.bk2")
        open(old, "w").close()
        os.utime(old, (1000, 1000))
        before = {old}
        check("a stale file from an earlier session is not this run's",
              find_new_bk2(d, before, started_at=time.time()), None)
        new = os.path.join(d, "new.bk2")
        open(new, "w").close()
        check("a brand new file is found", find_new_bk2(d, before, started_at=0), new)
        # stable-retro reuses -000000 numbering, so a rerun OVERWRITES rather
        # than creating a new name -- membership alone would miss it.
        started = time.time()
        os.utime(old, (started + 1, started + 1))
        check("a REWRITTEN pre-existing file counts as this session's",
              find_new_bk2(d, {old}, started_at=started), old)


def main():
    for fn in (test_reward_terms, test_player_substitution, test_macro_plan,
               test_board_features, test_find_new_bk2):
        fn()
        print("")
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("   %s" % f)
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
