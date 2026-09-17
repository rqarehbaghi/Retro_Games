"""The skip rule is judged in ONE place, TransitionGate, for training and play.

No ROM needed: a two-byte RAM stands in for a game with a mode byte (0 while
running, anything else while frozen) and a per-player alive flag.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.env import TransitionGate, episode_ended   # noqa: E402
from rl.vars import GameVars                        # noqa: E402

MODE, ALIVE = 0, 1
ENTRY = {"variables": {
    "game_mode": {"source": "ram", "address": MODE},
    "game_over_p2": {"source": "ram", "address": ALIVE},
}}
SKIP = {"var": "game_mode", "not_equals": 0, "max_frames": 5,
        "input_script": [{"start": 0, "hold": 1, "repeat_every": 2, "buttons": ["B"]}]}
END = {"var": "game_over_p2", "equals": 0}


def ram(mode, alive=1):
    r = np.zeros(4, dtype=np.uint8)
    r[MODE], r[ALIVE] = mode, alive
    return r


class MatchesTests(unittest.TestCase):
    def setUp(self):
        self.vars = GameVars("toy", entry=ENTRY)

    def test_not_equals_treats_every_other_value_alike(self):
        for mode, want in ((0, False), (1, True), (3, True), (249, True)):
            self.assertEqual(self.vars.matches_config(SKIP, ram(mode)), want, mode)

    def test_equals_still_works_and_missing_block_is_default(self):
        self.assertTrue(self.vars.matches_config(END, ram(0, alive=0)))
        self.assertFalse(self.vars.matches_config(END, ram(0, alive=1)))
        self.assertFalse(self.vars.matches_config(None, ram(3)))
        self.assertTrue(episode_ended(END, self.vars, ram(0, alive=0), {}))
        self.assertFalse(episode_ended(None, self.vars, ram(0, alive=0), {}))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.vars = GameVars("toy", entry=ENTRY)

    def test_skips_exactly_while_frozen_and_hands_back_on_the_running_frame(self):
        gate = TransitionGate(SKIP, self.vars, episode_end=END, time_limit=False)
        modes = [0, 0, 3, 3, 3, 0, 0, 1, 1, 0]
        held = [gate.update(ram(m), {}) for m in modes]
        self.assertEqual(held, [m != 0 for m in modes])

    def test_script_restarts_for_each_stretch(self):
        gate = TransitionGate(SKIP, self.vars, time_limit=False)
        out = []
        for m in (3, 3, 3, 0, 1, 1):
            if gate.update(ram(m), {}):
                out.append(tuple(gate.buttons()))
        self.assertEqual(out, [("B",), (), ("B",), ("B",), ()])

    def test_stands_down_once_the_agents_game_is_over(self):
        # A mode byte can read non-zero on the game-over screen; driving the
        # script into it (or holding a dead agent's step) is wrong.
        gate = TransitionGate(SKIP, self.vars, episode_end=END, time_limit=False)
        self.assertTrue(gate.update(ram(3, alive=1), {}))
        self.assertFalse(gate.update(ram(249, alive=0), {}))
        self.assertFalse(gate.active(ram(249, alive=0), {}))

    def test_training_times_out_but_play_waits_out_any_pause(self):
        train = TransitionGate(SKIP, self.vars, time_limit=True)
        play = TransitionGate(SKIP, self.vars, time_limit=False)
        t = [train.update(ram(1), {}) for _ in range(50)]
        p = [play.update(ram(1), {}) for _ in range(50)]
        self.assertEqual(sum(t), 5)
        self.assertTrue(train.timed_out)
        self.assertTrue(all(p))
        # The timeout latch lasts only for that stretch.
        self.assertFalse(train.update(ram(0), {}))
        self.assertTrue(train.update(ram(1), {}))

    def test_holds_after_resume_only_for_the_declared_value(self):
        cfg = dict(SKIP, resume_hold_frames={"3": 2}, max_frames=100)
        gate = TransitionGate(cfg, self.vars, episode_end=END, time_limit=False)
        # animation (3) then running: held 2 extra frames, no script input
        seq = [3, 3, 0, 0, 0, 0]
        out = []
        for m in seq:
            held = gate.update(ram(m), {})
            out.append((held, tuple(gate.buttons()) if held else None))
        self.assertEqual([h for h, _ in out], [True, True, True, True, False, False])
        self.assertEqual(out[2][1], ())
        self.assertEqual(out[3][1], ())
        # pause (1) then running: no hold
        held = [gate.update(ram(m), {}) for m in (1, 1, 0, 0)]
        self.assertEqual(held, [True, True, False, False])

    def test_hold_stands_down_if_the_game_ends(self):
        cfg = dict(SKIP, resume_hold_frames={"3": 5})
        gate = TransitionGate(cfg, self.vars, episode_end=END, time_limit=False)
        gate.update(ram(3), {})
        self.assertFalse(gate.update(ram(0, alive=0), {}))

    def test_training_asks_active_and_sees_the_hold_too(self):
        cfg = dict(SKIP, resume_hold_frames={"3": 1})
        gate = TransitionGate(cfg, self.vars, time_limit=False)
        gate.update(ram(3), {})
        self.assertTrue(gate.active(ram(0), {}))
        self.assertTrue(gate.update(ram(0), {}))
        self.assertFalse(gate.active(ram(0), {}))

    def test_active_and_update_agree(self):
        # The training env asks active(), live play calls update(); they must
        # never disagree about a frame.
        a = TransitionGate(SKIP, self.vars, episode_end=END)
        b = TransitionGate(SKIP, self.vars, episode_end=END)
        seq = [(0, 1), (3, 1), (3, 1), (0, 1), (1, 1), (249, 0), (0, 1)]
        for mode, alive in seq:
            r = ram(mode, alive)
            self.assertEqual(a.active(r, {}), b.update(r, {}), (mode, alive))
            a.update(r, {})


if __name__ == "__main__":
    unittest.main()
