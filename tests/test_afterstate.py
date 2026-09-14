"""Regressions for actual-outcome TD and one-placement macro boundaries."""
import os
import sys
import unittest
import contextlib
import io
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.afterstate import AfterstateReplayBuffer, AfterstateTrajectory
from rl.macro import MacroPlacementWrapper
from rl.simulators.grid_placement import GridPlacementSimulator
from rl import afterstate as module


class TrajectoryTests(unittest.TestCase):
    def test_observed_successor_and_terminal_credit(self):
        replay = AfterstateReplayBuffer()
        t = AfterstateTrajectory(replay)
        t.record(9, np.array([1.]))
        t.record(2, np.array([3.]))
        t.record(-6, terminated=True)
        self.assertEqual(len(replay), 2)
        self.assertEqual(replay.buffer[0][2].tolist(), [3.])
        self.assertFalse(replay.buffer[0][3])
        self.assertEqual(replay.buffer[1][1:], (-6., None, True))

    def test_missing_candidate_preserves_credit(self):
        r = AfterstateReplayBuffer()
        t = AfterstateTrajectory(r)
        t.record(0, np.array([4.]))
        t.record(0)
        t.record(-6, terminated=True)
        self.assertEqual(r.buffer[0][0].tolist(), [4.])
        self.assertEqual(r.buffer[0][1], -6)

    def test_cutoff_does_not_add_death(self):
        r = AfterstateReplayBuffer()
        t = AfterstateTrajectory(r)
        t.record(0, np.array([1.]))
        t.record(1, np.array([2.]))
        self.assertEqual(len(r), 1)
        self.assertFalse(r.buffer[0][3])

    def test_reject_missing_successor(self):
        with self.assertRaises(ValueError):
            AfterstateReplayBuffer().push(np.zeros(2), 0, None, False)


class SimulatorTests(unittest.TestCase):
    def test_platformer_keeps_pixel_ppo_path(self):
        from rl.env import TrainingSpec
        from train import policy_for
        config = {"training": {"algorithm": "ppo", "action_mode": "button_stream",
                              "action_set": [[], ["RIGHT"], ["RIGHT", "A"]],
                              "observation": {"kind": "pixels"}}}
        with patch("rl.env.load_entry", return_value=config):
            spec = TrainingSpec("PlatformerExample")
        self.assertEqual(spec.algorithm, "ppo")
        self.assertEqual(spec.action_mode, "button_stream")
        self.assertEqual(policy_for(spec), ("CnnPolicy", True))
        self.assertEqual(spec.actions[-1], ["RIGHT", "A"])

    def test_different_board_and_piece_alphabet(self):
        sim = GridPlacementSimulator({"board": {"rows": 6, "cols": 3},
            "shapes": {"8": [[[0, 0]]]}, "lines_var": "clears", "line_scale": 2,
            "line_tiers": [0, 5], "hole_penalty": 0})
        obs = np.zeros(27, dtype=np.float32)
        obs[26] = 1
        candidates = sim.get_candidates(obs)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0]["afterstate"].size, sim.feature_dim)
        self.assertEqual(sim.observed_reward(obs, obs, {"clears": 1}, {"clears": 2}), 10)
        self.assertEqual(sim.encode_observation(obs).size, sim.feature_dim)

    def test_terminal_pixels_not_scored(self):
        sim = GridPlacementSimulator({"board": {"rows": 2, "cols": 2}, "lines_var": "clears"})
        self.assertEqual(sim.observed_reward(np.zeros(4), None, {"clears": 0}, {"clears": 0}, True), 0)


class RunnerTests(unittest.TestCase):
    def run_case(self, ending, delayed=False):
        replay = AfterstateReplayBuffer()
        actions = []
        class Sim:
            feature_dim = 1  # not a grid game; deliberately no shapes property
            def get_candidates(self, obs, **kwargs):
                if obs[0] == 1 and ending == "missing":
                    return []
                return [{"action": 5, "afterstate": np.array([999.]), "immediate_reward": 999.}]
            def encode_observation(self, obs, info):
                return obs
            def observed_reward(self, *args, **kwargs):
                return 2.
        class Env:
            def reset(self):
                self.i = 0
                return np.array([0.]), {}
            def step(self, action):
                actions.append(action)
                self.i += 1
                end_at = 3 if delayed else 2
                return (np.array([float(self.i)]), 888.,
                        self.i == end_at and ending in ("death", "missing"),
                        self.i == end_at and (ending == "truncation" or delayed),
                        {"afterstate_ready": not (delayed and self.i == 2)})
            def close(self):
                pass
        class Agent:
            def __init__(self, **kwargs):
                pass
            def select_action(self, candidates, **kwargs):
                c = candidates[0]
                return c["action"], c["afterstate"], c["immediate_reward"]
            def save(self, *args):
                pass
        spec = SimpleNamespace(afterstate_config={"simulator": "toy", "terminal_penalty": -6},
                               features_name=None, game="toy", entry={}, report_stats=[], terms=[])
        with tempfile.TemporaryDirectory() as out:
            args = SimpleNamespace(game="toy", save_dir=out, lr=None, gamma=None,
                batch_size=64, device="cpu", ent_coef=None, ent_coef_final=None,
                explore_steps=3, timesteps=2, resume=None, save_every=100)
            with patch("rl.env.make_env", return_value=Env()), \
                    patch("rl.simulators.get_simulator", return_value=Sim()), \
                    patch.object(module, "AfterstateAgent", Agent), \
                    patch.object(module, "AfterstateReplayBuffer", return_value=replay), \
                    contextlib.redirect_stdout(io.StringIO()):
                module.train_afterstate(args, spec, {})
        return replay.buffer, actions

    def test_terminal_reward_on_observed_predecessor(self):
        rows, _ = self.run_case("death")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0].tolist(), [1.])
        self.assertEqual(rows[0][1:], (-4., None, True))

    def test_no_candidate_death(self):
        rows, actions = self.run_case("missing")
        self.assertEqual(actions[:2], [5, None])
        self.assertEqual(rows[0][1:], (-6., None, True))

    def test_budget_and_truncation_bootstrap_actual(self):
        for ending in ("budget", "truncation"):
            rows, _ = self.run_case(ending)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1], 2.)
            self.assertEqual(rows[0][2].tolist(), [2.])
            self.assertFalse(rows[0][3])

    def test_delayed_spawn_waits_and_scores_once(self):
        rows, actions = self.run_case("budget", delayed=True)
        self.assertEqual(actions, [5, 5, None])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2].tolist(), [3.])
        self.assertEqual(rows[0][1], 2.)


class MacroTests(unittest.TestCase):
    def macro(self, samples):
        calls = []
        class Env:
            def step_raw_frame(self, buttons):
                calls.append(buttons)
                return None, 0, False, False, {}
            def finalize_macro_step(self, obs, info, term, trunc):
                return obs, 0, term, trunc, info
        class Macro(MacroPlacementWrapper):
            def _read_var(self, name, default=0):
                kind, row = samples[min(len(calls), len(samples) - 1)]
                return {"rot": 0, "col": 7, "type": kind, "row": row}.get(name, default)
        m = Macro.__new__(Macro)
        m.env = Env()
        m.cfg = {"spawn_wait_frames": 3, "valid_piece_types": [1, 2]}
        m.settle_detect = {"min_previous": 5}
        m.columns = 10
        m.rot_var, m.col_var, m.piece_type_var, m.row_var = "rot", "col", "type", "row"
        m.col_offset, m.spawn_col = 3, 4
        m.commit_button, m.max_settle_frames, m.capture_lock = "DOWN", 4, False
        return m, calls

    def test_type_change_stops_controls(self):
        m, calls = self.macro([(1, 12), (2, 12), (2, 2)])
        result = m.step(0)
        self.assertEqual(calls, [["LEFT"], []])
        self.assertTrue(result[4]["afterstate_ready"])

    def test_same_type_row_reset(self):
        m, calls = self.macro([(1, 12), (1, 2)])
        m.step(0)
        self.assertEqual(calls, [["LEFT"], []])

    def test_wait_is_neutral(self):
        m, calls = self.macro([(1, 2)])
        m.step(None)
        self.assertEqual(calls, [[]])

    def test_timeout_is_not_death(self):
        m, calls = self.macro([(1, 12)])
        result = m.step(4)
        self.assertFalse(result[2])
        self.assertTrue(result[3])
        self.assertFalse(result[4]["afterstate_ready"])

    def test_configured_rotation_origin(self):
        from rl.macro import MacroPlanGenerator
        config = {"columns": 10, "rotations": 4, "col_offset": 3,
                  "piece_col_offsets": {"8": [3, 2, 3, 2]}, "tap_hold": 1, "tap_release": 0}
        plan = MacroPlanGenerator.plan(13, config, current_rot=1, current_col=4, piece_type=8)
        self.assertEqual(plan["frames"], [["LEFT"], ["LEFT"]])

    def test_wrapper_uses_measured_origin(self):
        m, calls = self.macro([(1, 12)])
        m.cfg["piece_col_offsets"] = {"1": [3, 2, 3, 2]}
        m.step(13)
        self.assertEqual(sum(buttons == ["LEFT"] for buttons in calls), 4)


if __name__ == "__main__":
    unittest.main()
