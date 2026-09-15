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
    def test_replay_capacity_retains_recent_transitions(self):
        replay = AfterstateReplayBuffer(capacity=2)
        for value in range(5):
            replay.push(np.array([value]), value, None, True)
        self.assertEqual(len(replay), 2)
        self.assertEqual({row[1] for row in replay.buffer}, {3., 4.})
        with self.assertRaises(ValueError):
            AfterstateReplayBuffer(capacity=0)

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
    def test_declared_grid_shapes_are_encodable_and_controllable(self):
        import json
        from pathlib import Path
        data = json.loads((Path(__file__).resolve().parents[1] / 'games.json').read_text())
        for entry in data['games'].values():
            training = entry.get('training', {})
            cfg = training.get('afterstate', {})
            if cfg.get('simulator') != 'grid_placement':
                continue
            types = {int(t) for t in cfg['shapes']}
            self.assertEqual(types, set(training['macro_config']['valid_piece_types']))
            self.assertGreater(cfg['piece_types'], max(types))
            self.assertEqual(cfg['piece_types'], training['observation']['grid']['piece_types'])
            sim = GridPlacementSimulator(cfg)
            for kind in types:
                obs = np.zeros(sim.rows * sim.cols + cfg['piece_types'] + 3)
                obs[sim.rows * sim.cols + kind] = 1
                self.assertTrue(sim.get_candidates(obs))

    def test_discounted_potential_preserves_task_return(self):
        config = {"board": {"rows": 2, "cols": 2}, "lines_var": "clears",
                  "hole_penalty": 4, "height_penalty": .4, "bump_penalty": .3,
                  "board_term_mode": "potential", "board_term_scale": .1,
                  "discount": .99, "survival_reward": 1, "reward_scale": .1}
        shaped = GridPlacementSimulator(config)
        task = GridPlacementSimulator(dict(config, board_term_scale=0))
        boards = [np.array([1, 0, 0, 0]), np.array([1, 0, 1, 0]), np.zeros(4)]
        def total(sim):
            r0 = sim.observed_reward(boards[0], boards[1], {'clears': 0}, {'clears': 0})
            r1 = sim.observed_reward(boards[1], boards[2], {'clears': 0}, {'clears': 1})
            rt = sim.observed_reward(boards[2], None, {'clears': 1}, {'clears': 1}, True)
            return r0 + .99*r1 + .99**2*rt
        initial_phi = shaped._phi(boards[0].reshape(2, 2))
        self.assertAlmostEqual(total(shaped) - total(task), -initial_phi * .1 * .1)

    def test_recovery_beats_early_death_under_task_reward(self):
        sim = GridPlacementSimulator({'board': {'rows': 2, 'cols': 2}, 'lines_var': 'clears',
              'hole_penalty': 4, 'board_term_mode': 'potential', 'board_term_scale': .1,
              'survival_reward': 1, 'reward_scale': .1, 'line_scale': 10})
        board = np.array([1, 0, 0, 0])
        death = sim.observed_reward(board, None, {'clears': 0}, {'clears': 0}, True)
        recovery = sim.observed_reward(board, np.zeros(4), {'clears': 0}, {'clears': 1})
        recovery += .99 * sim.observed_reward(np.zeros(4), None, {'clears': 1}, {'clears': 1}, True)
        self.assertGreater(recovery, death)

    def test_missing_grid_settings_fail_explicitly(self):
        from rl.env import TrainingSpec
        with patch("rl.env.load_entry", return_value={"training": {"observation": {"kind": "grid"}}}):
            with self.assertRaisesRegex(ValueError, "Declare observation.grid"):
                TrainingSpec("OtherGridGame")

    def test_missing_reward_defaults_fail_explicitly(self):
        sim = GridPlacementSimulator({"shapes": {"1": [[[0, 0]]]}})
        with self.assertRaisesRegex(ValueError, "hole_penalty"):
            sim.validate_training(SimpleNamespace(grid={}))

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
    def run_case(self, ending, delayed=False, backup="sampled"):
        replay = AfterstateReplayBuffer()
        from rl.control import CandidateReplay
        control = CandidateReplay(50000)
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
                        {"afterstate_ready": not ((delayed and self.i == 2) or
                                                   (ending in ("stall", "frame_stall") and self.i >= 2)),
                         "frames": self.i * 100 if ending == "frame_stall" else 0})
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
        spec = SimpleNamespace(afterstate_config={"simulator": "toy", "terminal_penalty": -6, "backup": backup,
                               "max_wait_steps": 3, "max_wait_frames": 100},
                               features_name=None, game="toy", entry={}, report_stats=[], terms=[])
        with tempfile.TemporaryDirectory() as out:
            args = SimpleNamespace(game="toy", save_dir=out, lr=None, gamma=None,
                batch_size=64, device="cpu", ent_coef=None, ent_coef_final=None,
                explore_steps=3, timesteps=2, resume=None, save_every=100)
            with patch("rl.env.make_env", return_value=Env()), \
                    patch("rl.simulators.get_simulator", return_value=Sim()), \
                    patch.object(module, "AfterstateAgent", Agent), \
                    patch.object(module, "AfterstateReplayBuffer", return_value=replay), \
                    patch("rl.control.CandidateReplay", return_value=control), \
                    contextlib.redirect_stdout(io.StringIO()):
                module.train_afterstate(args, spec, {})
        return (replay.buffer if backup == "sampled" else control.buffer), actions

    def test_control_uses_observed_predecessor_and_candidates(self):
        rows, _ = self.run_case("budget", backup="greedy_candidates")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0].tolist(), [1.])
        self.assertEqual(rows[0][1].tolist(), [[999.]])

    def test_control_missing_candidate_terminal_once(self):
        rows, _ = self.run_case("missing", backup="greedy_candidates")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0].tolist(), [1.])
        self.assertEqual(rows[0][3], -6.)

    def test_control_wait_does_not_duplicate_decision(self):
        rows, actions = self.run_case("budget", delayed=True, backup="greedy_candidates")
        self.assertEqual(actions, [5, 5, None])
        self.assertEqual(len(rows), 1)

    def test_terminal_reward_on_observed_predecessor(self):
        rows, _ = self.run_case("death")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0].tolist(), [1.])
        self.assertEqual(rows[0][1:], (-4., None, True))

    def test_pending_stall_is_bounded_without_frame_counter(self):
        with self.assertRaisesRegex(RuntimeError, "3 neutral steps"):
            self.run_case("stall")

    def test_frame_limit_bounds_neutral_macro_wait(self):
        with self.assertRaisesRegex(RuntimeError, "1 neutral steps / 100 frames"):
            self.run_case("frame_stall")

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
