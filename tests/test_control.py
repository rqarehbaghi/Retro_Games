"""Generic control-backup tests, with no game-shaped assumptions."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from rl.control import CandidateReplay, candidate_targets


class ControlTests(unittest.TestCase):
    def test_reward_rate_scaling_applies_before_max_backup(self):
        replay = CandidateReplay(4, reward_key='task_reward')
        replay.push(np.array([0.]), candidates=[
            {'afterstate': np.array([1.]), 'immediate_reward': 999., 'task_reward': 4.},
            {'afterstate': np.array([2.]), 'immediate_reward': -999., 'task_reward': 1.},
        ])
        net = lambda x: torch.zeros(x.shape[0])
        agent = SimpleNamespace(device='cpu', gamma=.99, value_reward_scale=.01,
                                val_net=net, target_net=net)
        self.assertAlmostEqual(candidate_targets(agent, replay.buffer).item(), 0.04)

    def test_policy_guidance_selects_action_but_target_stays_task_only(self):
        replay = CandidateReplay(4, reward_key='task_reward',
                                 selection_reward_key='immediate_reward')
        replay.push(np.array([0.]), candidates=[
            {'afterstate': np.array([1.]), 'immediate_reward': -10., 'task_reward': 4.},
            {'afterstate': np.array([2.]), 'immediate_reward': 10., 'task_reward': 1.},
        ])
        net = lambda x: torch.zeros(x.shape[0])
        agent = SimpleNamespace(device='cpu', gamma=.99, value_reward_scale=.01,
                                val_net=net, target_net=net)
        # Guidance chooses candidate 2, but V still learns its task reward (1),
        # not the shaped selection reward (10).
        self.assertAlmostEqual(candidate_targets(agent, replay.buffer).item(), 0.01)

    def test_terminal_candidate_never_bootstraps(self):
        agent = SimpleNamespace(device='cpu', gamma=.99,
                                val_net=lambda x: x[:, 0], target_net=lambda x: x[:, 0])
        replay = CandidateReplay(2)
        item = replay.push([0], [{'afterstate': [1000.], 'immediate_reward': 99.}])
        replay.correct(item, 0, 0., terminal=True)
        self.assertEqual(candidate_targets(agent, replay.buffer).tolist(), [0.])

    def test_terminal_reward_uses_value_units(self):
        agent = SimpleNamespace(device='cpu', gamma=.99, value_reward_scale=.01,
                                val_net=lambda x: x[:, 0], target_net=lambda x: x[:, 0])
        replay = CandidateReplay(2)
        replay.push([0], terminal_reward=-20.)
        self.assertAlmostEqual(candidate_targets(agent, replay.buffer).item(), -.2)

    def test_selects_best_candidate_not_historical_move(self):
        agent = SimpleNamespace(device='cpu', gamma=.5,
                                val_net=lambda x: x[:, 0],
                                target_net=lambda x: x[:, 0] * 2)
        replay = CandidateReplay(4)
        replay.push([0], [{'afterstate': [2], 'immediate_reward': 0},
                          {'afterstate': [0], 'immediate_reward': 4}])
        replay.push([0], terminal_reward=-3)
        self.assertEqual(candidate_targets(agent, replay.buffer).tolist(), [4., -3.])

    def test_unknown_next_piece_is_not_terminal(self):
        with self.assertRaises(ValueError):
            CandidateReplay(4).push([0], [])

    def test_copies_inputs(self):
        state = np.array([1.])
        feat = np.array([2.])
        replay = CandidateReplay(4)
        replay.push(state, [{'afterstate': feat}])
        state[:] = feat[:] = 9
        self.assertEqual(replay.buffer[0][0].tolist(), [1.])
        self.assertEqual(replay.buffer[0][1].tolist(), [[2.]])

if __name__ == '__main__':
    unittest.main()
