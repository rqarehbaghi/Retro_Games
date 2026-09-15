"""Generic control-backup tests, with no game-shaped assumptions."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from rl.control import CandidateReplay, candidate_targets


class ControlTests(unittest.TestCase):
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
