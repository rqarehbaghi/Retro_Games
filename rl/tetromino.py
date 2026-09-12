"""
Tetromino kinematics for Tetris (proxies to rl.simulators.tetris).
Kept for backward compatibility with existing tests and scripts.
"""
from rl.simulators.tetris import (
    TETROMINO_SHAPES,
    UNIQUE_ROTATIONS,
    simulate_drop,
    extract_tetris_features,
    TetrisSimulator
)
import numpy as np


def get_piece_shape(piece_type: int, rotation: int) -> np.ndarray:
    return TETROMINO_SHAPES[int(piece_type) % 7][int(rotation) % 4]


def get_all_placements(board: np.ndarray, piece_type: int):
    sim = TetrisSimulator()
    obs = np.zeros(207, dtype=np.float32)
    obs[:200] = board.ravel()
    obs[200 + (int(piece_type) % 7)] = 1.0
    cands = sim.get_candidates(obs)
    out = []
    for c in cands:
        out.append({
            "action": c["action"],
            "rot": c["rot"],
            "col": c["col"],
            "afterstate": c["afterstate"],
            "lines_cleared": c["lines_cleared"],
            "landing_row": 0
        })
    return out


def get_action_mask(board: np.ndarray, piece_type: int) -> np.ndarray:
    mask = np.zeros(40, dtype=bool)
    for p in get_all_placements(board, piece_type):
        mask[p["action"]] = True
    return mask
