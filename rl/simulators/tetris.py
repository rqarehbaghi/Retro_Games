"""
Tetris-specific afterstate simulator.

Implements BaseAfterstateSimulator:
  - Models the 7 tetromino pieces and their 4 rotation states.
  - Evaluates legal drop trajectories on a 20x10 binary board.
  - Calculates resulting afterstate topology (column heights, holes, bumpiness).
  - Assigns immediate line clear rewards and hole penalties.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from rl.simulators.base import BaseAfterstateSimulator

# Piece Shapes & Rotations
TETROMINO_SHAPES: Dict[int, List[np.ndarray]] = {
    # 0: T
    0: [
        np.array([[1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        np.array([[0, 1], [1, 1], [0, 1]], dtype=np.uint8),
        np.array([[0, 1, 0], [1, 1, 1]], dtype=np.uint8),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=np.uint8),
    ],
    # 1: J
    1: [
        np.array([[1, 1, 1], [0, 0, 1]], dtype=np.uint8),
        np.array([[0, 1], [0, 1], [1, 1]], dtype=np.uint8),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=np.uint8),
        np.array([[1, 1], [1, 0], [1, 0]], dtype=np.uint8),
    ],
    # 2: Z
    2: [
        np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
        np.array([[0, 1], [1, 1], [1, 0]], dtype=np.uint8),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
        np.array([[0, 1], [1, 1], [1, 0]], dtype=np.uint8),
    ],
    # 3: O (Square)
    3: [
        np.array([[1, 1], [1, 1]], dtype=np.uint8),
        np.array([[1, 1], [1, 1]], dtype=np.uint8),
        np.array([[1, 1], [1, 1]], dtype=np.uint8),
        np.array([[1, 1], [1, 1]], dtype=np.uint8),
    ],
    # 4: S
    4: [
        np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8),
        np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8),
        np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8),
    ],
    # 5: L
    5: [
        np.array([[1, 1, 1], [1, 0, 0]], dtype=np.uint8),
        np.array([[1, 1], [0, 1], [0, 1]], dtype=np.uint8),
        np.array([[0, 0, 1], [1, 1, 1]], dtype=np.uint8),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=np.uint8),
    ],
    # 6: I (Line)
    6: [
        np.array([[1, 1, 1, 1]], dtype=np.uint8),
        np.array([[1], [1], [1], [1]], dtype=np.uint8),
        np.array([[1, 1, 1, 1]], dtype=np.uint8),
        np.array([[1], [1], [1], [1]], dtype=np.uint8),
    ],
}

UNIQUE_ROTATIONS: Dict[int, int] = {
    0: 4, 1: 4, 2: 2, 3: 1, 4: 2, 5: 4, 6: 2
}


def simulate_drop(
    board: np.ndarray,
    piece_type: int,
    rot: int,
    col: int
) -> Tuple[bool, Optional[np.ndarray], int]:
    shape = TETROMINO_SHAPES[int(piece_type) % 7][int(rot) % 4]
    ph, pw = shape.shape
    rows, cols = board.shape

    if col < 0 or col + pw > cols:
        return False, None, 0

    landing_r = -1
    for r in range(0, rows - ph + 1):
        sub = board[r:r + ph, col:col + pw]
        if np.any((sub == 1) & (shape == 1)):
            landing_r = r - 1
            break
    else:
        landing_r = rows - ph

    if landing_r < 0:
        return False, None, 0

    new_board = board.copy()
    new_board[landing_r:landing_r + ph, col:col + pw] |= shape

    full_mask = np.all(new_board == 1, axis=1)
    lines_cleared = int(np.sum(full_mask))

    if lines_cleared > 0:
        remaining_rows = new_board[~full_mask]
        empty_rows = np.zeros((lines_cleared, cols), dtype=np.uint8)
        new_board = np.vstack([empty_rows, remaining_rows])

    return True, new_board, lines_cleared


def extract_tetris_features(board: np.ndarray) -> np.ndarray:
    rows, cols = board.shape
    raw = board.astype(np.float32).ravel()

    heights = np.zeros(cols, dtype=np.float32)
    holes = 0.0

    for c in range(cols):
        col_cells = board[:, c]
        filled = np.nonzero(col_cells)[0]
        if filled.size > 0:
            top = filled[0]
            heights[c] = float(rows - top)
            holes += float(np.sum(col_cells[top:] == 0))

    norm_heights = heights / float(rows)
    norm_diffs = np.abs(np.diff(heights)) / float(rows) if cols > 1 else np.zeros(1, dtype=np.float32)
    norm_holes = np.array([holes / 20.0], dtype=np.float32)
    norm_max_h = np.array([np.max(heights) / float(rows)], dtype=np.float32)

    return np.concatenate([raw, norm_heights, norm_diffs, norm_holes, norm_max_h]).astype(np.float32)


class TetrisSimulator(BaseAfterstateSimulator):
    """Afterstate lookahead simulator for 10x20 Tetris."""

    def __init__(self, line_scale: float = 10.0, hole_penalty: float = 2.0):
        self.line_scale = line_scale
        self.hole_penalty = hole_penalty
        self._feature_dim = 200 + 10 + 9 + 1 + 1  # 221

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def get_candidates(
        self,
        obs: np.ndarray,
        ram: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        vars: Optional[Any] = None,
        spec: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        # Extract board and piece from observation
        if obs.size >= 207:
            board = (obs[:200].reshape(20, 10) > 0.5).astype(np.uint8)
            piece_type = int(np.argmax(obs[200:207]))
        elif vars is not None and ram is not None:
            player = getattr(spec, "player", 2)
            piece_type = int(vars.read(f"piece_type_p{player}", ram, {}, 0))
            from rl.features import read_well
            board = read_well(ram, player)
        else:
            return []

        ptype = int(piece_type) % 7
        max_rots = UNIQUE_ROTATIONS.get(ptype, 4)
        candidates = []

        # Previous holes count for penalty delta
        prev_holes = 0
        for c in range(10):
            col_cells = board[:, c]
            filled = np.nonzero(col_cells)[0]
            if filled.size > 0:
                prev_holes += int(np.sum(col_cells[filled[0]:] == 0))

        line_tiers = [0.0, 10.0, 30.0, 60.0, 120.0]

        for rot in range(max_rots):
            shape = TETROMINO_SHAPES[ptype][rot]
            pw = shape.shape[1]
            for col in range(0, 10 - pw + 1):
                valid, afterstate, lines = simulate_drop(board, ptype, rot, col)
                if not valid or afterstate is None:
                    continue

                feat = extract_tetris_features(afterstate)

                # Compute immediate reward
                imm_r = line_tiers[min(lines, 4)]
                curr_holes = 0
                for c in range(10):
                    col_cells = afterstate[:, c]
                    filled = np.nonzero(col_cells)[0]
                    if filled.size > 0:
                        curr_holes += int(np.sum(col_cells[filled[0]:] == 0))

                hole_delta = curr_holes - prev_holes
                if hole_delta > 0:
                    imm_r -= float(hole_delta * self.hole_penalty)

                candidates.append({
                    "action": rot * 10 + col,
                    "afterstate": feat,
                    "immediate_reward": float(imm_r),
                    "lines_cleared": lines,
                    "rot": rot,
                    "col": col,
                })

        return candidates
