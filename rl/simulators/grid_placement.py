"""
Generic afterstate simulator for grid-drop placement games (Tetris, Columns,
Dr. Mario, Puyo...).

It is fully game-agnostic. The board size, the piece shapes and the reward
parameters ALL come from the game's `afterstate` block in games.json -- there is
no hardcoded piece set. A game declares its own pieces, MEASURED against the
real ROM, and this evaluates every legal placement of the current piece:

    "afterstate": {
      "simulator": "grid_placement",
      "board": {"rows": 20, "cols": 10},
      "line_scale": 10.0,
      "hole_penalty": 2.0,
      "shapes": {"<piece_type>": [ [[r,c],...] per rotation ], ...}
    }

`shapes` is keyed by the game's own piece_type value, so nothing here needs to
know a "standard" tetromino order -- assuming one was the bug that made an
earlier version imagine the wrong piece for most of this game's types.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from rl.simulators.base import BaseAfterstateSimulator


def cells_to_array(cells) -> np.ndarray:
    """A list of [row, col] cells, top-left normalised, to a dense 0/1 array."""
    pts = [(int(r), int(c)) for r, c in cells]
    h = max(r for r, _ in pts) + 1
    w = max(c for _, c in pts) + 1
    a = np.zeros((h, w), dtype=np.uint8)
    for r, c in pts:
        a[r, c] = 1
    return a


def simulate_drop(board: np.ndarray, shape: np.ndarray, col: int
                  ) -> Tuple[bool, Optional[np.ndarray], int]:
    """Drop `shape` (its left edge at `col`) straight down into `board`. Returns
    (valid, resulting_board, lines_cleared). The board is settled cells only."""
    ph, pw = shape.shape
    rows, cols = board.shape
    if col < 0 or col + pw > cols:
        return False, None, 0

    landing_r = rows - ph
    for r in range(0, rows - ph + 1):
        sub = board[r:r + ph, col:col + pw]
        if np.any((sub == 1) & (shape == 1)):
            landing_r = r - 1
            break

    if landing_r < 0:
        return False, None, 0

    nb = board.copy()
    nb[landing_r:landing_r + ph, col:col + pw] |= shape

    full = np.all(nb == 1, axis=1)
    lines = int(np.sum(full))
    if lines:
        nb = np.vstack([np.zeros((lines, cols), dtype=np.uint8), nb[~full]])
    return True, nb, lines


def board_feature_vector(board: np.ndarray) -> Tuple[np.ndarray, float]:
    """The afterstate representation fed to the value net, and the raw hole
    count (used for the immediate-reward hole delta).

    Layout: flattened board, then per-column normalised heights, the normalised
    height differences between neighbours, the normalised hole count, and the
    normalised max height. All generic grid measurements.
    """
    rows, cols = board.shape
    raw = board.astype(np.float32).ravel()
    heights = np.zeros(cols, dtype=np.float32)
    holes = 0.0
    for c in range(cols):
        col = board[:, c]
        filled = np.nonzero(col)[0]
        if filled.size:
            top = filled[0]
            heights[c] = float(rows - top)
            holes += float(np.sum(col[top:] == 0))
    norm_heights = heights / float(rows)
    norm_diffs = (np.abs(np.diff(heights)) / float(rows)
                  if cols > 1 else np.zeros(1, dtype=np.float32))
    norm_holes = np.array([holes / 20.0], dtype=np.float32)
    norm_max = np.array([np.max(heights) / float(rows) if cols else 0.0],
                        dtype=np.float32)
    feat = np.concatenate([raw, norm_heights, norm_diffs, norm_holes, norm_max])
    return feat.astype(np.float32), holes


class GridPlacementSimulator(BaseAfterstateSimulator):
    """Afterstate lookahead for any game whose pieces are declared in games.json."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(config or {})
        board = cfg.get("board") or {}
        self.rows = int(board.get("rows", 20))
        self.cols = int(board.get("cols", 10))
        self.line_scale = float(cfg.get("line_scale", 10.0))
        self.hole_penalty = float(cfg.get("hole_penalty", 2.0))
        # {piece_type: [shape_array per rotation]}, straight from games.json.
        self.shapes: Dict[int, List[np.ndarray]] = {}
        for t, rots in (cfg.get("shapes") or {}).items():
            self.shapes[int(t)] = [cells_to_array(r) for r in rots]
        # board cells + heights + neighbour diffs + holes + max height
        self._feature_dim = (self.rows * self.cols + self.cols
                             + max(1, self.cols - 1) + 1 + 1)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _board_and_piece(self, obs, ram, vars, spec):
        """Read the settled board and the current piece type from the
        observation (a grid game's obs is board cells then a piece one-hot),
        falling back to RAM if the caller has no observation."""
        n = self.rows * self.cols
        if obs is not None and getattr(obs, "size", 0) >= n + 7:
            board = (obs[:n].reshape(self.rows, self.cols) > 0.5).astype(np.uint8)
            piece_type = int(np.argmax(obs[n:n + 7]))
            return board, piece_type
        if vars is not None and ram is not None:
            player = getattr(spec, "player", 2)
            piece_type = int(vars.read("piece_type_p%d" % player, ram, {}, 0))
            # A RAM-only board reader would go here for a game with no grid
            # observation; none is needed while the grid observation is on.
            return None, piece_type
        return None, None

    def get_candidates(self, obs, ram=None, info=None, vars=None, spec=None
                       ) -> List[Dict[str, Any]]:
        board, piece_type = self._board_and_piece(obs, ram, vars, spec)
        if board is None or piece_type is None:
            return []
        rots = self.shapes.get(int(piece_type))
        if not rots:
            # A transient piece_type value (between pieces) has no shape; nothing
            # to place, so no candidates.
            return []

        prev_holes = 0
        for c in range(self.cols):
            col = board[:, c]
            filled = np.nonzero(col)[0]
            if filled.size:
                prev_holes += int(np.sum(col[filled[0]:] == 0))

        s = self.line_scale
        tiers = [0.0, s, s * 3, s * 6, s * 12]     # a Tetris is worth far more
        out = []
        for rot, shape in enumerate(rots):
            pw = shape.shape[1]
            for col in range(0, self.cols - pw + 1):
                valid, after, lines = simulate_drop(board, shape, col)
                if not valid or after is None:
                    continue
                feat, curr_holes = board_feature_vector(after)
                imm = tiers[min(lines, 4)]
                hole_delta = curr_holes - prev_holes
                if hole_delta > 0:
                    imm -= float(hole_delta * self.hole_penalty)
                out.append({
                    "action": rot * self.cols + col,
                    "afterstate": feat,
                    "immediate_reward": float(imm),
                    "lines_cleared": lines,
                    "rot": rot,
                    "col": col,
                })
        return out
