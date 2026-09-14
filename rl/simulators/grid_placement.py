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
      "hole_penalty": 4.0, "height_penalty": 0.4, "bump_penalty": 0.3,
      "shapes": {"<piece_type>": [ [[r,c],...] per rotation ], ...}
    }

The reward for a placement is `line_scale`-tiered line bonus plus the change in
a board-quality potential Phi = -(hole_penalty*holes + height_penalty*height +
bump_penalty*bumpiness). height/bump penalties default to 0, so a game that omits
them keeps holes-only behaviour.

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


def board_stats(board: np.ndarray) -> Tuple[float, float, float]:
    """The three board-quality measurements the reward potential is built from:
    total holes (covered empty cells), aggregate column height, and bumpiness
    (summed neighbour height differences). All generic grid geometry."""
    rows, cols = board.shape
    heights = np.zeros(cols, dtype=np.float32)
    holes = 0.0
    for c in range(cols):
        col = board[:, c]
        filled = np.nonzero(col)[0]
        if filled.size:
            top = filled[0]
            heights[c] = float(rows - top)
            holes += float(np.sum(col[top:] == 0))
    agg_height = float(heights.sum())
    bumpiness = float(np.abs(np.diff(heights)).sum()) if cols > 1 else 0.0
    return holes, agg_height, bumpiness


def board_feature_vector(board: np.ndarray, hole_normalizer=None) -> Tuple[np.ndarray, float]:
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
    norm_holes = np.array([holes / float(hole_normalizer or rows)], dtype=np.float32)
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
        # All reward weights and tiers are supplied by the game configuration.
        self.survival_reward = float(cfg.get("survival_reward", 0.0))
        # Weights of the board-quality potential Phi (see get_candidates). Height
        # and bumpiness default to 0 so a game that declares neither keeps the
        # old holes-only behaviour; this game sets all three in games.json.
        self.hole_penalty = float(cfg.get("hole_penalty", 0.0))
        self.height_penalty = float(cfg.get("height_penalty", 0.0))
        self.bump_penalty = float(cfg.get("bump_penalty", 0.0))
        # One factor scaling the whole per-placement reward. Kept separate from
        # the weights so those stay interpretable; small values keep the value
        # net's targets small enough to converge (see reward_scale_why in
        # games.json -- at 1.0 the net diverged, at 0.3 it settled).
        self.reward_scale = float(cfg.get("reward_scale", 1.0))
        # How the board-quality term enters the reward.
        #   "delta"    : Phi(after) - Phi(before)  (a potential difference)
        #   "absolute" : Phi(after)                (a cost paid every placement)
        # MEASURED: with "delta" the value net learns V ~= C - Phi (corr(V,Phi)
        # reached -0.91, slope dV/dPhi -0.42 by 20k steps) and therefore CANCELS
        # the board term out of argmax(imm + gamma*V): the effective board weight
        # 1 + gamma*dV/dPhi fell from 1.00 untrained to 0.59, heading for 0.01.
        # Greedy play degraded with training (survival 42 untrained -> 23 at 10k
        # -> 17 at 20k). A difference is exactly what a value function absorbs.
        # "absolute" puts the candidate's own board quality only in the immediate
        # reward -- V(s') covers FUTURE boards -- so it cannot be cancelled.
        self.board_term_mode = str(cfg.get("board_term_mode", "delta"))
        # An absolute cost is paid every step, so it must be scaled down by about
        # (1 - gamma) to keep the value targets in the range that converged.
        self.board_term_scale = float(cfg.get("board_term_scale", 1.0))
        self.line_tiers = cfg.get("line_tiers")
        self.lines_var = cfg.get("lines_var")
        self.hole_normalizer = float(cfg.get("hole_normalizer", self.rows))
        # {piece_type: [shape_array per rotation]}, straight from games.json.
        self.shapes: Dict[int, List[np.ndarray]] = {}
        for t, rots in (cfg.get("shapes") or {}).items():
            self.shapes[int(t)] = [cells_to_array(r) for r in rots]
        self.piece_types = int(cfg.get("piece_types", max(self.shapes, default=0) + 1))
        # board cells + heights + neighbour diffs + holes + max height
        self._feature_dim = (self.rows * self.cols + self.cols
                             + max(1, self.cols - 1) + 1 + 1)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def validate_training(self, spec):
        if not self.shapes:
            raise ValueError("grid_placement requires configured shapes")
        if not self.lines_var:
            raise ValueError("grid_placement training requires afterstate.lines_var")
        if (int(spec.grid.get("rows", 0)), int(spec.grid.get("cols", 0))) != (self.rows, self.cols):
            raise ValueError("Simulator and observation board dimensions must agree")
        if int(spec.grid.get("piece_types", 1)) != self.piece_types:
            raise ValueError("Simulator and observation piece_types must agree")

    def encode_observation(self, obs, info=None):
        board = (np.asarray(obs)[:self.rows * self.cols].reshape(self.rows, self.cols) > 0.5).astype(np.uint8)
        return board_feature_vector(board, self.hole_normalizer)[0]

    def _phi(self, board):
        holes, height, bump = board_stats(board)
        return -(self.hole_penalty * holes + self.height_penalty * height + self.bump_penalty * bump)

    def _board_term(self, after, before):
        """Board-quality contribution for one placement (see board_term_mode)."""
        if self.board_term_mode == "absolute":
            return self._phi(after) * self.board_term_scale
        return (self._phi(after) - self._phi(before)) * self.board_term_scale

    def _line_reward(self, lines):
        if self.line_tiers is None:
            return self.line_scale * lines
        if not 0 <= lines < len(self.line_tiers):
            raise ValueError("Observed line clear is outside configured line_tiers")
        return self.line_scale * float(self.line_tiers[lines])

    def observed_reward(self, obs, next_obs, info, next_info, terminated=False):
        if not self.lines_var or self.lines_var not in info or self.lines_var not in next_info:
            raise ValueError("Grid training requires afterstate.lines_var with an observed counter")
        lines = max(0, int(next_info[self.lines_var]) - int(info[self.lines_var]))
        bonus = self._line_reward(lines)
        # Terminal animations are not settled boards. Only counter rewards are
        # meaningful there; the runner adds the configured terminal reward once.
        if terminated:
            return bonus * self.reward_scale
        n = self.rows * self.cols
        before = (np.asarray(obs)[:n].reshape(self.rows, self.cols) > 0.5).astype(np.uint8)
        after = (np.asarray(next_obs)[:n].reshape(self.rows, self.cols) > 0.5).astype(np.uint8)
        return (self.survival_reward + bonus + self._board_term(after, before)) * self.reward_scale

    def _board_and_piece(self, obs, ram, vars, spec):
        """Read the settled board and the current piece type from the
        observation (a grid game's obs is board cells then a piece one-hot),
        falling back to RAM if the caller has no observation."""
        n = self.rows * self.cols
        if obs is not None and getattr(obs, "size", 0) >= n + self.piece_types:
            board = (obs[:n].reshape(self.rows, self.cols) > 0.5).astype(np.uint8)
            onehot = obs[n:n + self.piece_types]
            piece_type = int(np.argmax(onehot)) if np.any(onehot) else None
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

        # Selection predicts the same configured reward that observed_reward
        # measures after execution. Board-potential differences are an explicit
        # reward objective, not a claim of discount-invariant shaping.
        out = []
        for rot, shape in enumerate(rots):
            pw = shape.shape[1]
            for col in range(0, self.cols - pw + 1):
                valid, after, lines = simulate_drop(board, shape, col)
                if not valid or after is None:
                    continue
                feat, _curr_holes = board_feature_vector(after, self.hole_normalizer)
                imm = (self.survival_reward
                       + self._line_reward(lines)
                       + self._board_term(after, board)) * self.reward_scale
                out.append({
                    "action": rot * self.cols + col,
                    "afterstate": feat,
                    "immediate_reward": float(imm),
                    "lines_cleared": lines,
                    "rot": rot,
                    "col": col,
                })
        return out
