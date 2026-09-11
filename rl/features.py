#!/usr/bin/env python3
"""
Per-game feature hooks: the parts of a reward that a JSON value cannot express.

Most shaping needs nothing here. A term like "pay for score going up" or "pay
when course_clear hits 255" is just a named variable from games.json, and the
generic reward in rl/env.py handles it. A hook is for the cases where the
number the agent should be paid on has to be COMPUTED from raw memory --
Tetris's holes and bumpiness come from unpacking a bit-packed board and then
doing geometry on it, which no amount of schema would capture.

A hook is a plain function:

    def compute(vars, ram, info, player) -> {name: float}

registered under the key a game names in games.json ("training": {"features":
...}). The values it returns can then be used by feature_delta / feature_level
reward terms by name. Returning an empty dict is fine and leaves those terms
inert, which is what should happen when a game has no hook.
"""
import numpy as np

# ---------------------------------------------------------------- Tetris ---
# The board is bit-packed: one 16-byte line per row with both players side by
# side, four bits per cell. It is also DOUBLE BUFFERED across two bases and the
# game alternates which copy is current -- reading a fixed one would hand the
# agent an empty board (measured: 0x0700 filled to 7 cells as pieces landed
# while 0x0600 sat flat at 0, and in another run the reverse). A stale buffer
# lags BEHIND, never ahead, so the mirror holding more cells is the live one.
WELL_BASES = (0x0600, 0x0700)
WELL_ROW_STRIDE = 16
WELL_PLAYER_OFF = 8      # player 1 at +0, player 2 at +8
WELL_ROWS = 13
WELL_SKIP_NIBBLES = 3    # wall nibbles at each end of the 16-nibble row
WELL_COLS = 10


def _well_at(ram, base, player):
    off = WELL_PLAYER_OFF if player == 2 else 0
    grid = np.zeros((WELL_ROWS, WELL_COLS), dtype=np.uint8)
    for r in range(WELL_ROWS):
        b = base + WELL_ROW_STRIDE * r + off
        nib = []
        for x in ram[b:b + 8]:
            nib += [int(x) >> 4, int(x) & 0xF]
        cells = nib[WELL_SKIP_NIBBLES:WELL_SKIP_NIBBLES + WELL_COLS]
        grid[r] = [1 if c else 0 for c in cells]
    return grid


def read_well(ram, player=2):
    """One player's LOCKED board as 13x10 of 0/1, from BOTH blocks together.

    The falling piece is NOT in here -- only cells that have come to rest.

    The two blocks are not mirrors of one board, which is what an earlier
    version assumed when it took whichever held more cells. Measured: each lock
    writes to exactly ONE of them, alternating, so a piece that landed in the
    other block was invisible and the shaping scored a board with pieces
    missing. Their union tracks the real board far more closely -- it grows on
    every lock, where either block alone stalls on the locks it did not receive.

    The tradeoff is that a stale block can hold a row that has since been
    cleared, so the union can briefly keep a cleared row. That is rare and
    self-correcting once both blocks are written again, and far less wrong than
    permanently missing half the pieces.
    """
    grids = [_well_at(ram, b, player) for b in WELL_BASES]
    out = grids[0]
    for g in grids[1:]:
        out = out | g
    return out


def tetris(vars, ram, info, player=2):
    """Holes, height and bumpiness from legacy 13-row RAM decode.

    NOTE: The RAM-based well decoder reads only 13 rows and cannot observe the
    full 20-22 row playfield due to NES double-buffering limits. For complete
    heuristics, games.json configures the pixel-sampling hook 'tetris_pixels'.
    """
    grid = read_well(ram, player)
    rows, cols = grid.shape
    heights = np.zeros(cols, dtype=np.int32)
    holes = 0
    for c in range(cols):
        col = grid[:, c]
        filled = np.nonzero(col)[0]
        if filled.size:
            top = filled[0]
            heights[c] = rows - top
            holes += int((col[top:] == 0).sum())
    return {
        "holes": float(holes),
        "height": float(heights.sum()),
        "max_height": float(heights.max()) if cols else 0.0,
        "bumpiness": float(np.abs(np.diff(heights)).sum()) if cols > 1 else 0.0,
    }



# ---------------------------------------------------- board from pixels ---
# The board is read from the SCREEN, not from RAM. The RAM layout for this
# game's well resisted mapping: the two blocks at 0x0600/0x0700 are neither
# mirrors nor halves, a decode of them reproduced only 3 of 22 rows against the
# rendered frame, and the piece row counter reaches 21 while any 13-row decode
# compresses the stack. Sampling the rendered cells is exact by construction --
# it agrees with what is on screen because it IS what is on screen -- and it
# needs no archaeology. The cost is that it only sees what is drawn, which for a
# board is all that matters.
def grid_from_frame(frame, spec):
    """Binary rows x cols board sampled from the rendered frame."""
    spec = spec or {}
    x0 = int(spec.get("x", 153))
    y0 = int(spec.get("y", 56))
    cell = int(spec.get("cell", 8))
    cols = int(spec.get("cols", 10))
    rows = int(spec.get("rows", 20))
    thr = float(spec.get("threshold", 40))
    pad = max(1, cell // 4)
    g = np.zeros((rows, cols), dtype=np.uint8)
    h, w = frame.shape[:2]
    for r in range(rows):
        y = y0 + r * cell
        if y + cell > h:
            break
        for c in range(cols):
            x = x0 + c * cell
            if x + cell > w:
                break
            patch = frame[y + pad:y + cell - pad, x + pad:x + cell - pad]
            g[r, c] = 1 if patch.mean() > thr else 0
    return g


def board_features(grid):
    """Holes, height and bumpiness from a binary board."""
    rows, cols = grid.shape
    heights = np.zeros(cols, dtype=np.int32)
    holes = 0
    for c in range(cols):
        col = grid[:, c]
        filled = np.nonzero(col)[0]
        if filled.size:
            top = filled[0]
            heights[c] = rows - top
            holes += int((col[top:] == 0).sum())
    return {"holes": float(holes), "height": float(heights.sum()),
            "max_height": float(heights.max()) if cols else 0.0,
            "bumpiness": float(np.abs(np.diff(heights)).sum()) if cols > 1 else 0.0}


def mask_piece(grid, ram, vars, player, spec):
    """Erase the FALLING piece so the features score the settled stack only.

    The board is sampled from the screen, so the piece in flight is drawn into
    it, and the gap beneath it reads as holes that appear and vanish as it
    descends. Measured before this: the holes term swung by +-8 per step against
    a terminal of -10.

    The piece's row and column come from the verified RAM addresses. Their
    origin is offset from the grid, calibrated by diffing consecutive frames --
    the cells a falling piece VACATES are exactly the piece: ramRow 10 -> grid row 5
    and ramCol 7 -> grid cols 4..6, i.e. row - 6 and column - 3 with the calibrated
    20-row grid (y0=56). A 4x4 box covers any tetromino; one extra row of margin
    absorbs the row counter lagging the drawn position by a frame.
    """
    if ram is None:
        return grid
    r = vars.read("piece_row_p%d" % player, ram, {})
    c = vars.read("piece_col_p%d" % player, ram, {})
    if r is None or c is None:
        return grid
    row0 = int(r) - int((spec or {}).get("row_offset", 6)) - 1
    col0 = int(c) - int((spec or {}).get("col_offset", 3))
    out = grid.copy()
    rows, cols = out.shape
    for y in range(max(0, row0), min(rows, row0 + 5)):
        for x in range(max(0, col0), min(cols, col0 + 4)):
            out[y, x] = 0
    return out


def tetris_pixels(vars, ram, info, player=2, frame=None, spec=None):
    """Board heuristics measured from the SETTLED board (piece masked out)."""
    if frame is None:
        return {}
    g = grid_from_frame(frame, spec or {})
    if (spec or {}).get("mask_piece", True):
        g = mask_piece(g, ram, vars, player, spec)
    return board_features(g)

HOOKS = {"tetris": tetris, "tetris_pixels": tetris_pixels}


def get(name):
    """The hook a game asked for, or one that returns nothing."""
    if not name:
        return lambda *_a, **_k: {}
    hook = HOOKS.get(name)
    if hook is None:
        print("No feature hook named %r; feature reward terms will be inert. "
              "Known hooks: %s" % (name, ", ".join(sorted(HOOKS)) or "(none)"))
        return lambda *_a, **_k: {}
    return hook
