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
    """Holes, height and bumpiness -- the standard Tetris board heuristics.

    A hole is an empty cell with a filled cell above it in the same column: the
    thing that makes a stack unrecoverable, and the single most useful signal
    for teaching placement. Bumpiness is the summed height difference between
    neighbouring columns, punishing a jagged surface only an I-piece can fix.
    Line clears alone are far too sparse to learn from -- a random policy tops
    out having never made one -- so these are what give early training a
    gradient at all.
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
    x0 = int(spec.get("x", 153)); y0 = int(spec.get("y", 48))
    cell = int(spec.get("cell", 8))
    cols = int(spec.get("cols", 10)); rows = int(spec.get("rows", 22))
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


def tetris_pixels(vars, ram, info, player=2, frame=None, spec=None):
    """Board heuristics measured from the rendered board."""
    if frame is None:
        return {}
    return board_features(grid_from_frame(frame, spec or {}))

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
