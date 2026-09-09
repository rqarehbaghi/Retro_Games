#!/usr/bin/env python3
"""
Find RAM addresses for game variables the integration doesn't publish, by
matching their BEHAVIOUR in one of your own recordings.

Motivated by three gaps found while auditing SuperMarioBros3-Nes-v0:

  COINS   -- not exposed at all, so coin collection is invisible to shaping
             except through the score it also grants.
  P-METER -- SMB3's speed/power meter. It fills while running at full tilt and,
             once full, is what lets raccoon Mario take off. Nothing in the
             pipeline can see it, which is precisely why "explore the sky" is
             currently unreachable: there is no signal for the state that
             precedes flight.
  TIMER   -- the published `time` reads a constant 299 and then jumps to 43, so
             it is NOT the level timer. Whatever it is, it isn't usable.

Each is found by its signature over a demo rather than by guessing an address:

  --find coins   values in a small non-negative range that step UP by one and
                 essentially never fall (except a reset at 100 / new level)
  --find meter   a small bounded enum (0..~8) that rises AND falls repeatedly
                 and spends much of its time at the bottom -- a meter filling
                 and draining as you sprint and stop
  --find timer   decreases far more often than it increases, in slow steady
                 steps, over a wide range

Usage:
    python find_game_vars.py --demo ./human_demos_v2/<file>.bk2 --find coins
    python find_game_vars.py --demo ./human_demos_v2/<file>.bk2 --find meter
    python find_game_vars.py --demo ./human_demos_v2/<file>.bk2 --find timer
    python find_game_vars.py --demo ./human_demos_v2/<file>.bk2 --watch 0x0ABC

Record a demo that exercises the thing you're looking for: collect coins
deliberately for --find coins, and do long full-speed sprints (and stops) for
--find meter. Then confirm the winner with --watch and audit_ram.py.
"""
import argparse

import numpy as np
import stable_retro as retro

import os
import sys

# Custom integrations (games in this repo's integrations/, e.g. TetrisTime) are
# only visible to retro.make after they are registered. tools/ sits one level
# down, so reach the repo root for the helper.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import custom_integrations
    custom_integrations.register()
except Exception:                                                # noqa: BLE001
    pass



def replay(bk2_path):
    """Replay a recording, returning (ram_matrix, infos)."""
    movie = retro.Movie(bk2_path)
    movie.step()
    env = retro.make(
        game=movie.get_game(), state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players, render_mode="rgb_array",
    )
    env.initial_state = movie.get_state()
    env.reset()
    buttons = env.unwrapped.buttons
    rams, infos, presses = [], [], []
    while movie.step():
        # A 2-player movie stores one key set PER PLAYER and the env's action is
        # the concatenation (MultiBinary(num_buttons * players)). Reading only
        # player 0 -- what this did -- fed a short action with P2's inputs
        # MISSING, so a 2-player recording replayed as a game that never
        # happened: P2 sat still, the RNG diverged, and every RAM value read
        # back was fiction. studio.py always did this correctly; the tools did not.
        keys = [movie.get_key(i, p) for p in range(movie.players)
                for i in range(env.num_buttons)]
        _obs, _rew, terminated, truncated, info = env.step(keys)
        rams.append(env.get_ram().copy())
        infos.append(info)
        presses.append(keys[:env.num_buttons])   # P1 only: analyses zip this with `buttons`
        if terminated or truncated:
            break
    env.close()
    if not rams:
        raise SystemExit("No frames replayed -- is that a valid .bk2?")
    return (np.array(rams, dtype=np.int32), infos,
            np.array(presses, dtype=bool), buttons)


def _rowdec(row, enc):
    """Decode a single RAM row under one encoding, over base addresses 0..A-3
    (aligned so 1-, 2- and 3-byte encodings share a base index)."""
    b0 = row[:-2].astype(np.int64); b1 = row[1:-1].astype(np.int64); b2 = row[2:].astype(np.int64)

    def bcd(b):
        hi, lo = b >> 4, b & 0xF
        v = hi * 10 + lo
        v[(hi > 9) | (lo > 9)] = -1
        return v

    return {
        "u8": b0, "bcd8": bcd(b0),
        "u16le": b0 + (b1 << 8), "u16be": (b0 << 8) + b1,
        "bcd16le": bcd(b0) + bcd(b1) * 100, "bcd16be": bcd(b0) * 100 + bcd(b1),
        "u24le": b0 + (b1 << 8) + (b2 << 16),
        "bcd24le": bcd(b0) + bcd(b1) * 100 + bcd(b2) * 10000,
        "bcd24be": bcd(b0) * 10000 + bcd(b1) * 100 + bcd(b2),
    }[enc]


SCALAR_ENCS = ["u8", "bcd8", "u16le", "u16be", "bcd16le", "bcd16be",
               "u24le", "bcd24le", "bcd24be"]


def find_value(ram, obs, tol=3):
    """Find every address+encoding that equals the observed value AT each
    observed frame (within +/- tol frames). Covers binary, BCD, and DIGIT
    TILES (one digit per byte, byte = digit + offset), which is how games that
    keep no packed number store the HUD. This is the discovery half; confirm a
    winner with --watch against the video."""
    F, A = ram.shape
    for f, _ in obs:
        if not 0 <= f < F:
            raise SystemExit("frame %d is outside the %d-frame recording" % (f, F))

    hits = []
    # binary / BCD, aligned to base range 0..A-3
    for enc in SCALAR_ENCS:
        acc = None
        for f, v in obs:
            wnd = np.zeros(A - 2, dtype=bool)
            for ff in range(max(0, f - tol), min(F, f + tol + 1)):
                wnd |= (_rowdec(ram[ff], enc) == v)
            acc = wnd if acc is None else (acc & wnd)
        for base in np.where(acc)[0]:
            hits.append((int(base), enc, None))

    # digit tiles: MSD-first, common offsets (0 = raw digits, 0x30 = ASCII/tile)
    maxw = max(len(str(int(v))) for _, v in obs)
    for width in range(maxw, 7):
        digmat = np.array([[int(c) for c in str(int(v)).rjust(width, "0")]
                           for _, v in obs])
        nbase = A - width
        for off in (0, 0x30):
            acc = None
            for i, (f, v) in enumerate(obs):
                wnd = np.zeros(nbase, dtype=bool)
                for ff in range(max(0, f - tol), min(F, f + tol + 1)):
                    row = ram[ff]
                    ok = np.ones(nbase, dtype=bool)
                    for pos in range(width):
                        ok &= (row[pos:pos + nbase] == digmat[i, pos] + off)
                    wnd |= ok
                acc = wnd if acc is None else (acc & wnd)
            for base in np.where(acc)[0]:
                hits.append((int(base), "tiles%d" % width, off))
    return hits


def find_latch(ram, tail_frames=30, head_frac=0.25, after=0):
    """Find addresses that LATCH: a value absent from early play that then
    holds constant to the very end of the recording. That is the shape of a
    game-over / course-clear / level flag -- one steady value all game, then a
    different steady value for the end sequence (SMB3's course_clear 0x00C4 is
    exactly this: 0 all level, 255 for the clear). Returns
    (addr, early_value, latched_value, latch_frame), latch_frame being where
    the final constant run begins. A flag that latches LATEST -- nearest the
    real ending -- is the likeliest game-over signal, so results are sorted
    that way. Confirm the top one with --watch against the video."""
    F, A = ram.shape
    head = max(1, int(F * head_frac))
    tail_frames = min(tail_frames, F - 1)
    hits = []
    for a in range(A):
        col = ram[:, a]
        c1 = int(col[-1])
        if not np.all(col[F - tail_frames:] == c1):      # tail must be flat
            continue
        if np.any(col[:head] == c1):                     # c1 must be NEW
            continue
        start = F - tail_frames                          # extend the final run
        while start > 0 and int(col[start - 1]) == c1:
            start -= 1
        if start < after:
            continue
        early = int(np.bincount(col[:head]).argmax())    # typical early value
        hits.append((a, early, c1, int(start)))
    hits.sort(key=lambda h: -h[3])
    return hits


def parse_watch(term):
    """'0x0418' -> (addr, 'raw', 1);  '0x0418:tiles6' -> (addr, 'tiles', 6)."""
    term = term.strip()
    if ":" in term:
        a, enc = term.split(":", 1)
        addr = int(a, 0)
        if enc.startswith("tiles"):
            return addr, "tiles", int(enc[5:] or "6")
    return int(term, 0), "raw", 1


def sprint_mask(presses, buttons, run_frames=16):
    """Frames where the player was SUSTAINING a sprint: run button held together
    with a direction, for at least run_frames consecutively.

    A P-meter is defined by what causes it, so searching for what it correlates
    with beats searching for its shape -- especially when the display ('6 arrows
    plus a P') suggests the state may be split across more than one variable."""
    idx = {b: i for i, b in enumerate(buttons) if b}
    run = presses[:, idx["B"]] if "B" in idx else np.zeros(len(presses), bool)
    move = np.zeros(len(presses), bool)
    for d in ("LEFT", "RIGHT"):
        if d in idx:
            move |= presses[:, idx[d]]
    holding = run & move
    # Require a sustained hold: the meter fills only after running a while.
    sustained = np.zeros_like(holding)
    streak = 0
    for i, h in enumerate(holding):
        streak = streak + 1 if h else 0
        if streak >= run_frames:
            sustained[i] = True
    return sustained


def score_coins(col, n):
    """A coin counter: small values, steps of +1, almost never decreases."""
    d = np.diff(col)
    ups = int(np.count_nonzero(d > 0))
    downs = int(np.count_nonzero(d < 0))
    if ups < 3 or col.max() > 99 or col.min() < 0:
        return None
    step_ones = int(np.count_nonzero(d == 1))
    if step_ones < ups * 0.8:      # increments must be single coins
        return None
    if downs > 3:                  # a reset at 100 or a new level, not more
        return None
    return ups * 10 - downs * 5


def score_meter_correlated(col, sprinting):
    """Rank a byte by how well it behaves like a meter DRIVEN BY SPRINTING:
    rising while the sprint is sustained, falling once it stops."""
    d = np.diff(col)
    sprint_at = sprinting[1:]
    ups = d > 0
    downs = d < 0
    n_up, n_down = int(ups.sum()), int(downs.sum())
    if n_up < 3 or n_down < 3:
        return None
    if col.max() - col.min() < 2 or col.max() > 255:
        return None
    up_while_sprint = float((ups & sprint_at).sum()) / n_up
    down_while_idle = float((downs & ~sprint_at).sum()) / n_down
    # Both must be better than chance to be interesting at all.
    if up_while_sprint < 0.55 or down_while_idle < 0.55:
        return None
    return (up_while_sprint + down_while_idle) * 100 + min(n_up, n_down) * 0.1


def score_meter(col, n):
    """A P-meter: bounded small enum that rises AND falls, idling at the bottom.

    Deliberately permissive. A stricter version returned nothing on a real demo,
    which is the wrong failure: a meter that never fills (no sustained sprint in
    the recording) still shows its signature at low values, and hard filters
    hide that. Rank instead of reject."""
    d = np.diff(col)
    ups = int(np.count_nonzero(d > 0))
    downs = int(np.count_nonzero(d < 0))
    if col.max() > 16 or col.max() < 1:
        return None
    if ups < 3 or downs < 3:
        return None
    balance = min(ups, downs) / max(ups, downs)
    bottom_frac = float(np.count_nonzero(col == col.min())) / n
    # Reward balance and bottom-dwelling rather than requiring them.
    return ups + downs + balance * 60 + bottom_frac * 40


def score_timer(col, n):
    """A countdown byte.

    Handles BOTH encodings, because a strict "mostly decreases" test found
    nothing on a real demo: SMB3 keeps the timer as separate BCD DIGITS, and a
    single digit wraps 0 -> 9 constantly, so its ups and downs are nearly equal.
    A whole-value timer still shows the plain decreasing signature."""
    d = np.diff(col)
    ups = int(np.count_nonzero(d > 0))
    downs = int(np.count_nonzero(d < 0))
    if downs < 5:
        return None
    # Plain countdown: decreases dominate.
    if downs >= ups * 3 and col.max() - col.min() >= 5:
        return downs * 10 - ups * 5
    # BCD digit: 0..9, steps of -1 with periodic +9 wraps.
    if col.max() <= 9 and col.min() >= 0:
        minus_one = int(np.count_nonzero(d == -1))
        wraps = int(np.count_nonzero(d == 9))
        if minus_one >= 5 and minus_one >= wraps:
            return minus_one * 8 + wraps
    return None


def find_bcd_timer(ram, n, top):
    """Look for adjacent digit bytes that TOGETHER count down.

    SMB3 shows a 3-digit timer; if it is stored as one digit per byte, no single
    address looks like a countdown but the combination does."""
    results = []
    size = ram.shape[1]
    for addr in range(size - 2):
        trio = ram[:, addr:addr + 3]
        if trio.max() > 9 or trio.min() < 0:
            continue
        value = trio[:, 0] * 100 + trio[:, 1] * 10 + trio[:, 2]
        d = np.diff(value)
        downs = int(np.count_nonzero(d < 0))
        ups = int(np.count_nonzero(d > 0))
        if downs < 10 or downs < ups * 3:
            continue
        if value.max() - value.min() < 20:
            continue
        results.append((downs * 10 - ups * 5, addr, int(value.min()), int(value.max()), ups, downs))
    results.sort(reverse=True)
    if results:
        print("\n3-digit BCD timer candidates (addr = the HUNDREDS digit;")
        print("value = ram[addr]*100 + ram[addr+1]*10 + ram[addr+2]):\n")
        print(f"  {'ADDR':>8}  {'DEC':>6}  {'RANGE':>12}  {'UPS':>5}  {'DOWNS':>5}")
        for _s, addr, lo, hi, ups, downs in results[:top]:
            print(f"  0x{addr:04X}  {addr:6d}  {lo:5d}..{hi:<5d}  {ups:5d}  {downs:5d}")
    else:
        print("\n(no 3-digit BCD timer found either)")


SCORERS = {"coins": score_coins, "meter": score_meter, "timer": score_timer}

HINTS = {
    "coins": "Steps up by 1 each coin. Cross-check a printed frame against the\n"
             "  coin counter on screen in the video.",
    "meter": "SMB3's P-meter is usually 0..7. The winner should climb while you\n"
             "  sprint and fall when you stop -- confirm with --watch, then it can\n"
             "  be rewarded to make sustained running (and therefore flight)\n"
             "  learnable.",
    "timer": "Should tick down steadily during play. The published `time` key\n"
             "  does NOT (it sits at 299 then jumps to 43), so it is not this.",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", required=True, help="A .bk2 recording that exercises the variable")
    p.add_argument("--find", choices=sorted(SCORERS), help="Which signature to search for")
    p.add_argument("--watch", default=None, help="Comma-separated addresses to print over time instead of searching. Append :tilesN to decode N display-digit tiles into a number, e.g. 0x0418:tiles6 for a 6-digit HUD score.")
    p.add_argument("--find-value", default=None, metavar="FRAME:VALUE,...", help="Discovery search: given values read off the HUD at known frames (e.g. \"3613:19,9034:405,14454:6411\"), find the address+encoding that holds them. Tries binary, BCD and digit-tiles. Get frame numbers from the video timestamp x 60.0988.")
    p.add_argument("--tol", type=int, default=3, help="Frame tolerance for --find-value, since a timestamp->frame is not exact (default: %(default)s)")
    p.add_argument("--find-latch", action="store_true", help="Discovery search for a game-over / course-clear / level FLAG: an address whose value is absent from early play and then holds steady to the end of the recording. Record the winner in games.json the way course_clear is.")
    p.add_argument("--after", type=int, default=0, help="For --find-latch, only report flags that latch at or after this frame -- pass the frame the ending starts to cut coincidental early latches (default: %(default)s).")
    p.add_argument("--control", default=None, metavar="BK2", help="For --find-latch: a SECOND recording that never reaches the state (e.g. a run that changes level but never tops out). A real flag never takes its latched value in a run that never hit the state, so any candidate whose latched value appears in the control is dropped. This is the cross-recording check the house rules require -- it is what separates a game-over flag from a per-level or per-piece byte.")
    p.add_argument("--compare", default=None, help="Two addresses (e.g. 0x25A2,0x2167) to diff frame by frame. Use when a search returns several candidates that look equally good: identical everywhere means one is a copy of the other (either works); any divergence tells you which is the real variable and which is a display mirror.")
    p.add_argument("--every", type=int, default=60, help="Sample interval for --watch (default: %(default)s)")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--run-frames", type=int, default=16, help="Consecutive frames of run+direction that count as a sustained sprint for --find meter (default: %(default)s)")
    args = p.parse_args()

    if not (args.find or args.watch or args.compare or args.find_value or args.find_latch):
        raise SystemExit("Pass --find {coins,meter,timer}, --find-value FRAME:VALUE,..., --watch 0xADDR, or --compare 0xA,0xB")

    print(f"Replaying {args.demo} ...")
    ram, infos, presses, buttons = replay(args.demo)
    n = ram.shape[0]
    print(f"{n} frames, RAM size {ram.shape[1]} bytes.\n")

    if args.find_latch:
        hits = find_latch(ram, after=args.after)
        if args.control:
            print("Cross-checking against control %s ..." % args.control)
            cram, _ci, _cp, _cb = replay(args.control)
            kept = []
            for h in hits:
                addr, _early, c1, _fr = h
                if not np.any(cram[:, addr] == c1):   # latched value never occurs
                    kept.append(h)
            print("  %d of %d candidates survive (their latched value never"
                  % (len(kept), len(hits)))
            print("  appears in a run that did not reach the state).")
            print()
            hits = kept
        if not hits:
            print("No latching address found -- no value that is absent early")
            print("and then holds steady to the end. If the ending is short,")
            print("the tail may not be flat; try a longer recording.")
            return
        print("Latching flags (a NEW value that holds to the end -- likely a")
        print("game-over / clear signal; latest-latching first):")
        print()
        print("  %-8s  %6s -> %-6s  %8s  %s" % ("ADDR", "EARLY", "LATCH", "FRAME", "VIDEO"))
        for a, early, c1, fr in hits[:args.top]:
            secs = fr / 60.0988
            print("  0x%04X  %6d -> %-6d  %8d  %02d:%06.3f"
                  % (a, early, c1, fr, int(secs // 60), secs - 60 * int(secs // 60)))
        print()
        print("Verify the top one: --watch 0x%04X -- it should flip exactly" % hits[0][0])
        print("when the game ends, and stay flipped.")
        return

    if args.find_value:
        obs = []
        for term in args.find_value.split(","):
            fr, val = term.split(":")
            obs.append((int(fr, 0), int(val, 0)))
        hits = find_value(ram, obs, tol=args.tol)
        if not hits:
            print("No address matched all observations. Check the frame numbers")
            print("(timestamp x 60.0988) and that the values are exactly on-screen,")
            print("or widen --tol.")
            return
        # tiles and BCD are more specific than a lone u8, so surface them first
        order = {"tiles": 0, "bcd": 1, "u16": 2, "u24": 2, "u8": 3}
        def rank(h):
            for k, r in order.items():
                if h[1].startswith(k):
                    return r
            return 9
        print("Candidates (verify the top one with --watch against the video):\n")
        print("  %-8s  %-8s  %s" % ("ADDR", "ENCODING", "note"))
        for base, enc, off in sorted(hits, key=lambda h: (rank(h), h[0]))[:20]:
            note = "" if off is None else ("digit offset 0x%02X" % off)
            print("  0x%04X  %-8s  %s" % (base, enc, note))
        return

    if args.compare:
        a, b = [int(x, 0) for x in args.compare.split(",")]
        ca, cb = ram[:, a], ram[:, b]
        diff_idx = np.where(ca != cb)[0]
        print(f"Comparing 0x{a:04X} and 0x{b:04X} over {n} frames:\n")
        if len(diff_idx) == 0:
            print("  IDENTICAL at every frame.")
            print("  One is a copy of the other (typically the live counter and the")
            print("  value the HUD renders from). Either works -- prefer the LOWER")
            print("  address, which is more often the primary. The distinction only")
            print("  matters at edge cases this demo never reached: crossing 100")
            print("  coins (1-Up + reset) and level transitions.")
        else:
            print(f"  DIVERGE on {len(diff_idx)} of {n} frames. First few:")
            print(f"    {'FRAME':>7}  {'VIDEO':>9}  {('0x%04X' % a):>8}  {('0x%04X' % b):>8}")
            for i in diff_idx[:15]:
                secs = i / 60.0988
                print(f"    {i:7d}  {int(secs//60):02d}:{secs%60:06.3f}  {int(ca[i]):8d}  {int(cb[i]):8d}")
            print("\n  Check these frames on the video: whichever matches the on-screen")
            print("  value is the real variable; the other lags or is a scratch copy.")
        return

    if args.watch:
        # Each term is either a bare address (prints the raw byte) or
        # "0xADDR:tilesN" -- N display-digit tiles, one digit per byte, byte =
        # digit + 0x30, most-significant first -- which prints the DECODED
        # number. HUD score/lines/level are usually stored that way.
        specs = [parse_watch(t) for t in args.watch.split(",")]

        def read(frame, spec):
            addr, kind, arg = spec
            if kind == "tiles":
                ds = [int(ram[frame, addr + i]) - 0x30 for i in range(arg)]
                ds = [d if 0 <= d <= 9 else 0 for d in ds]
                return int("".join(str(d) for d in ds))
            return int(ram[frame, addr])

        head = f"  {'FRAME':>6}  {'VIDEO':>9}  {'score':>7}"
        for addr, kind, arg in specs:
            tag = ("0x%04X:t%d" % (addr, arg)) if kind == "tiles" else ("0x%04X" % addr)
            head += f"  {tag:>10}"
        print(head)
        for frame in range(0, n, args.every):
            secs = frame / 60.0988
            line = (f"  {frame:6d}  {int(secs//60):02d}:{secs%60:06.3f}"
                    f"  {str(infos[frame].get('score')):>7}")
            for spec in specs:
                line += f"  {read(frame, spec):>10}"
            print(line)
        print("\nA correct variable moves when the thing it measures moves --")
        print("check a few of these frames against the video.")
        return

    rows = []
    if args.find == "meter":
        # Search by CAUSE rather than shape: a P-meter is whatever fills while
        # you sprint and drains when you stop. Shape alone was too weak -- the
        # display ("6 arrows plus a P") hints the state may span more than one
        # variable, and a demo without long sprints shows no shape at all.
        sprinting = sprint_mask(presses, buttons, run_frames=args.run_frames)
        frac = float(sprinting.sum()) / max(1, len(sprinting))
        print(f"Sustained sprint detected on {sprinting.sum()} of {len(sprinting)} "
              f"frames ({frac*100:.1f}%).")
        if sprinting.sum() < 60:
            print("  That is very little sprinting -- record a demo with several")
            print("  LONG flat-out runs (hold B plus a direction) for a clean result.\n")
        else:
            print()
        for addr in range(ram.shape[1]):
            sc = score_meter_correlated(ram[:, addr], sprinting)
            if sc is not None:
                col = ram[:, addr]
                d = np.diff(col)
                rows.append((sc, addr, int(col.min()), int(col.max()),
                             int(np.count_nonzero(d > 0)), int(np.count_nonzero(d < 0))))
        rows.sort(reverse=True)
        print(f"Candidates for 'meter' ({len(rows)} rise with sprinting and fall without):\n")
        print(f"  {'ADDR':>8}  {'DEC':>6}  {'RANGE':>12}  {'UPS':>5}  {'DOWNS':>5}")
        for _sc, addr, lo, hi, ups, downs in rows[:args.top]:
            print(f"  0x{addr:04X}  {addr:6d}  {lo:5d}..{hi:<5d}  {ups:5d}  {downs:5d}")
        if rows:
            print(f"\n{HINTS['meter']}")
            print(f"\nConfirm against the video:")
            print(f"    python find_game_vars.py --demo {args.demo} --watch 0x{rows[0][1]:04X}")
        else:
            print("  (none matched)")
            print("\n  Record a demo with several LONG full-speed sprints (hold B and")
            print("  a direction until the meter fills) separated by full stops.")
        return

    scorer = SCORERS[args.find]
    for addr in range(ram.shape[1]):
        col = ram[:, addr]
        s = scorer(col, n)
        if s is not None:
            d = np.diff(col)
            rows.append((s, addr, int(col.min()), int(col.max()),
                         int(np.count_nonzero(d > 0)), int(np.count_nonzero(d < 0))))
    rows.sort(reverse=True)

    print(f"Candidates for '{args.find}' ({len(rows)} matched the signature):\n")
    print(f"  {'ADDR':>8}  {'DEC':>6}  {'RANGE':>12}  {'UPS':>5}  {'DOWNS':>5}")
    for _s, addr, lo, hi, ups, downs in rows[:args.top]:
        print(f"  0x{addr:04X}  {addr:6d}  {lo:5d}..{hi:<5d}  {ups:5d}  {downs:5d}")
    if args.find == "timer":
        # A 3-digit timer stored one digit per byte shows nothing per-address.
        find_bcd_timer(ram, n, args.top)

    if not rows:
        print("  (none matched this signature)")
        if args.find == "meter":
            print("\n  A P-meter only shows its signature if the recording actually")
            print("  fills it. Record a demo with several LONG full-speed sprints")
            print("  (hold B and run flat out until the meter fills) and stops in")
            print("  between, then search again.")
        elif args.find == "timer":
            print("\n  If the BCD search above also found nothing, the timer may be")
            print("  packed (two digits per byte) or simply not run in this demo.")
        else:
            print("\n  Record a demo that exercises this variable more.")
        return

    print(f"\n{HINTS[args.find]}")
    print(f"\nConfirm the top pick against the video:")
    print(f"    python find_game_vars.py --demo {args.demo} --watch 0x{rows[0][1]:04X}")


if __name__ == "__main__":
    main()
