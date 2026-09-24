#!/usr/bin/env python3
"""Generate the frozen start-state suite the checkpoint sweep is measured on.

    python tools/progression/suite.py --game TetrisTime-Nes-v0 \
        --from-recording <a .bk2 that starts at the console boot> --count 32

Rules this enforces (CHECKPOINT_PROGRESSION_VIDEO_PLAN.md):

- ONE procedure, ONE variable. Every state is produced by replaying the SAME
  recorded menu navigation from the same power-on state; the only thing that
  differs is how many idle frames are inserted before it, which is what moves
  the game's RNG. Same mode, same level, empty board, zero counters, same
  player setup.
- NO trained checkpoint is involved. The menu inputs come from a human
  recording and nothing chooses them per state.
- Independence is proven by BEHAVIOUR, not by file hash. Two states can differ
  byte for byte and still deal identical pieces, which would make the suite a
  single trial repeated N times. Each candidate is probed with the same fixed
  input script and its first N piece types recorded; a candidate whose piece
  prefix matches one already accepted is rejected.
- The manifest is written and hashed BEFORE any checkpoint is evaluated, and
  records everything that can change an outcome.
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import custom_integrations                                       # noqa: E402
import stable_retro as retro                                     # noqa: E402
from rl.env import TrainingSpec, build_grid_observation          # noqa: E402
from rl.vars import GameVars                                     # noqa: E402
from tools.progression import ids                                # noqa: E402

PREFIX_PIECES = 20        # piece types recorded per state to prove independence
PROBE_FRAMES = 6000       # hard bound on the probe, whatever the game does


def menu_inputs(bk2, game, var_player):
    """P1/P2 buttons from the recording, up to the first frame of play.

    Read from a real recording rather than written by hand: the menu path for
    a game is exactly the kind of thing that is guessed wrong."""
    movie = retro.Movie(bk2)
    movie.step()
    env = retro.make(game=movie.get_game(), state=None,
                     use_restricted_actions=retro.Actions.ALL,
                     players=movie.players, render_mode=None)
    start_state = movie.get_state()
    if start_state:
        env.initial_state = start_state
    env.reset()
    nb = env.num_buttons
    gv = GameVars(game, entry=TrainingSpec(game, {"player": var_player}).entry)
    keys, frame = [], 0
    while movie.step() and frame < 20000:
        row = [bool(movie.get_key(i, p)) for p in range(movie.players) for i in range(nb)]
        env.step(row)
        keys.append(row)
        if frame > 60 and gv.read("game_mode", env.get_ram(), {}) == 0:
            break
        frame += 1
    players, buttons = movie.players, list(env.unwrapped.buttons)
    env.close()
    return keys, start_state, players, buttons


def piece_prefix(env, gv, player, nb, players, count=PREFIX_PIECES):
    """The first `count` piece types, under a FIXED script applied to every
    candidate alike.

    With no input at all the pieces pile in one column and the game ends after
    about a dozen: not enough to compare. The script spreads them (alternating
    sides, soft dropping) so the probe reaches `count` pieces. It is identical
    for every candidate, so a difference in the recorded prefix is a difference
    in the game's sequence, not in what we did."""
    var = "piece_type_p%d" % player
    seen, last, frames = [], None, 0
    while len(seen) < count and frames < PROBE_FRAMES:
        i = len(seen)
        phase = frames % 16
        press = []
        if phase < 6:
            press = ["LEFT"] if i % 2 == 0 else ["RIGHT"]
        elif phase < 14:
            press = ["DOWN"]
        act = [False] * (nb * players)
        for name in press:
            act[(player - 1) * nb + env.unwrapped.buttons.index(name)] = True
        env.step(act)
        frames += 1
        ram = env.get_ram()
        kind = gv.read(var, ram, {})
        if kind is not None and kind != last:
            if kind:                       # 0 shows between pieces
                seen.append(int(kind))
            last = kind
        if not gv.read("game_over_p%d" % player, ram, {}):
            break
    return seen


def empty_and_zeroed(frame, ram, spec, gv, player):
    """A usable start: board empty, counters at zero, agent alive."""
    board = build_grid_observation(frame, ram, spec.grid, gv, player)
    filled = int((board[:spec.grid.get("rows", 20) * spec.grid.get("cols", 10)] > 0.5).sum())
    lines = gv.read("lines_p%d" % player, ram, {}, 0) or 0
    level = gv.read("level_p%d" % player, ram, {}, 0) or 0
    alive = gv.read("game_over_p%d" % player, ram, {})
    return filled == 0 and int(lines) == 0 and int(level) == 0 and bool(alive), {
        "filled": filled, "lines": int(lines), "level": int(level), "alive": bool(alive)}


def build_candidate(game, keys, start_state, players, delay, spec, gv, player):
    """Replay the menu with `delay` idle frames inserted first, stop at play."""
    env = retro.make(game=game, state=None, use_restricted_actions=retro.Actions.ALL,
                     players=players, render_mode="rgb_array")
    if start_state:
        env.initial_state = start_state
    frame, _ = env.reset()
    nb = env.num_buttons
    idle = [False] * (nb * players)
    for _ in range(delay):
        frame, *_ = env.step(idle)
    for row in keys:
        frame, *_ = env.step(row)
    # Settle onto the first frames of live play.
    for _ in range(30):
        if gv.read("game_mode", env.get_ram(), {}) == 0:
            break
        frame, *_ = env.step(idle)
    ok, stats = empty_and_zeroed(frame, env.get_ram(), spec, gv, player)
    state_bytes = env.em.get_state()
    return env, state_bytes, ok, stats


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="TetrisTime-Nes-v0")
    p.add_argument("--from-recording", required=True,
                   help="a .bk2 that starts at the console boot and reaches play")
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--player", type=int, default=2, help="which player the agent drives")
    p.add_argument("--placement-cap", type=int, default=500,
                   help="recorded in the manifest; the sweep must use this value")
    p.add_argument("--prefix-name", default="prog",
                   help="state files are <prefix>_NN.state in the game's integration folder")
    p.add_argument("--out", default=os.path.join(ROOT, "progression_out"))
    p.add_argument("--max-delay", type=int, default=400,
                   help="how many idle-frame offsets to try before giving up")
    a = p.parse_args()

    custom_integrations.register()
    spec = TrainingSpec(a.game, {"player": a.player})
    gv = GameVars(a.game, entry=spec.entry)
    folder = os.path.join(custom_integrations.INTEGRATIONS_DIR, a.game)
    os.makedirs(a.out, exist_ok=True)

    print("Reading the menu path from %s ..." % os.path.basename(a.from_recording))
    keys, start_state, players, buttons = menu_inputs(a.from_recording, a.game, a.player)
    print("  %d frames of recorded input, %d players, buttons %s"
          % (len(keys), players, [b for b in buttons if b]))

    states, prefixes, rejected = [], {}, []
    delay = 0
    while len(states) < a.count and delay < a.max_delay:
        env, state_bytes, ok, stats = build_candidate(
            a.game, keys, start_state, players, delay, spec, gv, a.player)
        if not ok:
            rejected.append({"delay": delay, "why": "not a clean start", **stats})
            env.close()
            delay += 1
            continue
        prefix = tuple(piece_prefix(env, gv, a.player, env.num_buttons, players))
        env.close()
        if len(prefix) < PREFIX_PIECES:
            rejected.append({"delay": delay, "why": "probe ended after %d pieces" % len(prefix)})
        elif prefix in prefixes:
            rejected.append({"delay": delay, "why": "piece prefix duplicates delay %d"
                             % prefixes[prefix]})
        else:
            prefixes[prefix] = delay
            name = "%s_%02d" % (a.prefix_name, len(states) + 1)
            path = os.path.join(folder, name + ".state")
            with gzip.open(path, "wb") as fh:
                fh.write(state_bytes)
            states.append({"name": name, "file": os.path.basename(path),
                           "delay_frames": delay, "piece_prefix": list(prefix),
                           "sha256": ids.sha256_file(path)})
            print("  [%2d/%2d] %s  delay %3d  pieces %s"
                  % (len(states), a.count, name, delay, list(prefix[:8])))
        delay += 1

    if len(states) < a.count:
        sys.exit("Only %d of %d states were accepted after %d delays. Rejections: %s"
                 % (len(states), a.count, delay, rejected[:5]))

    manifest = {
        "game": a.game,
        "player": a.player,
        "players": players,
        "deterministic": True,
        "placement_cap": a.placement_cap,
        "prefix_pieces": PREFIX_PIECES,
        "states": states,
        "generation": {
            "command": "tools/progression/suite.py --game %s --from-recording %s --count %d"
                       % (a.game, os.path.basename(a.from_recording), a.count),
            "recording": os.path.basename(a.from_recording),
            "recording_sha256": ids.sha256_file(a.from_recording),
            "method": "replay the recorded menu navigation with N idle frames inserted "
                      "first; only that offset varies. No trained checkpoint is involved.",
            "probe": "first %d piece types under one fixed input script, identical for "
                     "every candidate" % PREFIX_PIECES,
        },
        "rom_sha256": ids.rom_hash(a.game),
        "config_sha256": ids.config_hash(spec),
        "code_commit": ids.code_commit(),
        "selection_rule": {
            "median_state": "rank 16 of 32 by the ZERO-VALUE CONTROL's placements, ascending",
            "hard_state": "rank 4 of 32 (10th percentile) by the same ranking",
            "ties": "broken on state name, ascending",
            "note": "checkpoint results are never consulted when choosing displayed states",
        },
        "rejected": rejected,
    }
    body = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_path = os.path.join(a.out, "suite_manifest.json")
    with open(manifest_path, "w") as fh:
        fh.write(body)
    digest = ids.sha256_text(body)
    with open(manifest_path + ".sha256", "w") as fh:
        fh.write(digest + "  suite_manifest.json\n")
    print("\n%d states written to %s" % (len(states), folder))
    print("manifest %s\n  sha256 %s" % (manifest_path, digest))
    print("The suite is now FROZEN: evaluate checkpoints only against this manifest.")


if __name__ == "__main__":
    main()
