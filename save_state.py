#!/usr/bin/env python3
"""
Extract emulator save states from a recording, so training can start anywhere.

Why this exists: training always resets to the integration's one default state,
which for SuperMarioBros3-Nes-v0 is the start of World 1-1. Every episode the
agent has ever played begins there, so 100% of its experience is level 1 -- and
dropped into level 2 it does the only thing it knows (run right) and dies. That
is missing training data, not bad tuning, and no reward change fixes it.

This walks a .bk2 and writes the emulator's state at chosen moments as .state
files that train.py can load with --state-file. Point training at level 2's
state and it trains on level 2.

Finding the moments:
  --frame N        save at exactly frame N (read one off audit_ram.py's
                   timestamped output, or inspect_progress --demo)
  --at-transition  save automatically just after each level/area change,
                   detected the same way RewardShaper detects a clear: level
                   position collapsing without a life being lost. Reliable
                   because the page byte absorbs the low byte's 256-wraps, so
                   ordinary travel never jumps backwards.

Usage:
    # See what's in the recording and where the transitions are
    python save_state.py --demo ./human_demos_v2/<file>.bk2 --list

    # Save a state at each level transition
    python save_state.py --demo ./human_demos_v2/<file>.bk2 \\
        --at-transition --out-dir ./states --prefix smb3

    # Then train on one of them
    python train.py --game SuperMarioBros3-Nes-v0 --state-file ./states/smb3_02.state
"""
import argparse
import gzip
import json
import os

import numpy as np
import stable_retro as retro

FPS = 60.0988


def stamp(frame):
    secs = frame / FPS
    return f"{int(secs // 60):02d}:{secs % 60:06.3f}"


def progress_reader(game, config="games.json"):
    """How to read level position for this game, from games.json, so transition
    detection matches what training uses rather than a second opinion."""
    path = config if os.path.isabs(config) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), config)
    try:
        with open(path) as fh:
            spec = (((json.load(fh).get("games") or {}).get(game) or {})
                    .get("variables") or {}).get("progress") or {}
    except Exception:
        spec = {}
    if spec.get("source") == "info16":
        hi = int(str(spec["address_high"]), 0)
        key = spec.get("key", "hpos")
        return lambda ram, info: (None if info.get(key) is None
                                  else int(info[key]) + (int(ram[hi]) << 8))
    if spec.get("source") in ("ram", "ram16"):
        lo = int(str(spec["address"]), 0)
        hi = int(str(spec["address_high"]), 0) if spec.get("address_high") else None
        return lambda ram, info: int(ram[lo]) + ((int(ram[hi]) << 8) if hi is not None else 0)
    return lambda ram, info: info.get("hpos")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", required=True, help="A .bk2 recording to walk")
    p.add_argument("--frame", type=int, action="append", default=[], help="Save at this frame (repeatable)")
    p.add_argument("--at-transition", action="store_true", help="Save just after every level/area change")
    p.add_argument("--settle", type=int, default=90, help="Frames to let a transition finish before saving, so the state lands in the new level rather than mid-wipe (default: %(default)s)")
    p.add_argument("--list", action="store_true", help="Only report what's in the recording; write nothing")
    p.add_argument("--out-dir", default="./states")
    p.add_argument("--prefix", default="state")
    args = p.parse_args()

    if not (args.frame or args.at_transition or args.list):
        raise SystemExit("Pass --frame N, --at-transition, or --list")

    movie = retro.Movie(args.demo)
    movie.step()
    game = movie.get_game()
    env = retro.make(game=game, state=None,
                     use_restricted_actions=retro.Actions.ALL,
                     players=movie.players, render_mode="rgb_array")
    env.initial_state = movie.get_state()
    env.reset()
    read_progress = progress_reader(game)

    os.makedirs(args.out_dir, exist_ok=True)
    want = set(args.frame)
    saved = []
    pending = None          # frames remaining before saving a transition state
    prev_progress = None
    prev_lives = None
    frame = 0

    print(f"Walking {args.demo} ({game})\n")
    while movie.step():
        keys = [movie.get_key(i, 0) for i in range(env.num_buttons)]
        _obs, _rew, terminated, truncated, info = env.step(keys)
        ram = env.get_ram()
        progress = read_progress(ram, info)
        lives = info.get("lives")

        # A large backward jump in level position without losing a life means a
        # level or area change -- the same test RewardShaper uses for a clear.
        if (args.at_transition and pending is None
                and progress is not None and prev_progress is not None
                and lives is not None and prev_lives is not None):
            if progress < prev_progress - 200 and lives >= prev_lives:
                print(f"  transition at frame {frame} [{stamp(frame)}]: "
                      f"position {prev_progress} -> {progress}")
                pending = args.settle
        if pending is not None:
            pending -= 1
            if pending <= 0:
                pending = None
                if not args.list:
                    path = os.path.join(args.out_dir, f"{args.prefix}_{len(saved)+1:02d}.state")
                    with gzip.open(path, "wb") as fh:
                        fh.write(env.em.get_state())
                    saved.append((frame, path))
                    print(f"    saved {path}  (position now {progress}, lives {lives})")
                else:
                    saved.append((frame, "(not written: --list)"))

        if frame in want and not args.list:
            path = os.path.join(args.out_dir, f"{args.prefix}_frame{frame}.state")
            with gzip.open(path, "wb") as fh:
                fh.write(env.em.get_state())
            saved.append((frame, path))
            print(f"  saved {path} at frame {frame} [{stamp(frame)}]")

        prev_progress = progress if progress is not None else prev_progress
        prev_lives = lives if lives is not None else prev_lives
        frame += 1
        if terminated or truncated:
            break
    env.close()

    print(f"\nWalked {frame} frames, {len(saved)} state(s).")
    if saved and not args.list:
        print("\nTrain from one with:")
        print(f"    python train.py --game {game} --state-file {saved[-1][1]}")
        print("\nNOTE the agent has only ever trained on the default start state,")
        print("so a state from a later level is genuinely new territory for it --")
        print("expect it to be poor there until it trains on that state too.")
    elif not saved:
        print("No transitions found. If the recording never finishes a level,")
        print("record one that does, or pass --frame N for a specific moment.")


if __name__ == "__main__":
    main()
