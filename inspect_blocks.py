#!/usr/bin/env python3
"""
Show what the ? block detector sees, as an overlay you can eyeball.

Background tiles never appear in OAM, so inspect_sprites.py structurally cannot
find a ? block -- yet those blocks are where the coins and power-ups are. They
are easy to find in the colour frame instead: a fixed 16x16 tile in a fixed
palette.

The distinction that matters is LIVE vs SPENT. A hit block turns into a solid
block of the same orange, so a colour-only test keeps reporting a target that is
already used up and the agent goes on butting at it. Measured across a whole
World 1-1 recording, the interior pink highlight separates them with no overlap:
live blocks read 0.09-0.25 (the spread is the block's pulse animation) and spent
blocks read exactly 0.00.

WHY RUN THIS: the thresholds in train.py were measured from an mp4, which is
lossy. The emulator's own frames are the clean NES palette -- close, but not
identical. This is how you confirm they still hold before spending a training
run on them. Green boxes are live blocks, red are spent.

Usage:
    # Against the live emulator (what training actually sees)
    python inspect_blocks.py --game SuperMarioBros3-Nes-v0 --shots ./block_shots

    # Against a recording, including an mp4 from play_and_record.py
    python inspect_blocks.py --video ./run.mp4 --shots ./block_shots
    python inspect_blocks.py --demo ./human_demos/run.bk2 --shots ./block_shots
"""
import argparse
import os

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("inspect_blocks.py needs opencv: pip install opencv-python")

from train import QBLOCK_HUD_FRAC, detect_qblocks


def frames_from_video(path):
    cap = cv2.VideoCapture(path)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()


def frames_from_demo(path):
    import stable_retro as retro
    movie = retro.Movie(path)
    movie.step()
    env = retro.make(game=movie.get_game(), state=None,
                     use_restricted_actions=retro.Actions.ALL,
                     players=movie.players, render_mode="rgb_array")
    env.initial_state = movie.get_state()
    obs, _info = env.reset()
    while movie.step():
        keys = [movie.get_key(i, 0) for i in range(env.num_buttons)]
        obs, _r, term, trunc, _i = env.step(keys)
        yield obs
        if term or trunc:
            break
    env.close()


def frames_from_game(game, state, max_frames):
    import stable_retro as retro
    env = retro.make(game=game, state=state or retro.State.DEFAULT, render_mode="rgb_array")
    buttons = env.unwrapped.buttons
    run_right = np.array([b in ("RIGHT", "B") for b in buttons], dtype=bool)
    obs, _info = env.reset()
    for _ in range(max_frames):
        obs, _r, term, trunc, _i = env.step(run_right)
        yield obs
        if term or trunc:
            obs, _info = env.reset()
    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--game", help="Run the live emulator -- what training actually sees")
    src.add_argument("--video", help="An mp4 to scan instead")
    src.add_argument("--demo", help="A .bk2 recording to scan instead")
    p.add_argument("--state", default=None)
    p.add_argument("--shots", default=None, help="Directory for overlay screenshots")
    p.add_argument("--every", type=int, default=120, help="Save/report every N frames (default: %(default)s)")
    p.add_argument("--hud-frac", type=float, default=QBLOCK_HUD_FRAC, help="Bottom fraction of the frame to ignore as HUD (default: %(default)s)")
    p.add_argument("--max-frames", type=int, default=3000)
    args = p.parse_args()

    if args.shots:
        os.makedirs(args.shots, exist_ok=True)

    if args.game:
        frames = frames_from_game(args.game, args.state, args.max_frames)
    elif args.video:
        frames = frames_from_video(args.video)
    else:
        frames = frames_from_demo(args.demo)

    n = live_frames = spent_total = live_total = 0
    live_max = 0
    for rgb in frames:
        if n >= args.max_frames:
            break
        found = detect_qblocks(rgb, args.hud_frac)
        live = [t for t in found if t[2]]
        spent = [t for t in found if not t[2]]
        live_total += len(live)
        spent_total += len(spent)
        live_max = max(live_max, len(live))
        if live:
            live_frames += 1

        if n % args.every == 0:
            print(f"frame {n:6d}  {len(live)} live, {len(spent)} spent")
            for x, y, is_live in found:
                print(f"    {'LIVE ' if is_live else 'spent'} at x={x:3d} y={y:3d}")
            if args.shots:
                vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for x, y, is_live in found:
                    colour = (0, 255, 0) if is_live else (0, 0, 255)
                    cv2.rectangle(vis, (x, y), (x + 15, y + 15), colour, 1)
                path = os.path.join(args.shots, f"f{n:06d}.png")
                cv2.imwrite(path, cv2.resize(vis, None, fx=3, fy=3,
                                             interpolation=cv2.INTER_NEAREST))
                print(f"    [{path}]")
        n += 1

    print("=" * 68)
    if not n:
        print("No frames read.")
        return
    pct = 100.0 * live_frames / n
    print(f"{n} frames. Live ? blocks on {live_frames} of them ({pct:.0f}%), "
          f"max {live_max} at once.")
    print(f"Detections: {live_total} live, {spent_total} spent.")
    if live_frames == 0:
        print("\nNOTHING DETECTED. The colour thresholds (QBLOCK_* in train.py) were")
        print("measured on World 1-1; another palette needs different ones. Training")
        print("refuses to start in this state rather than feeding zeroed slots.")
    elif spent_total == 0:
        print("\nNo spent blocks seen -- fine if nothing was hit in this clip, but it")
        print("means the live/spent split is unconfirmed here. Check a clip where a")
        print("block actually gets used.")
    else:
        print("\nGreen boxes are live blocks, red are spent. Confirm on the shots that")
        print("every ? block is boxed and nothing else is.")


if __name__ == "__main__":
    main()
