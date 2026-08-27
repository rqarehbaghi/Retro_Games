#!/usr/bin/env python3
"""
Turn a set of training checkpoints (from train.py) into short "progress
reel" videos -- a few seconds of gameplay per checkpoint, labeled with
the iteration number, so you can watch the agent go from random flailing
to actually playing. Produces two files:

    progress_youtube.mp4  -- 1920x1080, landscape
    progress_shorts.mp4   -- 1080x1920, portrait (Instagram/TikTok, also
                              works fine as a YouTube Short)

Reuses the exact same agent-play + bk2-to-mp4 pipeline as
play_and_record.py -- this isn't a separate recording method, just that
same one run once per checkpoint with a short clip length.

Usage:
    python make_progress_reel.py --game SuperMarioBros3-Nes-v0 \\
        --state Level1-1 --checkpoint-dir ./checkpoints \\
        --iterations 1 100 1000 10000
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

from play_and_record import find_new_bk2, play_agent_episode, render_to_mp4
from train import safe_name

# Ships with the fonts-dejavu-core apt package -- see README.
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def make_clip(game, state, checkpoint_path, max_steps, work_dir, label):
    record_dir = os.path.join(work_dir, f"raw_{label}")
    os.makedirs(record_dir, exist_ok=True)
    before = set(glob.glob(os.path.join(record_dir, "*.bk2")))

    play_agent_episode(
        game=game, state=state, model_path=checkpoint_path,
        max_steps=max_steps, record_dir=record_dir, render=False,
    )

    bk2_path = find_new_bk2(record_dir, before)
    if bk2_path is None:
        return None
    return render_to_mp4(bk2_path)


def build_segment(src_mp4, label_text, width, height, clip_seconds, out_path):
    """Fits src_mp4 into a width x height canvas (letterboxed/pillarboxed
    as needed, since NES footage isn't the same shape as either target),
    stamps the iteration label on top, and caps it at clip_seconds."""
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"drawtext=fontfile={FONT}:text='{label_text}':fontcolor=white:"
        f"fontsize={max(24, width // 20)}:x=(w-text_w)/2:y=40:"
        f"box=1:boxcolor=black@0.5:boxborderw=10"
    )
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-i", src_mp4,
            "-vf", vf, "-r", "60", "-t", str(clip_seconds),
            "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            out_path,
        ],
        check=True,
    )


def concat_segments(segment_paths, out_path, work_dir):
    filelist = os.path.join(work_dir, f"filelist_{os.path.basename(out_path)}.txt")
    with open(filelist, "w") as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0",
            "-i", filelist, "-c", "copy", out_path,
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True, help="Same game id used for training")
    parser.add_argument("--state", default=None)
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Parent folder passed to train.py -- the game-named subfolder inside it is read automatically")
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 100], help="Must match checkpoints train.py actually saved for this game (default: %(default)s, matching train.py's own default run length)")
    parser.add_argument("--clip-seconds", type=int, default=8, help="Gameplay seconds per checkpoint in the YouTube version (default: %(default)s)")
    parser.add_argument("--clip-seconds-short", type=int, default=4, help="Gameplay seconds per checkpoint in the IG/TikTok version (default: %(default)s)")
    parser.add_argument("--out-dir", default="./progress_reels")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(args.checkpoint_dir, safe_name(args.game))
    os.makedirs(args.out_dir, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix="progress_reel_")

    clips = {}
    longest_needed = max(args.clip_seconds, args.clip_seconds_short)
    for it in args.iterations:
        ckpt_path = os.path.join(checkpoint_dir, f"iter_{it}.zip")
        if not os.path.exists(ckpt_path):
            print(f"Skipping iteration {it}: no checkpoint found at {ckpt_path}")
            continue
        print(f"Playing checkpoint iter_{it} ...")
        mp4 = make_clip(
            args.game, args.state, ckpt_path,
            max_steps=longest_needed * 60,  # NES runs at 60fps
            work_dir=work_dir, label=f"iter{it}",
        )
        if mp4:
            clips[it] = mp4
        else:
            print(f"Warning: no clip produced for iteration {it} (episode may have ended instantly)")

    if not clips:
        print("No clips were produced -- nothing to assemble.")
        shutil.rmtree(work_dir, ignore_errors=True)
        sys.exit(1)

    formats = {
        "youtube": (1920, 1080, args.clip_seconds, "progress_youtube.mp4"),
        "shorts": (1080, 1920, args.clip_seconds_short, "progress_shorts.mp4"),
    }

    for name, (width, height, seconds, filename) in formats.items():
        segments = []
        for it in sorted(clips):
            label = f"Iteration {it:,}"
            seg_path = os.path.join(work_dir, f"{name}_{it}.mp4")
            build_segment(clips[it], label, width, height, seconds, seg_path)
            segments.append(seg_path)
        out_path = os.path.join(args.out_dir, filename)
        concat_segments(segments, out_path, work_dir)
        print(f"Done: {out_path}")

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
