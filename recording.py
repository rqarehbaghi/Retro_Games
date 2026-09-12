#!/usr/bin/env python3
"""
The .bk2 files a session produced, and turning one into an MP4.

These three functions were living in play_and_record.py, and studio.py,
play_engine.py and two tools all reached into that 563-line CLI script to get
them -- dragging in an argument parser, an SMB3 world-map navigator and a
policy builder to ask "which replay did I just write". They are the shared
part, so they live on their own here. play_and_record.py re-exports them, so
anything importing them from there still works.

Rendering goes through render_bk2.py in a SUBPROCESS, never in-process:
stable-retro allows one emulator per process, and every caller here has
already had one open.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _this_session(path, before, started_at):
    """Whether a .bk2 was written or REWRITTEN during this session.

    stable-retro numbers recordings from -000000 for every new env/process, so
    a rerun into the same folder OVERWRITES the previous run's file instead of
    creating a new name. Path membership against `before` alone is therefore
    wrong: the fresh recording has a path that was already there, gets filtered
    out as pre-existing, and the run reports no recording at all.
    """
    if path not in before:
        return True
    # 1s slack for coarse filesystem mtime resolution.
    return started_at is not None and os.path.getmtime(path) >= started_at - 1.0


def find_new_bk2(record_dir, before, started_at=None):
    """The .bk2 THIS session wrote, or None.

    None when nothing was written or rewritten this session -- rendering a
    genuinely stale replay from an earlier session as if it were this run is
    worse than reporting the failure.
    """
    new = [f for f in glob.glob(os.path.join(record_dir, "*.bk2"))
           if _this_session(f, before, started_at)]
    return max(new, key=os.path.getmtime) if new else None


def find_all_new_bk2(record_dir, before, started_at=None):
    """Every .bk2 this session wrote, oldest first.

    --on-death restart resets the env per attempt, and stable-retro starts a
    NEW movie file on each reset, so a session can produce several.
    """
    return sorted((f for f in glob.glob(os.path.join(record_dir, "*.bk2"))
                   if _this_session(f, before, started_at)),
                  key=os.path.getmtime)


def render_to_mp4(bk2_path, mp4_path=None, check=True):
    """Replay a .bk2 through the emulator and encode it. Needs ffmpeg on PATH.

    Writes at the emulator's native resolution (NES: ~256x224); stable-retro's
    playback tool has no scaling option.

    Always renders through render_bk2.py rather than the stock

        python -m stable_retro.scripts.playback_movie

    for two reasons, both of which bit real recordings: the stock subprocess
    does NOT register this repo's CUSTOM integrations, so replaying a game like
    TetrisTime-Nes-v0 died with "No romfiles found"; and it does
    score[p] += reward[p] for any >1-player movie, which crashes on a game
    whose scenario returns one scalar reward. render_bk2.py fixes both and is a
    plain pass-through for an ordinary single-player movie.
    """
    mp4_path = mp4_path or os.path.splitext(bk2_path)[0] + ".mp4"
    print("Rendering %s to MP4 (this replays the run through the emulator)..."
          % bk2_path)
    subprocess.run([sys.executable, os.path.join(HERE, "render_bk2.py"),
                    bk2_path, mp4_path], check=check)
    return mp4_path if os.path.exists(mp4_path) else None
