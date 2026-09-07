#!/usr/bin/env python3
"""
Render a MULTI-PLAYER .bk2 to MP4, working around a stable-retro bug.

stable_retro.scripts.playback_movie does, once per frame:

    if movie.players > 1:
        for p in range(movie.players):
            score[p] += reward[p]

but a game whose integration defines no PER-PLAYER scenario returns one
SCALAR reward no matter how many players the env has. SuperMarioBros3 is such
a game -- verified: retro.make(..., players=2).step() returns 0.0, a float. So
replaying any two-player recording dies with

    TypeError: 'float' object is not subscriptable

The video pipeline itself is fine; only the score bookkeeping is wrong, and the
score is not something this project uses. So rather than reimplement ffmpeg
piping and A/V sync, this reuses playback_movie and hands it an emulator whose
step() reports one reward PER PLAYER.

Run as a subprocess, not imported: stable-retro allows only ONE emulator
instance per process, and the caller has usually just finished playing in one.

Usage:
    python render_bk2.py <movie.bk2> [output.mp4]
"""
import os
import sys


def render(bk2_path, mp4_path=None):
    """Replay bk2_path and write an MP4 beside it (or at mp4_path)."""
    from stable_retro.scripts import playback_movie as pm
    try:
        import custom_integrations
        custom_integrations.register()
    except Exception:
        pass

    if mp4_path is None:
        mp4_path = os.path.splitext(bk2_path)[0] + ".mp4"

    emulator, movie, _duration = pm.load_movie(bk2_path)
    players = movie.players
    inner_step = emulator.step

    def step(action):
        obs, reward, terminated, truncated, info = inner_step(action)
        if players > 1 and not hasattr(reward, "__len__"):
            # The scalar is a whole-game reward; there is nothing to split it
            # by. Report it against player 1 and zero for the rest -- this only
            # feeds playback_movie's score column, which nothing here reads.
            reward = [reward] + [0.0] * (players - 1)
        return obs, reward, terminated, truncated, info

    emulator.step = step
    try:
        pm.playback_movie(emulator, movie, video_file=mp4_path)
    finally:
        try:
            emulator.close()
        except Exception:                                       # noqa: BLE001
            pass
    return mp4_path if os.path.exists(mp4_path) else None


def main():
    if not 2 <= len(sys.argv) <= 3:
        raise SystemExit(__doc__.strip())
    out = render(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
    if not out:
        raise SystemExit("render produced no file")
    print(out)


if __name__ == "__main__":
    main()
