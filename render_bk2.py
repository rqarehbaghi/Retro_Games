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
import time


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

    # playback_movie starts ffmpeg as a TCP LISTENER, sleeps a fixed 0.3s, then
    # connects exactly ONCE. When ffmpeg needs longer than that to bind -- a
    # loaded machine, a cold page cache -- the connect is refused and the whole
    # render dies with ConnectionRefusedError, costing the run its videos even
    # though the .bk2 is perfectly good. That is a startup race, not a real
    # failure, so retry it with more room each time. The movie must be RELOADED
    # per attempt: a failed run leaves its playback position advanced.
    attempts = 4
    for attempt in range(attempts):
        emulator, movie, _duration = pm.load_movie(bk2_path)
        players = movie.players
        inner_step = emulator.step

        # Bind per-attempt values as defaults: a plain closure would capture
        # the loop variables and read whichever attempt happened to run last.
        def step(action, _inner=inner_step, _players=players):
            obs, reward, terminated, truncated, info = _inner(action)
            if _players > 1 and not hasattr(reward, "__len__"):
                # The scalar is a whole-game reward; there is nothing to split
                # it by. Report it against player 1 and zero for the rest --
                # this only feeds playback_movie's score column, which nothing
                # here reads.
                reward = [reward] + [0.0] * (_players - 1)
            return obs, reward, terminated, truncated, info

        emulator.step = step
        try:
            pm.playback_movie(emulator, movie, video_file=mp4_path)
            break
        except ConnectionRefusedError:
            if attempt == attempts - 1:
                raise
            print("[render] ffmpeg was not listening yet (attempt %d/%d); "
                  "retrying with more room..." % (attempt + 1, attempts))
            time.sleep(1.0 * (attempt + 1))
        finally:
            try:
                emulator.close()
            except Exception:                                   # noqa: BLE001
                pass
    return mp4_path if os.path.exists(mp4_path) else None


def main():
    # Arguments were read straight out of sys.argv, so `--help` was taken as a
    # filename and died inside the emulator with "Could not load movie" -- as
    # did any typo'd path. Check what we were given before opening anything.
    args = [a for a in sys.argv[1:] if a not in ("-h", "--help")]
    if len(args) != len(sys.argv[1:]) or not 1 <= len(args) <= 2:
        print(__doc__.strip())
        print("")
        print("usage: render_bk2.py <replay.bk2> [output.mp4]")
        print("       output defaults to the .bk2's own name with .mp4")
        raise SystemExit(0 if len(args) != len(sys.argv[1:]) else 2)
    if not os.path.exists(args[0]):
        raise SystemExit("No such replay: %s" % args[0])
    if not args[0].endswith(".bk2"):
        print("warning: %s is not a .bk2; the emulator will probably refuse it"
              % args[0])
    out = render(args[0], args[1] if len(args) == 2 else None)
    if not out:
        raise SystemExit("render produced no file")
    print(out)


if __name__ == "__main__":
    main()
