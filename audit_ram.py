#!/usr/bin/env python3
"""
Audit EVERY value the reward shaping reads, timestamped against the video.

This is the single cross-check for all the RAM addresses and info keys the
training pipeline depends on. Each was found by a different tool at a different
time (find_ram_variable.py, inspect_progress.py, probe_after_death.py), and a
wrong one silently corrupts training rather than erroring -- which has happened
repeatedly in this project (hpos read as level position, 0x053C read from map
frames, a death-sequence byte that flipped too late).

It replays a recording (or plays a checkpoint), prints every tracked value on a
fixed interval with VIDEO timestamps, and prints an event line the instant
anything meaningful changes -- power-up gained, hit taken, life lost, play state
left. Render the same recording to MP4, scrub to a printed timestamp, and
confirm with your own eyes that the numbers describe what is on screen.

VALUES AUDITED
    info keys : lives, score, time, hpos     (published by the integration)
    RAM       : --progress-address  (camera scroll; SMB3 verified 0x00CF)
                --powerup-address   (power tier;   SMB3 verified 0x00ED)
                --playstate-address (game mode;    SMB3 verified 0x07F1, 5=play)
    derived   : progress = camera scroll + hpos  (the training progress signal)

Usage:
    # Best: replay a human demo -- it contains real power-ups, hits and deaths
    python audit_ram.py --demo ./human_demos_v2/<file>.bk2 --render-video

    # Or watch a trained checkpoint play
    python audit_ram.py --model ./checkpoints_v9/SuperMarioBros3-Nes-v0/latest_iter_500.zip

Then open the MP4 the run reports and check a few printed timestamps.
"""
import argparse
import os
import sys

import numpy as np
import stable_retro as retro

# NES NTSC frame rate. stable-retro's playback encodes at ~60.10 fps (visible in
# its ffmpeg output), so video time = frame / FPS lines up with the MP4.
FPS = 60.0988


def stamp(frame):
    """Video timestamp MM:SS.mmm for a frame index."""
    secs = frame / FPS
    return f"{int(secs // 60):02d}:{secs % 60:06.3f}"


class Tracker:
    """Collects every audited value per frame and reports changes as events."""

    def __init__(self, progress_addr, powerup_addr, playstate_addr, playstate_value,
                 progress_addr_high=None):
        self.progress_addr = progress_addr
        self.progress_addr_high = progress_addr_high
        self.powerup_addr = powerup_addr
        self.playstate_addr = playstate_addr
        self.playstate_value = playstate_value
        self.prev = {}
        self.events = []
        self.series = {k: [] for k in
                       ("lives", "score", "time", "hpos", "scroll", "progress",
                        "powerup", "playstate")}
        self.first_playstate_exit = None
        self.first_life_loss = None

    def read(self, ram, info):
        vals = {
            "lives": info.get("lives"),
            "score": info.get("score"),
            "time": info.get("time"),
            "hpos": info.get("hpos"),
        }
        scroll = None
        if self.progress_addr is not None:
            scroll = int(ram[self.progress_addr])
            if self.progress_addr_high is not None:
                scroll += int(ram[self.progress_addr_high]) << 8
        vals["scroll"] = scroll
        # The exact quantity RewardShaper uses when --progress-add-screen-x is on.
        vals["progress"] = (scroll + vals["hpos"]
                            if scroll is not None and vals["hpos"] is not None else None)
        vals["powerup"] = int(ram[self.powerup_addr]) if self.powerup_addr is not None else None
        vals["playstate"] = int(ram[self.playstate_addr]) if self.playstate_addr is not None else None
        return vals

    def update(self, frame, vals):
        for key, val in vals.items():
            if val is not None:
                self.series[key].append(val)

        def changed(key):
            old = self.prev.get(key)
            return old is not None and vals.get(key) is not None and old != vals[key]

        if changed("lives"):
            direction = "LOST A LIFE" if vals["lives"] < self.prev["lives"] else "GAINED A LIFE (1-Up)"
            self.event(frame, f"LIVES {self.prev['lives']} -> {vals['lives']}   <-- {direction}")
            if vals["lives"] < self.prev["lives"] and self.first_life_loss is None:
                self.first_life_loss = frame
        if changed("powerup"):
            old, new = self.prev["powerup"], vals["powerup"]
            names = {0: "small", 1: "big", 2: "fire", 3: "raccoon"}
            direction = "POWERED UP" if new > old else "TOOK A HIT / shrank"
            self.event(frame, f"POWERUP {old} ({names.get(old, '?')}) -> {new} ({names.get(new, '?')})   <-- {direction}")
        if changed("playstate"):
            old, new = self.prev["playstate"], vals["playstate"]
            note = ""
            if self.playstate_value is not None:
                if old == self.playstate_value:
                    note = "   <-- LEFT NORMAL PLAY (training ends the episode here)"
                    if self.first_playstate_exit is None:
                        self.first_playstate_exit = frame
                elif new == self.playstate_value:
                    note = "   <-- entered normal play"
            self.event(frame, f"PLAYSTATE {old} -> {new}{note}")
        if changed("score"):
            self.event(frame, f"SCORE {self.prev['score']} -> {vals['score']}   (+{vals['score'] - self.prev['score']})")

        self.prev = dict(vals)

    def event(self, frame, text):
        line = f">>> frame {frame:6d}  [{stamp(frame)}]  {text}"
        self.events.append(line)
        print(line)


def header():
    return (f"  {'FRAME':>6}  {'VIDEO':>10} | {'lives':>5} {'score':>7} {'time':>5} | "
            f"{'hpos':>5} {'scroll':>6} {'PROGRESS':>8} | {'power':>5} {'state':>5} {'play?':>5}")


def row(frame, v, playstate_value):
    def f(x, w):
        return f"{x if x is not None else '-':>{w}}"
    in_play = "-"
    if v["playstate"] is not None and playstate_value is not None:
        in_play = "YES" if v["playstate"] == playstate_value else "no"
    return (f"  {frame:6d}  {stamp(frame):>10} | {f(v['lives'],5)} {f(v['score'],7)} {f(v['time'],5)} | "
            f"{f(v['hpos'],5)} {f(v['scroll'],6)} {f(v['progress'],8)} | "
            f"{f(v['powerup'],5)} {f(v['playstate'],5)} {in_play:>5}")


def replay_demo(bk2_path, tracker, every):
    movie = retro.Movie(bk2_path)
    movie.step()
    env = retro.make(
        game=movie.get_game(), state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players, render_mode="rgb_array",
    )
    env.initial_state = movie.get_state()
    env.reset()
    print(f"Replaying {bk2_path} frame by frame (1 step = 1 emulator frame =")
    print(f"1 video frame, so printed timestamps line up with the MP4).\n")
    print(header())
    frame = 0
    while movie.step():
        keys = [movie.get_key(i, 0) for i in range(env.num_buttons)]
        _obs, _rew, terminated, truncated, info = env.step(keys)
        vals = tracker.read(env.get_ram(), info)
        if frame % every == 0:
            print(row(frame, vals, tracker.playstate_value))
        tracker.update(frame, vals)
        frame += 1
        if terminated or truncated:
            break
    env.close()
    return frame


def play_model(game, state, model_path, tracker, every, max_frames, record_dir):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    from train import ACTION_TABLE, VariableHoldDiscretizer, WarpFrame

    base = retro.make(game=game, state=state or retro.State.DEFAULT,
                      record=record_dir, render_mode="rgb_array")
    wrapped = WarpFrame(VariableHoldDiscretizer(base, ACTION_TABLE))
    venv = VecFrameStack(DummyVecEnv([lambda: wrapped]), n_stack=4)
    model = PPO.load(model_path)

    print(f"Playing {model_path}. One decision spans several emulator frames, so")
    print(f"the frame counter accumulates frames_this_step to stay aligned with")
    print(f"the video.\n")
    print(header())
    obs = venv.reset()
    frame = 0
    last_print = -every
    while frame < max_frames:
        action, _ = model.predict(obs, deterministic=False)
        obs, _r, done, infos = venv.step(action)
        info = infos[0]
        vals = tracker.read(base.get_ram(), info)
        if frame - last_print >= every:
            print(row(frame, vals, tracker.playstate_value))
            last_print = frame
        tracker.update(frame, vals)
        frame += info.get("frames_this_step", 1)
        if done[0]:
            break
    venv.close()
    return frame


def verdicts(tracker):
    print("\n" + "=" * 72)
    print("PER-ADDRESS VERDICTS")
    print("=" * 72)
    s = tracker.series

    def span(key):
        vals = s[key]
        return (min(vals), max(vals), len(set(vals))) if vals else (None, None, 0)

    lo, hi, distinct = span("lives")
    print(f"\nlives (info key)            range {lo}..{hi}, {distinct} distinct")
    print("  OK -- decrements on death." if tracker.first_life_loss is not None
          else "  NOT EXERCISED -- no death in this recording; can't confirm.")

    lo, hi, distinct = span("score")
    print(f"\nscore (info key)            range {lo}..{hi}, {distinct} distinct")
    print("  OK -- increases during play." if hi and hi > lo
          else "  NOT EXERCISED -- score never changed.")

    lo, hi, distinct = span("time")
    print(f"\ntime (info key)             range {lo}..{hi}, {distinct} distinct")
    print("  OK -- counts down in level." if distinct > 2
          else "  SUSPECT -- barely changed; expected a countdown during play.")

    lo, hi, distinct = span("hpos")
    print(f"\nhpos (info key)             range {lo}..{hi}, {distinct} distinct")
    print("  NOTE: this is Mario's ON-SCREEN x and is EXPECTED to cap (~144) and")
    print("  flatline while the level scrolls. It is NOT level position -- it is")
    print("  only added to the camera scroll to cover the pre-scroll walk.")

    lo, hi, distinct = span("scroll")
    print(f"\nprogress address (scroll)   range {lo}..{hi}, {distinct} distinct")
    if distinct > 5:
        print("  OK -- moves through many values during play (camera scrolling).")
    else:
        print("  SUSPECT -- barely moved. If the run had real forward travel this")
        print("  address is wrong; re-check with inspect_progress.py --watch.")

    lo, hi, distinct = span("progress")
    print(f"\nPROGRESS (scroll + hpos)    range {lo}..{hi}, {distinct} distinct")
    print("  This is the exact quantity the progress reward differences.")

    vals = sorted(set(s["powerup"])) if s["powerup"] else []
    print(f"\npowerup address             values seen: {vals}")
    if vals and max(vals) <= 7:
        print("  Plausible tier enum (0=small 1=big 2=fire 3=raccoon).")
        print("  Cross-check the POWERUP events above against the video.")
    elif vals:
        print("  SUSPECT -- values outside a small tier enum; likely wrong address.")

    vals = sorted(set(s["playstate"])) if s["playstate"] else []
    print(f"\nplaystate address           values seen: {vals}")
    if tracker.playstate_value is not None:
        frac = s["playstate"].count(tracker.playstate_value) / max(1, len(s["playstate"]))
        print(f"  in-play value {tracker.playstate_value} held {frac*100:.1f}% of frames")
        if frac < 0.5:
            print("  SUSPECT -- normal play should dominate a gameplay recording.")
        if tracker.first_playstate_exit is not None and tracker.first_life_loss is not None:
            lead = tracker.first_life_loss - tracker.first_playstate_exit
            print(f"  left play {lead} frames BEFORE lives decremented "
                  f"({lead / FPS:.2f}s of death sequence)")
            if lead >= 40:
                print("  OK -- flips at the hit, so the whole death sequence is excluded")
                print("  from progress. This is what the training gate depends on.")
            else:
                print("  SUSPECT -- flips too late; death-sequence frames would still be")
                print("  paid as progress. Re-run probe_after_death.py for an earlier byte.")

    print("\n" + "=" * 72)
    print(f"{len(tracker.events)} events printed above, each with a video timestamp.")
    print("Scrub the MP4 to a few and confirm they match what you see on screen.")
    print("=" * 72)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", help="A .bk2 recording to replay (best: real power-ups/hits/deaths)")
    src.add_argument("--model", help="A trained checkpoint to play and record instead")
    p.add_argument("--game", default="SuperMarioBros3-Nes-v0", help="Only needed with --model (default: %(default)s)")
    p.add_argument("--state", default=None)
    p.add_argument("--every", type=int, default=30, help="Print a table row every N frames (default: %(default)s = 0.5s)")
    p.add_argument("--max-frames", type=int, default=20000, help="Cap for --model runs")
    p.add_argument("--record-dir", default="./recordings", help="Where --model recordings land")
    p.add_argument("--render-video", action="store_true", help="Also render the recording to MP4 so timestamps can be checked visually")
    # Defaults are this project's VERIFIED SuperMarioBros3-Nes-v0 addresses.
    p.add_argument("--progress-address", type=lambda v: int(v, 0), default=0x00CF)
    p.add_argument("--progress-address-high", type=lambda v: int(v, 0), default=None)
    p.add_argument("--powerup-address", type=lambda v: int(v, 0), default=0x00ED)
    p.add_argument("--playstate-address", type=lambda v: int(v, 0), default=0x07F1)
    p.add_argument("--playstate-value", type=lambda v: int(v, 0), default=5)
    args = p.parse_args()

    print("Auditing these sources (defaults are the project's verified SMB3 values):")
    print(f"  progress address  : 0x{args.progress_address:04X} ({args.progress_address})"
          + (f" + high 0x{args.progress_address_high:04X}" if args.progress_address_high is not None else "")
          + "  [camera scroll]")
    print(f"  powerup address   : 0x{args.powerup_address:04X} ({args.powerup_address})  [0=small 1=big 2=fire 3=raccoon]")
    print(f"  playstate address : 0x{args.playstate_address:04X} ({args.playstate_address}), in-play value {args.playstate_value}")
    print(f"  info keys         : lives, score, time, hpos")
    print(f"  video frame rate  : {FPS} fps\n")

    tracker = Tracker(args.progress_address, args.powerup_address,
                      args.playstate_address, args.playstate_value,
                      progress_addr_high=args.progress_address_high)

    if args.demo:
        if not os.path.exists(args.demo):
            sys.exit(f"No such recording: {args.demo}")
        replay_demo(args.demo, tracker, args.every)
        bk2 = args.demo
    else:
        os.makedirs(args.record_dir, exist_ok=True)
        import glob
        import time as _time
        before = set(glob.glob(os.path.join(args.record_dir, "*.bk2")))
        started = _time.time()
        play_model(args.game, args.state, args.model, tracker, args.every,
                   args.max_frames, args.record_dir)
        from play_and_record import find_new_bk2
        bk2 = find_new_bk2(args.record_dir, before, started_at=started)

    verdicts(tracker)

    if args.render_video and bk2:
        from play_and_record import render_to_mp4
        mp4 = render_to_mp4(bk2)
        if mp4:
            print(f"\nVideo for cross-checking the timestamps above:\n  {mp4}")
    elif args.render_video:
        print("\nNo .bk2 available to render.")
    elif bk2:
        print(f"\nRecording: {bk2}   (add --render-video to encode it to MP4)")


if __name__ == "__main__":
    main()
