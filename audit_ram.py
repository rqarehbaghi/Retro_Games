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
    info keys : lives, score, time, hpos  -- the run prints the RAM address
                behind each, read from the integration's own data.json.
                NOTE `score` is published as one TENTH of the on-screen value.
    RAM       : --powerup-address   (SMB3: 0x00ED, CONFIRMED against video --
                                     tier flips exactly when a mushroom is taken)
                --progress-address  (SMB3: none yet. 0x00CF and 0x053C are both
                                     REJECTED; train.py refuses them.)
                --playstate-address (SMB3: none yet. 0x07F1 and 0x0749 are both
                                     REJECTED; train.py refuses them.)
    Columns for unset addresses are omitted rather than printed as dashes.

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
        # Per-kind event counts. A byte that changes constantly is not a state
        # variable, and printing every transition buries the events that matter
        # (power-ups, hits, deaths) in noise -- so each kind is capped and the
        # rest are counted for the verdict.
        self.event_counts = {}
        self.event_cap = 8
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
            self.event(frame, "LIVES", f"LIVES {self.prev['lives']} -> {vals['lives']}   <-- {direction}")
            if vals["lives"] < self.prev["lives"] and self.first_life_loss is None:
                self.first_life_loss = frame
        if changed("powerup"):
            old, new = self.prev["powerup"], vals["powerup"]
            names = {0: "small", 1: "big", 2: "fire", 3: "raccoon"}
            direction = "POWERED UP" if new > old else "TOOK A HIT / shrank"
            self.event(frame, "POWERUP", f"POWERUP {old} ({names.get(old, '?')}) -> {new} ({names.get(new, '?')})   <-- {direction}")
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
            self.event(frame, "PLAYSTATE", f"PLAYSTATE {old} -> {new}{note}")
        if changed("score"):
            delta = vals['score'] - self.prev['score']
            # The integration publishes score with its trailing zero dropped --
            # one tenth of the HUD value. Confirmed two ways: a Super Mushroom
            # (1000 on screen) appears here as +100, and a +100 on screen
            # appears as +10. Print both so video cross-checks don't look like
            # missing events.
            self.event(frame, "SCORE",
                       f"SCORE {self.prev['score']} -> {vals['score']}   "
                       f"(+{delta}, on screen +{delta * 10})")

        self.prev = dict(vals)

    def event(self, frame, kind, text):
        n = self.event_counts.get(kind, 0) + 1
        self.event_counts[kind] = n
        line = f">>> frame {frame:6d}  [{stamp(frame)}]  {text}"
        self.events.append(line)
        if n < self.event_cap:
            print(line)
        elif n == self.event_cap:
            print(f">>> ... further {kind} changes suppressed (a value changing "
                  f"this often is not a state variable -- see verdicts)")


def header(show_progress, show_playstate):
    """Only configured columns are shown -- unset addresses would print a wall
    of dashes and bury the values that matter."""
    h = f"  {'FRAME':>6}  {'VIDEO':>10} | {'lives':>5} {'score':>7} {'time':>5}  {'hpos':>5}"
    if show_progress:
        h += f" | {'scroll':>6} {'PROGRESS':>8}"
    h += f" | {'power':>5}"
    if show_playstate:
        h += f" {'state':>5} {'play?':>5}"
    return h


def row(frame, v, playstate_value, show_progress, show_playstate):
    def f(x, w):
        return f"{x if x is not None else '-':>{w}}"
    line = (f"  {frame:6d}  {stamp(frame):>10} | {f(v['lives'],5)} {f(v['score'],7)} "
            f"{f(v['time'],5)}  {f(v['hpos'],5)}")
    if show_progress:
        line += f" | {f(v['scroll'],6)} {f(v['progress'],8)}"
    line += f" | {f(v['powerup'],5)}"
    if show_playstate:
        in_play = "-"
        if v["playstate"] is not None and playstate_value is not None:
            in_play = "YES" if v["playstate"] == playstate_value else "no"
        line += f" {f(v['playstate'],5)} {in_play:>5}"
    return line


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
    print(header(tracker.progress_addr is not None, tracker.playstate_addr is not None))
    frame = 0
    while movie.step():
        keys = [movie.get_key(i, 0) for i in range(env.num_buttons)]
        _obs, _rew, terminated, truncated, info = env.step(keys)
        vals = tracker.read(env.get_ram(), info)
        if frame % every == 0:
            print(row(frame, vals, tracker.playstate_value,
                      tracker.progress_addr is not None, tracker.playstate_addr is not None))
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
    print(header(tracker.progress_addr is not None, tracker.playstate_addr is not None))
    obs = venv.reset()
    frame = 0
    last_print = -every
    while frame < max_frames:
        action, _ = model.predict(obs, deterministic=False)
        obs, _r, done, infos = venv.step(action)
        info = infos[0]
        vals = tracker.read(base.get_ram(), info)
        if frame - last_print >= every:
            print(row(frame, vals, tracker.playstate_value,
                      tracker.progress_addr is not None, tracker.playstate_addr is not None))
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
    zeros = s["scroll"].count(0) if s["scroll"] else 0
    zero_frac = zeros / max(1, len(s["scroll"]))
    drops = sum(1 for a, b in zip(s["scroll"], s["scroll"][1:]) if b < a - 8)
    if zero_frac > 0.15 and drops > 5:
        print(f"  *** REJECTED: sits at 0 for {zero_frac*100:.0f}% of frames with "
              f"{drops} large drops.")
        print("  An accumulated level position never repeatedly resets to zero --")
        print("  this reads like a per-frame delta / velocity, not a position. It")
        print("  looked monotonic earlier only because that check held RIGHT the")
        print("  whole time; real play (stopping, backing up) exposes it.")
    elif distinct > 5:
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
    ps_changes = tracker.event_counts.get("PLAYSTATE", 0)
    nframes = max(1, len(s["playstate"]))
    if ps_changes > 10 and ps_changes / nframes > 0.005:
        print(f"  *** REJECTED: changed {ps_changes} times in {nframes} frames -- this")
        print("  byte CYCLES during normal gameplay (animation / PPU phase), so it")
        print("  cannot mark 'in play'. Gating training on it terminates every")
        print("  episode within about a second. Do NOT pass --playstate-address.")
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
    p.add_argument("--progress-address", type=lambda v: int(v, 0), default=None, help="Camera/position address to audit. There is NO verified value for SMB3: 0x00CF was REJECTED (it returns to 0 whenever the player stops or reverses -- a per-frame delta, not accumulated position). Find a replacement with inspect_progress.py --demo.")
    p.add_argument("--progress-address-high", type=lambda v: int(v, 0), default=None)
    p.add_argument("--powerup-address", type=lambda v: int(v, 0), default=0x00ED)
    p.add_argument("--playstate-address", type=lambda v: int(v, 0), default=None, help="Game-mode address to audit. There is NO verified value for SMB3: 0x07F1 was REJECTED (it cycles 5->0->8 every ~24 frames DURING normal play, so it cannot mark 'in play' and would end every training episode within a second).")
    p.add_argument("--playstate-value", type=lambda v: int(v, 0), default=None, help="The value --playstate-address holds during normal play.")
    args = p.parse_args()

    # Same source table training prints, so the audit and the run can never
    # disagree about where a value comes from.
    game = args.game
    if args.demo:
        try:
            game = retro.Movie(args.demo).get_game()
        except Exception:
            pass
    from train import describe_value_sources, reject_known_bad
    describe_value_sources(
        game,
        progress_address=args.progress_address,
        progress_address_high=args.progress_address_high,
        powerup_address=args.powerup_address,
        playstate_address=args.playstate_address,
        playstate_value=args.playstate_value,
    )
    # Auditing a known-bad address is legitimate (that's how they get
    # disproven), so warn rather than exit -- unlike training, which refuses.
    for kind, addr in (("progress", args.progress_address),
                       ("playstate", args.playstate_address)):
        if (game, kind, addr) in __import__("train").KNOWN_BAD_ADDRESSES:
            print(f"  !! auditing a KNOWN-BAD {kind} address 0x{addr:04X}; "
                  f"train.py refuses this one.\n")
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
