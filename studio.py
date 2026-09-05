#!/usr/bin/env python3
"""
One command: play, then get finished videos staged for review and upload.

    python studio.py --game SuperMarioBros3-Nes-v0

That is the whole command. Play until you close the window; everything after
is decided from the run itself. The title is written from what happened
("cleared without dying", "died 3 times", "14 coin run"), the handle and
house style come from studio.json, and cuts are joined with a fade through
black -- chosen because cross-dissolving pixel art into pixel art muddies
both frames and pixelize on already-pixelated content reads as an encoding
fault. Every one of those is still overridable, but none has to be decided.

It produces, in a dated folder:

    <slug>_16x9.mp4    1920x1080 landscape master (YouTube)
    <slug>_9x16.mp4    1080x1920 short, cut to highlights (Shorts / Reels / TikTok)
    metadata.json      title, description and per-platform hashtags
    narration.txt      a timestamped script of what actually happened
    events.csv         the raw event timeline the script was built from

THE SHORT IS CUT, NOT TRUNCATED. --short-seconds (default 15) is a budget, not
a stop point: the run is cut down to the moments that earned their place and
those are joined in order, separated by a fade
through black so a jump between unrelated moments reads as an edit rather
than a glitch (--transition none/fade/dissolve/pixelize). Selection comes from the RAM event log -- the exact
frame a power-up was taken or a life was lost -- so it is ground truth rather
than inference. That is also why no AI highlight detector is involved: those
score interest from pixels, audio energy or a transcript, and gameplay with no
commentary gives them almost nothing, while this already knows exactly what
happened and when. The 16x9 master is always the full run.

WHY VERTICAL USES A BLURRED FILL: an NES frame is 256x224, about 8:7. Fitted
into 1080x1920 it scales to 1080x945 and fills the full width, so it covers
about half the height and the rest is dead space. Padding that with black
reads as a lazy upload; filling it with a blurred, zoomed copy of the same
frame reads as intentional. The sharp gameplay is laid on top at
nearest-neighbour, so the pixels stay crisp either way. This is an aesthetic
choice, not a size fix -- the gameplay is the same size in both.

NOT UPLOADED FOR YOU, deliberately. Of the three targets only TikTok supports
"post privately now, make it public after review": unaudited API clients post
at SELF_ONLY and the owner can flip each post to Everyone later. YouTube
uploads from an UNVERIFIED API project are LOCKED private and per Google's own
help pages that cannot be appealed -- you would have to re-upload by hand
anyway, so automating it destroys the footage. Instagram has no draft or
private state in its publishing API at all: media_publish goes live at once.
So this stages files; see --print-upload-plan for what to do with them.
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

import writer

from overlays import (DEFAULT_FONT, DEFAULT_STYLE, TRANSITIONS,
                      merge_style, render_spec, save_spec)





FPS = 60.0988


def stamp(frame):
    secs = frame / FPS
    return f"{int(secs // 60):01d}:{secs % 60:05.2f}"


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "session"






STUDIO_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "studio.json")

# Console suffixes stable-retro appends to every game id. Stripped for display
# so a title reads "Super Mario Bros 3" and not "SuperMarioBros3-Nes-v0".
CONSOLE_SUFFIXES = ("Nes", "Snes", "Genesis", "GameBoy", "GbColor", "GbAdvance",
                    "Atari2600", "Sms", "GameGear", "PCEngine", "N64", "Saturn",
                    "Sega32X", "SegaCD", "Master System")


def load_studio_config(path=STUDIO_CONFIG):
    """Per-channel defaults, so the handle and house style are set once here
    rather than retyped on every run."""
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception:
        return {}


def pretty_game(game_id):
    """SuperMarioBros3-Nes-v0 -> Super Mario Bros 3."""
    name = re.sub(r"-v\d+$", "", game_id)
    for suffix in CONSOLE_SUFFIXES:
        name = re.sub(r"-" + re.escape(suffix) + r"$", "", name, flags=re.I)
    name = name.replace("-", " ")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)      # SuperMario -> Super Mario
    name = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", name)      # Bros3 -> Bros 3
    return re.sub(r"\s+", " ", name).strip()


def auto_title(game, level=None):
    """The label to burn across the top: what game, and where in it.

    This used to append a statistic -- "14 coin run", "died 3 times" -- which
    is a scoreboard, not a title. Nobody scrolling needs the tally, and it
    dates the clip to one run. Game and level is what a viewer actually wants
    to know."""
    name = pretty_game(game)
    return "%s - %s" % (name, level) if level else name


# Lines timed to what actually happened, because the event log knows exactly
# when it happened. A handle burnt in the corner is branding, not commentary --
# these are the commentary. Pools rather than single strings so a run with four
# deaths does not print the same sentence four times.
#
# Voice is dry and self-deprecating: this is footage of the author playing,
# and a caption that boasts about a mushroom reads badly. {n} is filled with
# the running count of that event where a pool uses it.
CAPTION_LINES = {
    # The voice is a commentator who has watched a great deal of this and is
    # not impressed easily -- but is FAIR. Mistakes get taken apart; genuinely
    # good play gets credit, delivered as a backhanded compliment rather than
    # withheld. A caption that only ever sneers stops being funny by the third
    # one, because nothing is at stake in the praise.
    #
    # Lines can run long now: captions wrap onto up to three lines and stay on
    # screen in proportion to their length, so there is room for an actual joke
    # instead of a fragment.
    #
    # A STARTING TABLE, not the final wording. Every run writes the chosen
    # lines into overlays.json beside the video and restyle.py re-renders after
    # you edit it, so the funniest version of any of these is the one written
    # after actually watching the clip.
    "open": ["Right. A grown adult, a 1988 cartridge, and no excuses.",
             "He has played this before. Allegedly.",
             "One take. Everything you are about to see was avoidable."],
    "death": ["Killed by the first enemy in the game. A round one. With a face.",
              "He saw it coming and walked at it anyway. Bold.",
              "Death {n}. At this point the goombas are just spectating.",
              "That enemy has stood there since 1988 waiting for exactly this.",
              "Avoidable. Every single frame of that was avoidable."],
    "shrink": ["And it is gone. Nine glorious seconds of being large.",
               "Back to small. Back to being a target.",
               "The mushroom gave its life for absolutely nothing."],
    "powerdown": ["There goes the tail. He had it for one screen.",
                  "Downgraded, and honestly, earned."],
    "powerup": ["A mushroom. Look at him. Practically a professional.",
                "Genuine competence. I am noting it for the record.",
                "Upgraded. Enjoy it, it will not survive the next screen.",
                "Now he is big, and the confidence is already a problem."],
    "1up": ["An extra life, for a man who clearly needs the inventory.",
            "A 1-Up. Credit where it is due. He will waste it in a minute."],
    "coin": ["{n} coins. Financially secure, mechanically hopeless.",
             "{n} coins collected, none of which make him better at this.",
             "{n} coins. Genuinely tidy collecting, for whatever that is worth."],
    "clear": ["He cleared it. I am as surprised as you are, and I am paid to watch.",
              "Level complete, and the last stretch was actually good. There. Said it.",
              "Finished. Not pretty, but finished, and that counts."],
    "pipe": ["Into a pipe. Hiding from his responsibilities.",
             "Down a pipe, where there are fewer witnesses."],
}

CAPTION_HOLD = 2.6        # seconds each line stays up
CAPTION_GAP = 1.5         # clear seconds between one line leaving and the next
COIN_EVERY = 10           # only caption coins at milestones


def caption_script(events, duration, game, seed=None):
    """(start_seconds, text) pairs for the whole run.

    Deliberately sparse. A caption on every coin would be a wall of text over
    the gameplay, so coins only speak at milestones and nothing lands within
    CAPTION_GAP of the previous line."""
    rng = random.Random(seed if seed is not None else game)
    lines = []

    used = {}

    def pool(kind, count=None):
        """A line not yet used for this kind, until the pool runs dry.

        Plain rng.choice picks independently every time, so a long run printed
        "peak. it is all downhill from here." three times -- which is exactly
        what having pools was supposed to prevent."""
        options = CAPTION_LINES.get(kind)
        if not options:
            return None
        seen = used.setdefault(kind, set())
        fresh = [o for o in options if o not in seen]
        if not fresh:
            seen.clear()
            fresh = list(options)
        text = rng.choice(fresh)
        seen.add(text)
        return text.format(n=count, game=pretty_game(game))

    opener = pool("open")
    if opener:
        lines.append((0.6, opener))

    counts = {}
    for frame, kind, _detail in events:
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "coin" and counts[kind] % COIN_EVERY:
            continue
        if kind == "score":
            continue
        text = pool(kind, counts[kind])
        if not text:
            continue
        # Events are logged when the value changes, which for a death is after
        # the animation -- so the line lands as the punchline rather than
        # spoiling it. No extra offset needed.
        at = frame / FPS
        if at + CAPTION_HOLD > duration:
            continue
        # Space against when the previous line LEAVES the screen, not when it
        # arrived: holds scale with length now and can exceed a fixed gap, so a
        # constant would let a long caption still be up when the next appears.
        if lines:
            prev_at, prev_text = lines[-1]
            prev_hold = max(CAPTION_HOLD, len(prev_text) / 13.0)
            if at < prev_at + prev_hold + CAPTION_GAP:
                continue
        lines.append((at, text))
    return lines











# Weighted so the cut lands on what is worth watching. A clear or a 1-Up is the
# payoff, a power-up is the setup, a hit is at least dramatic; coins are common
# enough that they should only break ties.
EVENT_WEIGHTS = {"clear": 5.0, "1up": 5.0, "death": 4.5, "powerup": 4.0,
                 "powerdown": 2.0, "shrink": 2.5, "pipe": 2.0, "coin": 1.0,
                 "score": 0.25}

# How far BEFORE the logged frame each kind of moment actually starts.
#
# Events are logged when the underlying value changes, and some values change
# long after the thing you want to watch. A death is the worst case: `lives`
# only decrements at the END of the death sequence, after the death-jump
# animation and the screen fade -- roughly three seconds after the hit that
# caused it. A 1.5s lead there starts the clip on the aftermath and misses
# the kill entirely, which is exactly what it looked like. Coins are the other
# extreme: the counter moves on the frame you touch them.
EVENT_LEAD = {"death": 4.0, "clear": 4.0, "powerdown": 2.5, "shrink": 2.5, "pipe": 1.5, "1up": 2.0,
              "powerup": 1.5, "coin": 1.0, "score": 1.0}


def best_window(events, total_s, want_s):
    """Fallback for when there is nothing to cut on: the densest contiguous
    stretch of the timeline, or the opening if the timeline is empty."""
    if not events:
        return 0.0
    best_start, best_score = 0.0, -1.0
    for frame, _kind, _detail in events:
        start = max(0.0, min(frame / FPS - want_s / 2.0, total_s - want_s))
        score = sum(EVENT_WEIGHTS.get(k, 0.5) for f, k, _d in events
                    if start <= f / FPS <= start + want_s)
        if score > best_score:
            best_start, best_score = start, score
    return best_start


def highlight_segments(events, total_s, budget_s, seg_s=4.0, lead=None):
    """The moments worth keeping, as chronological (start, duration) pairs.

    NOT the first N seconds -- a level opens with walking right, so a naive
    truncation keeps the least watchable part. NOT an AI highlight detector
    either: those infer interest from pixels, audio energy or a transcript,
    and gameplay with no commentary gives them almost nothing to work with.
    This cuts on the RAM event log instead, which is ground truth -- the exact
    frame a mushroom was taken or a life was lost -- so the selection is
    exact rather than guessed.

    Segments are taken highest-weight first until the budget is spent, merged
    where they overlap, then put back in chronological order so the cut still
    reads as a run rather than a shuffle."""
    if total_s <= budget_s:
        return []                       # already short enough, keep it whole
    if not events:
        return [(best_window(events, total_s, budget_s), budget_s)]

    chosen, used = [], 0.0
    for frame, kind, _detail in sorted(
            events, key=lambda e: -EVENT_WEIGHTS.get(e[1], 0.5)):
        # A lead given on the command line applies to everything; otherwise
        # each kind gets the run-up it actually needs.
        ahead = lead if lead is not None else EVENT_LEAD.get(kind, 1.5)
        start = max(0.0, min(frame / FPS - ahead, total_s - seg_s))
        end = min(total_s, start + seg_s)
        for i, (existing_start, existing_end) in enumerate(chosen):
            if start <= existing_end and end >= existing_start:
                wider = (min(existing_start, start), max(existing_end, end))
                grow = (wider[1] - wider[0]) - (existing_end - existing_start)
                if used + grow <= budget_s:
                    chosen[i] = wider
                    used += grow
                break
        else:
            if used + (end - start) <= budget_s:
                chosen.append((start, end))
                used += end - start
    if not chosen:
        return [(best_window(events, total_s, budget_s), budget_s)]
    chosen.sort()
    return [(start, end - start) for start, end in chosen]






TRANSITION_DROP = 600
POWER_TIERS = {0: "small", 1: "big", 2: "fire", 3: "raccoon"}

# The power byte is cleared by the game on death AND on level exit, not just
# when something hits you -- games.json records this against 0x00ED, and
# RewardShaper already skips those cases. read_events did not, so finishing a
# level printed "back to small mario" as though it were a hit. A death does the
# same thing, roughly three seconds before the lives counter catches up.
# A level ending in SMB3 is the goal card, then a score tally, then the map --
# comfortably more than five seconds, so the first version of this window (300
# frames) expired before the clear was logged and the caption survived anyway.
POWER_NOISE_WINDOW = 900          # 15s, long enough to span the whole outro
END_OF_RUN_WINDOW = 900           # a drop this close to the end of the tape


def filter_power_noise(events, total_frames=None):
    """Drop power changes that are really a death or a level ending.

    The power byte at 0x00ED is cleared by the game on BOTH of those, not only
    when something hits you -- games.json records this, and RewardShaper
    already skips it. Two things make it awkward here.

    First, the giveaway arrives LATER than the drop: on a fatal hit the byte
    clears at once while `lives` only decrements at the end of the death
    animation, so nothing at the moment of the drop can say which it was. Hence
    a second pass rather than an inline test.

    Second, a run that ENDS at the level end never logs a clear at all -- the
    recording stops before the position collapses -- so anchoring only on a
    following event misses exactly the case that kept being reported. A drop
    within END_OF_RUN_WINDOW of the last frame is therefore treated as the
    outro as well: nothing happening with fifteen seconds left and no play
    afterwards is a hit worth captioning."""
    marks = [f for f, kind, _d in events if kind in ("death", "clear")]
    kept = []
    for frame, kind, detail in events:
        if kind in ("shrink", "powerdown"):
            if any(0 <= mark - frame <= POWER_NOISE_WINDOW or
                   0 <= frame - mark <= 120 for mark in marks):
                continue
            if total_frames is not None and total_frames - frame <= END_OF_RUN_WINDOW:
                continue
        kept.append((frame, kind, detail))
    return kept


# A pipe and a level ending look identical in position: both collapse it by
# more than TRANSITION_DROP with no life lost. The LEVEL TIMER separates them.
# It only ever counts down during play, and is reset UP when a new level
# starts, so an increase shortly after a collapse means a new level began --
# i.e. the previous one really ended. Going down a pipe keeps the same timer
# running down, because it is the same level.
CLEAR_CONFIRM_WINDOW = 600        # frames to wait for the timer to reset
END_OF_TAPE_WINDOW = 600          # a collapse this close to the end


def classify_collapses(collapses, timeline, total_frames):
    """Turn position collapses into 'clear' or 'pipe' events.

    Without this every pipe in the level was captioned as a level completion,
    because nothing at the moment of the collapse distinguishes them."""
    out = []
    for frame in collapses:
        window = [t for t in timeline[frame:frame + CLEAR_CONFIRM_WINDOW] if t is not None]
        before = next((t for t in reversed(timeline[:frame]) if t is not None), None)
        reset = before is not None and any(t > before + 5 for t in window)
        ended = total_frames - frame <= END_OF_TAPE_WINDOW
        if reset or (ended and not window):
            out.append((frame, "clear", "level ended"))
        else:
            out.append((frame, "pipe", "went down a pipe"))
    return out


def read_events(bk2_path, game):
    """Replay the recording and note what happened, so narration and titles
    describe the real run instead of being generic. Degrades to an empty
    timeline for any game games.json has no variables for."""
    try:
        import stable_retro as retro

        from train import load_game_config
    except Exception as exc:
        print(f"  (event scan unavailable: {exc})")
        return []

    defaults, _rewards = load_game_config(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "games.json"), game)
    coin_addr = defaults.get("coin_address")
    power_addr = defaults.get("powerup_address")

    movie = retro.Movie(bk2_path)
    movie.step()
    env = retro.make(game=movie.get_game(), state=None,
                     use_restricted_actions=retro.Actions.ALL,
                     players=movie.players, render_mode="rgb_array")
    env.initial_state = movie.get_state()
    env.reset()

    prog_key = None
    prog_hi = None
    prog = (defaults.get("progress_use_info_x"), defaults.get("progress_address_high"))
    if prog[0]:
        prog_key = "hpos"
        prog_hi = prog[1]

    timer_addr = defaults.get("timer_address")

    events, frame = [], 0
    prev = {}
    last_pos = None
    collapses = []
    timeline = []
    while movie.step():
        keys = [movie.get_key(i, p) for p in range(movie.players)
                for i in range(env.num_buttons)]
        _obs, _r, term, trunc, info = env.step(keys)
        ram = env.get_ram()
        position = None
        if prog_key is not None:
            low = info.get(prog_key)
            if low is not None:
                position = int(low) + (int(ram[prog_hi]) << 8 if prog_hi is not None else 0)
        if position is not None and last_pos is not None:
            # A big backward jump with no life lost is the level ending, not
            # movement -- the page byte absorbs hpos wraps, so ordinary travel
            # never produces one. Same test RewardShaper uses.
            if position - last_pos < -TRANSITION_DROP:
                # A CANDIDATE only. Going down a pipe collapses position in
                # exactly the same way a level ending does, which is why every
                # pipe was captioned "level complete". Confirmed below against
                # the timer, which is the one thing that tells them apart.
                collapses.append(frame)
        if position is not None:
            last_pos = position
        if timer_addr is not None:
            try:
                timeline.append(int(ram[timer_addr]) * 100
                                + int(ram[timer_addr + 1]) * 10
                                + int(ram[timer_addr + 2]))
            except Exception:
                timeline.append(None)
        else:
            timeline.append(None)

        current = {
            "score": info.get("score"),
            "lives": info.get("lives"),
            "coins": int(ram[coin_addr]) if coin_addr is not None else None,
            "power": int(ram[power_addr]) if power_addr is not None else None,
        }
        for key, now in current.items():
            was = prev.get(key)
            if was is None or now is None or now == was:
                continue
            if key == "lives" and now < was:
                events.append((frame, "death", f"lost a life ({was} -> {now})"))
            elif key == "lives" and now > was:
                events.append((frame, "1up", f"extra life ({was} -> {now})"))
            elif key == "coins" and 0 < now - was <= 5:
                events.append((frame, "coin", f"coin ({now})"))
            elif key == "power" and now > was:
                events.append((frame, "powerup",
                               f"{POWER_TIERS.get(was, was)} -> {POWER_TIERS.get(now, now)}"))
            elif key == "power" and now < was:
                # Dropping to 0 and dropping a tier are different events: losing
                # the tail still leaves you big. Calling both "back to small"
                # was wrong on every raccoon -> big hit.
                kind = "shrink" if now == 0 else "powerdown"
                events.append((frame, kind,
                               f"{POWER_TIERS.get(was, was)} -> {POWER_TIERS.get(now, now)}"))
            elif key == "score" and now > was:
                # info score is one tenth of the HUD value for this game
                events.append((frame, "score", f"+{(now - was) * 10} points"))
        prev = current
        frame += 1
        if term or trunc:
            break
    env.close()
    events.extend(classify_collapses(collapses, timeline, frame))
    events.sort(key=lambda e: e[0])
    return filter_power_noise(events, total_frames=frame)


# Talking over the whole run, not just its highlights. A commentary track that
# only speaks when something happens leaves most of a 90-second video silent,
# so the gaps get filled with observation -- which is what a real commentator
# does while nothing much is going on.
NARRATION_OPEN = [
    "Here we go. {game}, {level}. No save states, no rewinds, no second takes.",
    "{game}, {level}. A cartridge older than most of the audience, and a man who "
    "insists he remembers how this goes.",
    "Welcome back. {game}, {level}, played straight through. Whatever happens "
    "next, it happened live.",
]

NARRATION_FILL = [
    "Look at this. Not one pixel of it has changed since 1988, and neither has "
    "the correct way to play it.",
    "There is a rhythm to this level that people spent childhoods learning. We "
    "will see whether any of it survived.",
    "Everything on this screen is trying to kill him, politely, on a fixed "
    "schedule, exactly as it did decades ago.",
    "The music has not stopped. The music never stops. That is half of why "
    "anyone remembers this game at all.",
    "Nothing here is random. Every enemy, every block, every gap is in the same "
    "place it has always been.",
    "This is the part where muscle memory is supposed to take over. Any moment "
    "now.",
    "Somewhere out there is a person who can do this level in forty seconds "
    "without breathing hard. This is not that person.",
    "Steady progress. I want to acknowledge it now, because I suspect it will "
    "not last.",
]

NARRATION_EVENT = {
    "death": [
        "And there it is. Killed by the oldest, slowest, most telegraphed enemy "
        "in the entire game.",
        "He walked into that. Not tripped, not got cornered. Walked.",
        "That was avoidable, and I want that noted. Every frame of it was "
        "avoidable.",
        "Death number {n}. The enemies are not even trying at this point.",
    ],
    "shrink": [
        "And the mushroom is gone. Nine seconds of being large, wasted.",
        "Back to small. Back to being exactly one mistake from the start of the "
        "level.",
    ],
    "powerdown": [
        "There goes the tail, after roughly one screen of ownership.",
        "Downgraded. Somewhere a raccoon is filing a complaint.",
    ],
    "powerup": [
        "A mushroom, and credit where it is due, that was a clean grab.",
        "Powered up. He looks dangerous now. He is not, but he looks it.",
        "Good. Genuinely good. I am contractually obliged to say something "
        "unkind later, so enjoy this.",
    ],
    "1up": [
        "An extra life. Earned, and almost certainly about to be spent.",
        "A one up. That is the good news. The bad news is the next thirty "
        "seconds.",
    ],
    "coin": [
        "{n} coins now. Financially secure. Mechanically, we will see.",
        "Coins are stacking up. That is real, that counts, that is skill.",
    ],
    "clear": [
        "And that is the level. Finished, on camera, no retries. Say what you "
        "like about the middle section, the ending was clean.",
        "Cleared. I am going to be honest with you, I did not expect that, and I "
        "am paid to expect things.",
    ],
    "pipe": [
        "Into a pipe, away from his problems.",
        "Down we go. Fewer witnesses down here.",
    ],
}

NARRATION_CLOSE = [
    "That is the run. {tally} Somewhere a version of this that went smoothly "
    "exists, and this was not it.",
    "And that is where we leave it. {tally} Same time next level.",
    "Run over. {tally} You may now go and remember this game as harder than it "
    "was.",
]

WORDS_PER_MINUTE = 150      # a plain TTS read
NARRATION_GAP = 6.0         # fill any silence longer than this


def _tally(events):
    counts = {}
    for _frame, kind, _detail in events:
        counts[kind] = counts.get(kind, 0) + 1
    bits = []
    for kind, word in (("coin", "coin"), ("death", "death"),
                       ("powerup", "power-up"), ("1up", "extra life")):
        n = counts.get(kind)
        if n:
            bits.append("%d %s%s" % (n, word, "" if n == 1 else "s"))
    return ("Final count: " + ", ".join(bits) + ".") if bits else ""


def narration(events, duration_s, game, players, level=None, seed=None):
    """A full commentary script, timed across the whole run.

    Previously this listed the handful of moments the captions already covered,
    which left roughly eighty seconds of a ninety-second video with nothing
    being said over it. A commentary track has to keep talking, so event lines
    are the spine and observation fills the silences between them."""
    rng = random.Random(seed if seed is not None else "%s-narration" % game)
    name = pretty_game(game)
    where = level or "from the top"
    used = {}

    def pick(pool, key, **fmt):
        seen = used.setdefault(key, set())
        fresh = [line for line in pool if line not in seen] or list(pool)
        if len(fresh) == len(pool):
            seen.clear()
        line = rng.choice(fresh)
        seen.add(line)
        return line.format(game=name, level=where, **fmt)

    script = [(0.4, pick(NARRATION_OPEN, "open"))]
    counts = {}
    for frame, kind, _detail in events:
        counts[kind] = counts.get(kind, 0) + 1
        pool = NARRATION_EVENT.get(kind)
        if not pool:
            continue
        if kind == "coin" and counts[kind] % 10:
            continue
        at = frame / FPS
        if at > duration_s - 3:
            continue
        script.append((at, pick(pool, kind, n=counts[kind])))

    # Fill the silences. Reading rate sets how long a line occupies, so a gap
    # is only filled when there is genuinely room for another one.
    script.sort()

    def spoken(text):
        return len(text.split()) / WORDS_PER_MINUTE * 60.0

    filled = []
    for i, (at, text) in enumerate(script):
        filled.append((at, text))
        cursor = at + spoken(text)
        # Against the SORTED neighbour: indexing the unsorted list here left a
        # fifteen-second silence in the middle of a run it thought it had filled.
        next_at = script[i + 1][0] if i + 1 < len(script) else duration_s
        while next_at - cursor > NARRATION_GAP:
            line = pick(NARRATION_FILL, "fill")
            filled.append((cursor + 1.0, line))
            cursor += 1.0 + spoken(line)

    tally = _tally(events)
    close = pick(NARRATION_CLOSE, "close", tally=tally)
    # After everything else has finished speaking, not merely near the end --
    # placing it by duration alone put the sign-off before the last event.
    last_end = max((at + spoken(text) for at, text in filled), default=0.0)
    # Late enough to be a sign-off, early enough to finish before the footage
    # does -- placing it purely by duration put it before the final event, and
    # placing it purely after that final line ran it past the end of the video.
    close_at = max(last_end + 0.5, duration_s - spoken(close) - 0.5)
    overruns = close_at + spoken(close) > duration_s
    filled.append((min(close_at, max(0.0, duration_s - 0.5)), close))
    filled.sort()

    lines = ["# Commentary script for %s%s" % (name, ", " + level if level else ""),
             "# %d seconds of footage, about %d words, read at ~%d wpm."
             % (duration_s,
                sum(len(t.split()) for _a, t in filled), WORDS_PER_MINUTE),
             "# Timestamps are when each line should START.",
             ""]
    if overruns:
        lines.insert(3, "# NOTE the sign-off runs past the end of the footage. "
                        "Trim it, or hold the last frame.")
    for at, text in filled:
        lines.append("[%s] %s" % (stamp(int(at * FPS)), text))
    return "\n".join(lines)


# ------------------------------------------------------------ the copy ----
# Nostalgia is the actual pitch. The audience for a World 1-1 run is not
# looking for a tutorial -- they played this, or watched a sibling play it, and
# the description has to say so before it says anything else.
DESC_OPEN = [
    "If you were there for this one, you already know the sound of that first "
    "overworld theme better than you know your own phone number.",
    "Some games you remember. This one you can still hear. The moment that "
    "music starts, you are eight years old and it is a Saturday.",
    "There was a time when this screen was the whole world, and getting past it "
    "was the most important thing happening that week.",
]

DESC_BODY = [
    "No save states, no rewinds, no second attempts. One take, mistakes "
    "included, exactly the way it was played when the only thing standing "
    "between you and the start of the level was a very small number of lives.",
    "Played straight through on emulation of the original hardware. Same enemy "
    "placement, same physics, same unforgiving gap you have been falling into "
    "since the eighties.",
    "Everything here is the original game, untouched. No romhacks, no practice "
    "tools, no rewinding the bad bits. What you see is what happened.",
]

DESC_CLOSE = [
    "Drop a comment if you remember where the secret in this level is. Half of "
    "you do. The other half are about to find out.",
    "Tell me in the comments how old you were when you first played this. Be "
    "honest, we are all in the same boat here.",
    "If this brought something back, subscribe -- more of the same, one level at "
    "a time, no save states.",
]

YOUTUBE_TAGS = ["retro gaming", "nes", "nintendo", "nostalgia", "80s games",
                "90s kids", "retro games", "classic gaming", "gameplay",
                "full playthrough", "no commentary", "childhood games"]


def build_description(game, level, events, duration_s, players, watermark, seed=None):
    """A real YouTube description: hook, context, what happened, and a prompt.

    The old one was three lines and a stat count, which is not a description --
    it is a caption with delusions. Long-form platforms expect several hundred
    words and use them for search."""
    rng = random.Random(seed if seed is not None else "%s-desc" % game)
    name = pretty_game(game)
    where = ("%s, %s" % (name, level)) if level else name

    counts = {}
    for _frame, kind, _detail in events:
        counts[kind] = counts.get(kind, 0) + 1

    chapters = []
    for frame, kind, detail in events:
        if kind in ("powerup", "1up", "clear", "death", "pipe"):
            secs = int(frame / FPS)
            chapters.append("%d:%02d  %s" % (secs // 60, secs % 60, {
                "powerup": "Power-up",
                "1up": "Extra life",
                "clear": "Level clear",
                "death": "Death",
                "pipe": "Into a pipe",
            }[kind]))

    parts = [where + ".", "", rng.choice(DESC_OPEN), "", rng.choice(DESC_BODY), ""]
    if players == 2:
        parts += ["Two players: one human, one trained neural network. "
                  "The machine has played this level more times than any person "
                  "alive, and it still cannot be trusted near a pit.", ""]
    if chapters:
        # YouTube only turns these into real chapters when the list starts at
        # 0:00, so the run-up is an entry rather than an omission.
        chapters.insert(0, "0:00  Start")
        parts.append("CHAPTERS")
        parts.extend("  " + c for c in chapters[:14])
        parts.append("")
    tally = _tally(events)
    if tally:
        parts += [tally, ""]
    parts += [rng.choice(DESC_CLOSE), "",
              "Runtime: %d:%02d" % (int(duration_s) // 60, int(duration_s) % 60), ""]
    if watermark:
        parts += [watermark, ""]
    parts.append(" ".join("#" + t.replace(" ", "") for t in YOUTUBE_TAGS[:8]))
    return "\n".join(parts)


def build_short_caption(game, level, events, watermark, tags, seed=None):
    """The one-field caption TikTok and Instagram give you: hook then tags."""
    rng = random.Random(seed if seed is not None else "%s-short" % game)
    name = pretty_game(game)
    hooks = [
        "%s. If you know, you know." % (level or name),
        "%s%s, one take, no save states. Tell me you remember this."
        % (name, ", " + level if level else ""),
        "Be honest, how old were you when you first played this? %s%s"
        % (name, ", " + level if level else ""),
    ]
    counts = {}
    for _f, kind, _d in events:
        counts[kind] = counts.get(kind, 0) + 1
    if counts.get("death"):
        hooks.append("Died %d time%s on a level I have beaten a hundred times. %s"
                     % (counts["death"], "" if counts["death"] == 1 else "s", name))
    return "%s\n\n%s\n%s" % (rng.choice(hooks), watermark, tags)


def build_metadata(game, players, events, title, watermark, level=None,
                   duration_s=0.0):
    counts = {}
    for _frame, kind, _detail in events:
        counts[kind] = counts.get(kind, 0) + 1
    tags = list(YOUTUBE_TAGS)
    if players == 2:
        tags += ["ai", "human vs ai", "machine learning", "reinforcement learning"]
    hashtags = {
        "youtube": " ".join("#" + t.replace(" ", "") for t in tags[:10]),
        "tiktok": " ".join("#" + t.replace(" ", "")
                           for t in ["retro", "nostalgia", "nes", "gaming",
                                     "90skids", "fyp"]),
        "instagram": " ".join("#" + t.replace(" ", "")
                              for t in ["retrogaming", "nostalgia", "nes",
                                        "nintendo", "90skids", "80skids",
                                        "retro", "gaming", "reels"]),
    }
    return {
        "title": title,
        "game": game,
        "level": level,
        "players": players,
        "duration_seconds": round(duration_s, 1),
        "event_counts": counts,
        "description": build_description(game, level, events, duration_s,
                                         players, watermark),
        "tags": tags,
        "hashtags": hashtags,
        "captions": {
            "tiktok": build_short_caption(game, level, events, watermark,
                                          hashtags["tiktok"]),
            "instagram": build_short_caption(game, level, events, watermark,
                                             hashtags["instagram"]),
        },
    }


# Limits worth catching BEFORE you are standing in the upload form with copy
# the site silently truncates. YouTube's 100-character title and Instagram's
# 30-hashtag cap are the two that actually bite; the description limits are
# generous but a long description is exactly when you stop counting.
LIMITS = {
    "youtube_title": 100,
    "youtube_description": 5000,
    "youtube_tags_total": 500,
    "caption": 2200,
    "instagram_hashtags": 30,
}


def _warn(label, value, limit, unit="characters"):
    if value > limit:
        return "  !! %s is %d %s, over the %d limit -- it gets cut off\n" % (
            label, value, unit, limit)
    return ""


def paste_block(meta):
    """Every upload form's fields, in the order that form asks for them.

    The upload stays manual, so the tedium left is retyping copy into three
    different sites. This is that, ready to copy -- including the full
    description rather than a three-line stub."""
    title = meta.get("title", "")
    description = meta.get("description", "")
    tags = meta.get("tags", [])
    hashtags = meta.get("hashtags", {})
    captions = meta.get("captions", {})
    yt_tags = ", ".join(tags)
    tt = captions.get("tiktok", "")
    ig = captions.get("instagram", "")

    warn = ""
    warn += _warn("YouTube title", len(title), LIMITS["youtube_title"])
    warn += _warn("YouTube description", len(description), LIMITS["youtube_description"])
    warn += _warn("YouTube tags", len(yt_tags), LIMITS["youtube_tags_total"])
    warn += _warn("TikTok caption", len(tt), LIMITS["caption"])
    warn += _warn("Instagram caption", len(ig), LIMITS["caption"])
    warn += _warn("Instagram hashtags", len(hashtags.get("instagram", "").split()),
                  LIMITS["instagram_hashtags"], "tags")

    bar = "=" * 70
    rule = lambda label: ("-- " + label + " ").ljust(70, "-")
    out = [
        bar,
        "YOUTUBE  --  studio.youtube.com, upload the _16x9.mp4",
        bar,
        "\n" + rule("Title"),
        title,
        "\n" + rule("Description (%d chars)" % len(description)),
        description,
        "\n" + rule("Tags (comma separated)"),
        yt_tags,
        "\n" + rule("Visibility"),
        "Private, or Schedule -- then review it and make it public.",
        "\n" + bar,
        "TIKTOK  --  tiktok.com/upload, upload the _9x16.mp4",
        bar,
        "\n" + rule("Caption (%d chars)" % len(tt)),
        tt,
        "\n" + bar,
        "INSTAGRAM  --  Reels, upload the _9x16.mp4",
        bar,
        "\n" + rule("Caption (%d chars)" % len(ig)),
        ig,
    ]
    if warn:
        out += ["\n" + bar, "LENGTH WARNINGS", bar, warn.rstrip()]
    return "\n".join(out) + "\n"


UPLOAD_PLAN = """\
UPLOAD PLAN -- what is safe to automate today, and what is not.

FIRST, THE SHORTCUT. "Verified" can mean two different things, and only one of
them is work for you:
  (a) YOUR OWN API project passing each platform's audit. Free, but it is three
      separate approval processes and they gate everything below.
  (b) A THIRD-PARTY POSTING API that has ALREADY passed all three -- Ayrshare,
      Postproxy, Post for Me and similar. You authorise your accounts to their
      app and post through one endpoint. This is what Google's help page means
      by "re-upload via a verified API service": their project is verified, so
      YouTube uploads through it are NOT locked private, and you can set
      privacyStatus=private and flip it public yourself. It also solves
      Instagram's public-URL requirement (they host the media) and token
      refresh. Roughly $10-150/month depending on volume.
If the goal is working automation rather than owning the integration, (b) is
the shorter path by weeks, and the per-platform notes below stop applying.

IF YOU BUILD IT YOURSELF, per platform:

TikTok      AUTOMATABLE NOW. An unaudited Content Posting API client may post
            at SELF_ONLY visibility (5 users / 24h, and the account must be
            private at post time). You then flip the account public and each
            post to "Everyone" by hand. This is exactly the review-then-publish
            flow, and it is the only platform that has it.

YouTube     DO NOT AUTOMATE YET. Uploads via videos.insert from an unverified
            API project are LOCKED as private. That is not the same as setting
            privacyStatus=private -- per Google's help pages the lock cannot be
            appealed, and the fix is to re-upload via a verified project or by
            hand. Automating now permanently burns every video you upload.
            Upload <slug>_16x9.mp4 through the website until the API audit
            passes.

Instagram   CANNOT DO THIS FLOW AT ALL. The publishing API has no draft or
            private state: media_publish goes live immediately. It also needs a
            Business/Creator account, App Review for instagram_content_publish,
            and a PUBLIC URL for the file, because Meta fetches the media
            rather than accepting bytes -- so it drags in hosting too. Post
            <slug>_9x16.mp4 from the phone until that is in place.
"""


def main():
    cfg = load_studio_config()
    # Style is data now: whatever studio.json sets is merged over the
    # renderer defaults and written into overlays.json, so restyle.py can
    # change any of it later without touching code.
    # Styling comes from a "style" object plus a few scalars, NEVER from
    # whatever happens to share a name with a style group: studio.json's
    # top-level "watermark" is the handle text, while DEFAULT_STYLE's
    # "watermark" is how to draw it, and letting the two meet put a string
    # where a dict belonged.
    style_cfg = dict(cfg.get("style", {}))
    for key in ("font", "title_font", "char_width_ratio", "transition",
                "transition_seconds", "blur", "crf"):
        if key in cfg:
            style_cfg.setdefault(key, cfg[key])
    style = merge_style(style_cfg)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", help="stable-retro game id (see list_games.py). Required unless --paste-block or --print-upload-plan.")
    parser.add_argument("--players", type=int, choices=[1, 2], default=1, help="1 = you alone. 2 = you plus an AI player, via play_human_vs_ai. (default: %(default)s)")
    parser.add_argument("--mode", choices=["versus", "coop", "race"], default="versus", help="Two-player match type, ignored when --players 1. (default: %(default)s)")
    parser.add_argument("--model", default=None, help="Checkpoint driving the AI player when --players 2. Without one the AI plays randomly, which makes for a much weaker video.")
    parser.add_argument("--state", default=None)
    parser.add_argument("--title", default=None, help="Overlay text and metadata title. Left out, one is written from what happened in the run -- cleared without dying, died twice, 14 coin run -- so nothing needs typing.")
    parser.add_argument("--watermark", default=cfg.get("watermark", ""), help="Handle burnt into the bottom of both videos. Set it once as \"watermark\" in studio.json instead of passing it every run. (default: from studio.json)")
    parser.add_argument("--record-dir", default=None, help="Where the emulator writes its .bk2. Defaults to the staged folder itself, so a run produces one self-contained folder; give a path only if you want the raw captures kept separately, and the .bk2 is moved into the staged folder afterwards either way.")
    parser.add_argument("--out-dir", default="./studio_out", help="Parent for the dated review folder (default: %(default)s)")
    parser.add_argument("--from-mp4", default=None, help="Skip playing and re-cut an existing mp4 -- for redoing overlays without replaying.")
    parser.add_argument("--no-events", action="store_true", help="Skip the replay scan that builds the event timeline (faster, but narration.txt becomes generic)")
    parser.add_argument("--short-seconds", type=float, default=cfg.get("short_seconds", 15.0), help="Total length budget for the 9x16 short. The 16x9 master is always the full run. 0 keeps the short full length too. (default: %(default)s)")
    parser.add_argument("--clip-seconds", type=float, default=4.0, help="Seconds kept around each highlight. Smaller means more separate moments in the same budget, larger means fewer but with more room to breathe. (default: %(default)s)")
    parser.add_argument("--clip-lead", type=float, default=-1.0, help="Seconds of run-up kept before each event. The default of -1 means per-event: a death starts 4s early because the lives counter only moves at the END of the death sequence, while a coin starts 1s early. Any value >= 0 overrides that for every kind. (default: %(default)s)")
    parser.add_argument("--transition", choices=TRANSITIONS, default=cfg.get("transition", "fade"), help="How cuts are joined in the short. fade goes through black and is the safest; dissolve and pixelize cross-fade the pair and cost overlap at every join; none hard-cuts. (default: %(default)s)")
    parser.add_argument("--transition-seconds", type=float, default=0.25, help="Length of each transition in seconds. (default: %(default)s)")
    parser.add_argument("--no-captions", action="store_true", help="Turn off the timed commentary captions. They are written from the event log, so they land on the thing they are about.")
    parser.add_argument("--level", default=cfg.get("level"), help="Where in the game this run is, e.g. World 1-1. Shown after the game name in the title. Set it once as the level key in studio.json.")
    parser.add_argument("--writer", choices=("table", "ollama"), default=cfg.get("writer", "table"), help="Who writes the captions, commentary and descriptions. 'table' uses the built-in pools, which repeat across runs. 'ollama' asks a model running on this machine to write them fresh against what actually happened, falling back to the tables if it is unreachable or returns nothing usable. (default: %(default)s)")
    parser.add_argument("--writer-model", default=cfg.get("writer_model", writer.DEFAULT_MODEL), help="Ollama model for --writer ollama. See writer.py for what fits a 24GB card. (default: %(default)s)")
    parser.add_argument("--writer-host", default=cfg.get("writer_host", writer.DEFAULT_HOST), help="Where Ollama is listening. (default: %(default)s)")
    parser.add_argument("--list-writer-models", action="store_true", help="Show which Ollama models are installed, with notes on what suits a 24GB card, then exit")
    parser.add_argument("--paste-block", metavar="DIR", default=None, help="Print the copy-paste block for an already staged folder (or a metadata.json) and exit. A normal run also writes it to paste.txt.")
    parser.add_argument("--print-upload-plan", action="store_true", help="Explain what can and cannot be automated per platform, then exit")
    args = parser.parse_args()

    if args.print_upload_plan:
        print(UPLOAD_PLAN)
        return

    if args.list_writer_models:
        if not writer.available(args.writer_host):
            print(f"No Ollama server at {args.writer_host}. Start one with:  ollama serve")
        else:
            found = writer.installed_models(args.writer_host)
            print("Installed:" if found else "No models pulled yet.")
            for name in found:
                print("  " + name)
        print()
        print(writer.MODEL_NOTES)
        return

    if args.paste_block:
        path = args.paste_block
        if os.path.isdir(path):
            path = os.path.join(path, "metadata.json")
        if not os.path.exists(path):
            sys.exit(f"No metadata.json at {path}")
        with open(path) as handle:
            print(paste_block(json.load(handle)))
        return

    if not args.game:
        sys.exit("--game is required (see list_games.py)")

    for key in ("font", "title_font"):
        if not os.path.exists(style[key]):
            # Warn and carry on rather than stopping: a font path that has
            # moved should not cost a recording that has already been played.
            print(f"WARNING: {key} not found: {style[key]}")
            print(f"         falling back to {DEFAULT_FONT}")
            style[key] = DEFAULT_FONT
    if not os.path.exists(style["font"]):
        sys.exit(f"No usable font. Install the fallback:\n"
                 f"  sudo apt install -y fonts-dejavu-core")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is not on PATH -- see README step 1.")

    bk2_path = None

    # The staged folder is created BEFORE playing, and the emulator records
    # straight into it, so one run produces exactly one folder holding
    # everything: the .bk2, the capture it renders from, both finished videos
    # and all the social collateral. Nothing is left anywhere else, and nothing
    # is duplicated -- the .bk2 used to live in recordings/ AND be copied here.
    slug = slugify(args.title or pretty_game(args.game))
    folder = os.path.join(args.out_dir, f"{datetime.now():%Y%m%d-%H%M%S}_{slug}")
    os.makedirs(folder, exist_ok=True)
    # Only a --record-dir needs creating; the staged folder already exists.
    record_dir = args.record_dir or folder
    if args.record_dir:
        os.makedirs(record_dir, exist_ok=True)

    if args.from_mp4:
        # Someone else's file: copy it in rather than move it.
        native = os.path.join(folder, f"{slug}_source.mp4")
        shutil.copy2(args.from_mp4, native)
    else:
        before = set(glob.glob(os.path.join(record_dir, "*.bk2")))
        started = time.time()
        if args.players == 1:
            from play_and_record import play_human_episode
            print(f"Starting {args.game} -- close the window when you are done.\n")
            play_human_episode(args.game, args.state, record_dir)
        else:
            from play_human_vs_ai import play_match
            if not args.model:
                print("WARNING: --players 2 with no --model means the AI player is "
                      "picking random buttons. Fine for a pipeline test, weak as content.\n")
            play_match(args.game, args.state, args.model, record_dir, mode=args.mode)

        from play_and_record import find_new_bk2, render_to_mp4
        bk2_path = find_new_bk2(record_dir, before, started_at=started)
        if not bk2_path:
            sys.exit("No .bk2 was written -- nothing to render.")
        if os.path.dirname(os.path.abspath(bk2_path)) != os.path.abspath(folder):
            # Only when --record-dir sent it elsewhere.
            bk2_path = shutil.move(bk2_path, os.path.join(folder, os.path.basename(bk2_path)))

        # playback_movie writes the mp4 beside the .bk2 it replayed, which is
        # already the staged folder, so this is an in-folder rename.
        native = render_to_mp4(bk2_path)
        if not native:
            sys.exit("playback_movie produced no mp4 (ffmpeg missing?). The .bk2 is kept.")
        staged = os.path.join(folder, f"{slug}_source.mp4")
        if os.path.abspath(native) != os.path.abspath(staged):
            native = shutil.move(native, staged)

    # Events first: the short is cut from them, so this has to run before the
    # encodes rather than after.
    events = []
    if bk2_path and not args.no_events:
        print("Scanning the replay for events ...")
        try:
            events = read_events(bk2_path, args.game)
        except Exception as exc:
            # The scan replays the whole run through the emulator, so it has
            # plenty of ways to fail that have nothing to do with video. It
            # must not take the videos down with it -- without this, a scan
            # error left a staged folder holding the source and nothing else.
            print(f"WARNING: event scan failed ({exc.__class__.__name__}: {exc})")
            print("         rendering without captions; the videos are unaffected.")
            events = []
        with open(os.path.join(folder, "events.csv"), "w", newline="") as handle:
            rows = csv.writer(handle)
            rows.writerow(["frame", "timestamp", "kind", "detail"])
            for frame, kind, detail in events:
                rows.writerow([frame, stamp(frame), kind, detail])

    duration = 0.0
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration", "-of", "csv=p=0", native],
                               capture_output=True, text=True, check=True)
        duration = float(probe.stdout.strip())
    except Exception:
        pass

    segments = []
    if args.short_seconds and duration:
        segments = highlight_segments(events, duration, args.short_seconds,
                                      seg_s=args.clip_seconds,
                                      lead=(args.clip_lead if args.clip_lead >= 0 else None))

    title = args.title or auto_title(args.game, args.level)

    # A model, when one is asked for and reachable; the tables otherwise. Every
    # writer entry point returns None on any failure, so an unreachable model
    # costs a bit of variety and never the run.
    ai = None
    if args.writer == "ollama":
        if not writer.available(args.writer_host):
            print(f"WARNING: no Ollama at {args.writer_host} -- using the built-in tables.")
            print( "         start one with 'ollama serve', or drop --writer ollama.")
        else:
            print(f"Writing with {args.writer_model} ...")
            ai = dict(model=args.writer_model, host=args.writer_host)

    ctx = dict(game=pretty_game(args.game), level=args.level, duration_s=duration,
               players=args.players, events=events, fps=FPS)

    lines = []
    if not args.no_captions:
        written_caps = writer.captions(**ctx, **ai) if ai else None
        if written_caps:
            lines = [(c["at"], c["text"]) for c in written_caps]
        else:
            if ai:
                print("  (captions: falling back to the tables)")
            lines = caption_script(events, duration, args.game)
    if not args.title:
        print("Title: %s" % title)
    if lines:
        print("Captions (%d):" % len(lines))
        for at, text in lines:
            print("    %s  %s" % (stamp(int(at * FPS)), text))

    if segments:
        kept = sum(d for _s, d in segments)
        print("Short is a cut: %d highlight%s, %.1fs of %.1fs"
              % (len(segments), "" if len(segments) == 1 else "s", kept, duration))
        for seg_start, seg_dur in segments:
            near = [d for f, _k, d in events
                    if seg_start <= f / FPS <= seg_start + seg_dur]
            print("    %s +%.1fs  %s" % (stamp(int(seg_start * FPS)), seg_dur,
                                         ", ".join(near[:3]) or "(no event)"))

    # Everything drawn on the video, as data. This is the file restyle.py edits:
    # wording, timing, font, size, colour and position all live here rather
    # than in the code, so changing any of them never means changing Python.
    spec = {
        # Relative, so the whole folder can be moved or copied and still
        # re-render. overlays.render_spec resolves it against the folder.
        "source": os.path.basename(native),
        "outputs": [
            {"file": f"{slug}_16x9.mp4", "width": 1920, "height": 1080},
            {"file": f"{slug}_9x16.mp4", "width": 1080, "height": 1920},
            # Clean master: HD, correctly framed, nothing burnt in. The source
            # mp4 is native NES resolution and unusable as delivery footage, so
            # without this there was no full-size copy free of text.
            {"file": f"{slug}_16x9_clean.mp4", "width": 1920, "height": 1080,
             "overlays": False},
        ],
        "title": title,
        "watermark": args.watermark,
        "captions": [{"at": round(at, 2), "text": text} for at, text in lines],
        "style": style,
    }
    if segments:
        spec["segments"] = [[round(a, 3), round(b, 3)] for a, b in segments]
    save_spec(spec, os.path.join(folder, "overlays.json"))

    written = render_spec(spec, out_dir=folder)
    wide, tall, clean = written[0], written[1], written[2]

    with open(os.path.join(folder, "narration.txt"), "w") as handle:
        written_narr = writer.narration(**ctx, **ai) if ai else None
        if written_narr:
            handle.write("# Commentary for %s%s, written by %s.\n"
                         % (pretty_game(args.game),
                            ", " + args.level if args.level else "",
                            args.writer_model))
            handle.write("# Timestamps are when each line should START.\n\n")
            for item in written_narr:
                handle.write("[%s] %s\n"
                             % (stamp(int(item["at"] * FPS)), item["text"]))
        else:
            if ai:
                print("  (narration: falling back to the tables)")
            handle.write(narration(events, duration, args.game, args.players,
                                   level=args.level) + "\n")
    with open(os.path.join(folder, "captions.txt"), "w") as handle:
        for at, text in lines:
            handle.write("%s  %s\n" % (stamp(int(at * FPS)), text))
    meta = build_metadata(args.game, args.players, events, title,
                          args.watermark, level=args.level,
                          duration_s=duration)
    written_copy = writer.copy(**ctx, watermark=args.watermark, **ai) if ai else None
    if written_copy:
        meta["description"] = written_copy["description"]
        meta["tags"] = written_copy["tags"] or meta["tags"]
        meta["captions"] = {"tiktok": written_copy["tiktok"],
                            "instagram": written_copy["instagram"]}
        meta["written_by"] = args.writer_model
    elif ai:
        print("  (descriptions: falling back to the tables)")
    with open(os.path.join(folder, "metadata.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    with open(os.path.join(folder, "UPLOAD.txt"), "w") as handle:
        handle.write(UPLOAD_PLAN)
    block = paste_block(meta)
    with open(os.path.join(folder, "paste.txt"), "w") as handle:
        handle.write(block)

    print("\n" + "=" * 66)
    print(f"Staged for review: {folder}")
    print(f"  {os.path.basename(wide)}   -> YouTube (upload by hand until audited)")
    span = ("[%d highlights, %.1fs]" % (len(segments), sum(d for _s, d in segments))
            if segments else "[full length]")
    print(f"  {os.path.basename(tall)} {span}  -> TikTok / Reels / Shorts")
    print(f"  {os.path.basename(clean)}   <- HD, no text, for re-edits and thumbnails")
    print(f"  {os.path.basename(native)}   <- the capture everything is rendered from")
    if bk2_path:
        print(f"  {os.path.basename(bk2_path)}   <- the raw replay, a few KB")
    print("  paste.txt        <- the three upload forms, ready to copy")
    print(f"  overlays.json, metadata.json, narration.txt, events.csv ({len(events)} events)")
    print(f"\nEdit overlays.json and re-render in seconds:")
    print(f"  python restyle.py {folder}")
    print("\n" + block)


if __name__ == "__main__":
    main()
