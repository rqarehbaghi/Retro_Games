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

It produces one self-contained folder, and prints two paths: the folder, and
the brief. Nothing else is printed, because everything else is IN the folder:

    UPLOAD_BRIEF.md      paste it into Claude and it can walk the upload
    <slug>_16x9.mp4      1920x1080 master, captioned (YouTube)
    <slug>_9x16.mp4      1080x1920, captioned (Shorts / Reels / TikTok)
    <slug>_16x9_clean.mp4  the same master with no text, for thumbnails
    <slug>_source.mp4    the capture everything above is rendered from
    <slug>.bk2           the replay, which is what makes the folder standalone
    paste.txt            the same copy as three upload forms, to retype by hand
    overlays.json        every word and style rule -- edit, then restyle.py
    metadata.json        title, description and per-platform hashtags
    captions.txt         the on-screen text with timecodes
    events.csv           the raw event timeline the writing was built from

With --voice it also writes narration.txt, narration.wav and _narrated cuts.

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

import tts
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


# Weighted so the cut lands on what is worth watching. A clear or a 1-Up is the
# payoff, a power-up is the setup, a hit is at least dramatic; coins are common
# enough that they should only break ties.
EVENT_WEIGHTS = {"clear": 5.0, "1up": 5.0, "death": 4.5, "powerup": 4.0,
                 "powerdown": 2.0, "shrink": 2.5, "pipe": 2.0, "coin": 1.0,
                 "score": 0.25}



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
    for frame, _kind, _detail in sorted(
            events, key=lambda e: -EVENT_WEIGHTS.get(e[1], 0.5)):
        # A lead given on the command line applies to everything; otherwise
        # each kind gets the run-up it actually needs.
        ahead = lead if lead is not None else 0.0
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
# A clear is now logged at the moment the card is touched (games.json,
# course_clear), and the power byte is not wiped until the course finishes
# unloading about six seconds later. Anchoring 2s after the mark, which was
# enough when the mark WAS the unload, would let that wipe through as "back to
# small mario".
CLEAR_OUTRO_WINDOW = 600


def filter_power_noise(events, total_frames=None):
    """Drop power changes that are really a death or a level ending.

    The power byte at 0x00ED is cleared by the game on BOTH of those, not only
    when something hits you -- games.json records this, and RewardShaper
    already skips it. Two things make it awkward here.

    First, the giveaway arrives LATER than the drop: on a fatal hit the byte
    clears at once while `lives` only decrements at the end of the death
    animation, so nothing at the moment of the drop can say which it was. Hence
    a second pass rather than an inline test.

    Second, for a game with no course_clear address the ending is inferred
    from position, and a run that stops AT the end never collapses position at
    all -- so anchoring only on a following event misses exactly the case that
    kept being reported. A drop within END_OF_RUN_WINDOW of the last frame is
    therefore treated as the outro as well: nothing happening with fifteen
    seconds left and no play afterwards is a hit worth captioning."""
    marks = [(f, kind) for f, kind, _d in events if kind in ("death", "clear")]
    kept = []
    for frame, kind, detail in events:
        if kind in ("shrink", "powerdown"):
            if any(0 <= mark - frame <= POWER_NOISE_WINDOW or
                   0 <= frame - mark <= (CLEAR_OUTRO_WINDOW if mkind == "clear"
                                         else 120)
                   for mark, mkind in marks):
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
CLEAR_TAIL = 180                  # how late the unload may follow the flag


def classify_collapses(collapses, timeline, total_frames):
    """Turn position collapses into 'clear' or 'pipe' events.

    THE FALLBACK, used only for a game games.json has no course_clear address
    for. It is guesswork and it was measurably wrong: checked against rendered
    frames on four SuperMarioBros3 recordings it called a pipe entry a clear,
    and logged nothing at all for three courses that were genuinely cleared.
    Where a clear flag exists, split_collapses uses it instead."""
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


def split_collapses(collapses, clear_windows):
    """Collapses, given a real clear flag to compare them against.

    A collapse inside a clear window is that course unloading and is already
    covered by the clear event; anything else is a pipe. CLEAR_TAIL allows for
    the unload landing just after the flag drops."""
    inside = []
    for frame in collapses:
        if any(start - 60 <= frame <= end + CLEAR_TAIL
               for start, end in clear_windows):
            continue
        inside.append((frame, "pipe", "went down a pipe"))
    return inside


def load_game_ram_config(path, game):
    """Read RAM addresses and variable specs from games.json without importing ML libraries."""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return {}
    entry = data.get(game) or {}
    variables = entry.get("variables") or {}

    def addr(name):
        spec = variables.get(name) or {}
        if spec.get("source") in (None, "unknown") or spec.get("address") is None:
            return None
        return int(spec["address"], 0) if isinstance(spec["address"], str) else int(spec["address"])

    mapped = {
        "progress_address": addr("progress"),
        "powerup_address": addr("powerup"),
        "playstate_address": addr("playstate"),
        "coin_address": addr("coins"),
        "speed_address": addr("pmeter"),
        "clear_address": addr("course_clear"),
        "timer_address": addr("timer"),
    }
    clear = variables.get("course_clear") or {}
    if clear.get("clear_value") is not None:
        mapped["clear_value"] = int(str(clear["clear_value"]), 0)
    progress = variables.get("progress") or {}
    if progress.get("address_high") is not None:
        mapped["progress_address_high"] = int(str(progress["address_high"]), 0)
    if progress.get("source") == "info16":
        mapped["progress_use_info_x"] = True
        if progress.get("address_high") is not None:
            mapped["progress_address_high"] = int(str(progress["address_high"]), 0)
    playstate = variables.get("playstate") or {}
    if playstate.get("in_play_value") is not None:
        mapped["playstate_value"] = int(str(playstate["in_play_value"]), 0)
    meter = variables.get("pmeter") or {}
    if meter.get("full_value") is not None:
        mapped["speed_full"] = int(str(meter["full_value"]), 0)
    return mapped


def read_events(bk2_path, game):
    """Replay the recording and note what happened, so narration and titles
    describe the real run instead of being generic. Degrades to an empty
    timeline for any game games.json has no variables for."""
    try:
        import stable_retro as retro
    except Exception as exc:
        print(f"  (event scan unavailable: {exc})")
        return []

    defaults = load_game_ram_config(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "games.json"), game)
    coin_addr = defaults.get("coin_address")
    power_addr = defaults.get("powerup_address")

    movie = retro.Movie(bk2_path)
    movie.step()
    env = retro.make(game=movie.get_game(), state=None,
                     use_restricted_actions=retro.Actions.ALL,
                     players=movie.players, render_mode="rgb_array")
    state_bytes = movie.get_state()
    if state_bytes:
        env.initial_state = state_bytes
    env.reset()

    prog_key = None
    prog_hi = None
    prog = (defaults.get("progress_use_info_x"), defaults.get("progress_address_high"))
    if prog[0]:
        prog_key = "hpos"
        prog_hi = prog[1]

    timer_addr = defaults.get("timer_address")
    # The one honest signal that a course ended. Absent for a game that has no
    # course_clear entry, in which case the position heuristic is all there is.
    clear_addr = defaults.get("clear_address")
    clear_val = defaults.get("clear_value")

    events, frame = [], 0
    clear_windows, clear_since = [], None
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

        if clear_addr is not None:
            byte = int(ram[clear_addr])
            lit = byte == clear_val if clear_val is not None else byte != 0
            if lit and clear_since is None:
                clear_since = frame
            elif not lit and clear_since is not None:
                clear_windows.append((clear_since, frame))
                clear_since = None

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
    if clear_since is not None:
        # Still lit when the tape stopped -- the run ended on the clear.
        clear_windows.append((clear_since, frame))
    if clear_addr is not None:
        events.extend((start, "clear", "course cleared")
                      for start, _end in clear_windows)
        events.extend(split_collapses(collapses, clear_windows))
    else:
        events.extend(classify_collapses(collapses, timeline, frame))
    events.sort(key=lambda e: e[0])
    return filter_power_noise(events, total_frames=frame)


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


# Per-caption keys the renderer understands and the writer may set. Anything
# not listed stays out of overlays.json, which is a file a person edits.
CAPTION_KEYS = ("hold", "hold_bonus", "closing", "color", "box", "box_color",
                "border_w", "case", "size", "font", "y")


def spec_captions(written, lines):
    """The caption list as overlays.json carries it.

    `lines` holds the times with --caption-offset already applied; `written`
    holds everything else the writer decided, which used to be dropped here --
    so the ask's longer hold never reached the renderer."""
    out = []
    for cap, (at, text) in zip(written, lines):
        entry = {"at": round(at, 2), "text": text}
        entry.update({k: cap[k] for k in CAPTION_KEYS if k in cap})
        out.append(entry)
    return out


def build_metadata(game, players, events, title, watermark, written,
                   level=None, duration_s=0.0, model=""):
    """The upload copy, assembled around what the model wrote.

    Nothing here composes prose any more. Every phrase pool this module used to
    carry was removed: they gave the same eight jokes and the same description
    under every upload, which is the opposite of what a channel needs."""
    counts = {}
    for _frame, kind, _detail in events:
        counts[kind] = counts.get(kind, 0) + 1
    tags = written.get("tags") or []
    return {
        "title": title,
        "game": game,
        "level": level,
        "players": players,
        "duration_seconds": round(duration_s, 1),
        "event_counts": counts,
        "written_by": model,
        "description": written.get("description", ""),
        "tags": tags,
        "hashtags": {
            "youtube": " ".join("#" + t.replace(" ", "") for t in tags[:10]),
        },
        "captions": {
            "tiktok": written.get("tiktok", ""),
            "instagram": written.get("instagram", ""),
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


def win_path(path):
    """The Windows spelling of a path made inside WSL, or None.

    The videos are produced in WSL and then attached in a Windows browser's
    file dialog, which cannot open /home/... or /mnt/g/... at all. A drive
    mount maps back to its letter; everything else lives on the distro's own
    filesystem, which Windows reaches through the wsl.localhost share."""
    distro = os.environ.get("WSL_DISTRO_NAME")
    if not distro:
        return None
    path = os.path.abspath(path)
    mount = re.match(r"^/mnt/([a-z])(/.*)?$", path)
    if mount:
        return mount.group(1).upper() + ":" + (mount.group(2) or "/").replace("/", "\\")
    return "\\\\wsl.localhost\\" + distro + path.replace("/", "\\")


def _field(label, text, limit=None):
    """One form field, fenced so a paste survives newlines and punctuation."""
    head = "**%s**" % label
    if limit:
        over = " -- OVER THE %d LIMIT" % limit if len(text) > limit else ""
        head += " (%d / %d characters%s)" % (len(text), limit, over)
    # Four backticks: the copy is model-written and may itself contain three.
    return "%s\n\n````\n%s\n````\n" % (head, text.strip())


def staged_videos(folder):
    """The rendered cuts in a staged folder, each with where it is posted.

    Found rather than passed, so a brief can be rebuilt for a folder made by
    an earlier run. The suffixes are the ones render_spec writes."""
    found = []
    for suffix, role in (("_16x9.mp4", "YouTube"),
                         ("_9x16.mp4", "TikTok / Reels / Shorts")):
        hits = [f for f in sorted(glob.glob(os.path.join(folder, "*" + suffix)))
                if "_clean" not in f and "_narrated" not in f]
        if hits:
            found.append((role, hits[0]))
    return found


def upload_brief(meta, folder, files):
    """One self-contained file to hand to Claude in a browser.

    Everything the upload needs is inline -- the instructions, the rules and
    the full copy -- because the reader is a chat window that cannot open this
    folder. The paths are there so the human knows what to attach."""
    title = meta.get("title", "")
    description = meta.get("description", "")
    tags = ", ".join(meta.get("tags", []))
    captions = meta.get("captions", {})
    tiktok = captions.get("tiktok", "")
    instagram = captions.get("instagram", "")

    rows, native_rows, translated = [], [], False
    for role, path in files:
        if not path:
            continue
        native = os.path.abspath(path)
        win = win_path(native)
        translated = translated or bool(win)
        rows.append("| %s | `%s` |" % (role, win or native))
        native_rows.append("- `%s`" % native)
    paths = "\n".join(rows)
    # Only worth a second listing when the first one is a translation. Off
    # WSL the two are the same text and repeating it is noise.
    other = ("""
<details><summary>The same files, as the shell that made them sees them</summary>

%s

</details>
""" % "\n".join(native_rows)) if translated else ""

    return """# Upload brief -- %s

Paste this whole file into Claude. Everything needed is below; it was written
by the studio pipeline into `%s`.

## What I want

This is my own gameplay -- one human, one sitting, no save states. Help me put
it on YouTube, TikTok and Instagram.

Rules, and they are not negotiable:

- **Private or unlisted first, every time.** I review the post and make it
  public myself. Never press publish, post or "make public" for me.
- **I attach the video.** A file picker is mine to click, even if you can drive
  my browser. Fill the text fields, then stop and tell me what to attach.
- **One platform at a time.** Give me a field, wait, then the next one.
- **Do not invent anything.** No stats about the run, no achievements, no extra
  hashtags. The copy below is all there is. If a line reads badly, say so and
  offer a rewrite -- do not quietly change it.
- If a field is over its limit, tell me and offer a shorter version.

## The files to attach

| Goes to | File |
|---|---|
%s
%s
## YouTube -- studio.youtube.com

Upload the 16x9 file. Visibility **Private**, or Schedule.

%s
%s
%s
## TikTok -- tiktok.com/upload

Upload the 9x16 file. Set it to **Only me** before posting.

%s
## Instagram -- Reels

Upload the 9x16 file. Save to **Drafts** rather than sharing.

%s
""" % (title or "this run", os.path.abspath(folder), paths, other,
       _field("Title", title, LIMITS["youtube_title"]),
       _field("Description", description, LIMITS["youtube_description"]),
       _field("Tags (comma separated)", tags, LIMITS["youtube_tags_total"]),
       _field("Caption", tiktok, LIMITS["caption"]),
       _field("Caption", instagram, LIMITS["caption"]))


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
    parser.add_argument("--list-games", action="store_true", help="List imported games that have valid ROM files present and exit")
    parser.add_argument("--list-all-games", action="store_true", help="List all known game definitions in stable-retro (including unimported ones) and exit")
    parser.add_argument("--game", help="stable-retro game id. Required unless --brief, --paste-block, --list-games or --print-upload-plan.")
    parser.add_argument("--players", type=int, choices=[1, 2], default=1, help="1 = you alone. 2 = you plus an AI player, via play_engine. (default: %(default)s)")
    parser.add_argument("--boot-screen", action="store_true", help="Start from the power-on title screen (state=NONE) instead of a mid-level save state, so you can pick 1P/2P and the game mode yourself. Passed through to the play window.")
    parser.add_argument("--two-human", action="store_true", help="Two HUMAN players, no AI. Player 1 = keyboard/pad 1, Player 2 = a SECOND gamepad (or the numeric keypad). Pair it with --boot-screen to choose 2 PLAYER GAME at the title; for SMB3 that is the classic alternating two-player game. To keep the AI opponent instead, use --players 2 with --model <checkpoint>.")
    parser.add_argument("--gamepad", action="store_true", help="(Now default) Gamepad and keyboard are both supported seamlessly via pygame.")
    parser.add_argument("--scale", type=int, default=cfg.get("scale", 4), help="Window display scale multiplier (e.g. 4 or 5 for large modern screens, default: %(default)s)")
    parser.add_argument("--fullscreen", action="store_true", help="Launch play window in borderless full screen mode")
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
    parser.add_argument("--clip-lead", type=float, default=0.0, help="Seconds of run-up kept before each event when the short is a highlight cut, so a moment has a little context before it. (default: %(default)s)")
    parser.add_argument("--transition", choices=TRANSITIONS, default=cfg.get("transition", "fade"), help="How cuts are joined in the short. fade goes through black and is the safest; dissolve and pixelize cross-fade the pair and cost overlap at every join; none hard-cuts. (default: %(default)s)")
    parser.add_argument("--transition-seconds", type=float, default=0.25, help="Length of each transition in seconds. (default: %(default)s)")
    parser.add_argument("--no-captions", action="store_true", help="Turn off the timed commentary captions. They are written from the event log, so they land on the thing they are about.")
    parser.add_argument("--level", default=cfg.get("level"), help="Where in the game this run is, e.g. World 1-1. Shown after the game name in the title. Set it once as the level key in studio.json.")
    parser.add_argument("--writer", choices=writer.BACKENDS, default=cfg.get("writer", writer.DEFAULT_BACKEND), help="Who writes the captions, commentary and descriptions. 'auto' cascades: Claude Code -> Anthropic API -> Gemini -> Ollama. 'gemini' calls Google Gemini (GEMINI_API_KEY); 'claude' calls Anthropic API; 'ollama' runs locally. (default: %(default)s)")
    parser.add_argument("--writer-cli", default=cfg.get("writer_cli", writer.DEFAULT_CLI), help="Path to the Claude Code binary for --writer claude-code, if it is not on PATH. Also looked for at ~/.local/bin/claude and ~/.claude/local/claude. (default: %(default)s)")
    parser.add_argument("--claude-model", default=cfg.get("claude_model", writer.CLAUDE_MODEL), help="Claude model for --writer claude. (default: %(default)s)")
    parser.add_argument("--claude-effort", choices=("low", "medium", "high", "xhigh", "max"), default=cfg.get("claude_effort", writer.CLAUDE_EFFORT), help="How hard the Claude model works: low, medium, high, xhigh or max. Lower spends fewer tokens. Writing captions is not intelligence-sensitive, so the default is a step below the API's own. (default: %(default)s)")
    parser.add_argument("--gemini-model", default=cfg.get("gemini_model", writer.GEMINI_MODEL), help="Gemini model for --writer gemini. (default: %(default)s)")
    parser.add_argument("--writer-model", default=cfg.get("writer_model", writer.DEFAULT_MODEL), help="Ollama model for --writer ollama. See writer.py for what fits a 24GB card. (default: %(default)s)")
    parser.add_argument("--no-think", action="store_true", help="Turn off reasoning on the ollama backend. It is ON by default -- a thinking model with thinking disabled writes noticeably worse, and it was disabled for a parsing bug that is since fixed.")
    parser.add_argument("--writer-host", default=cfg.get("writer_host", writer.DEFAULT_HOST), help="Where Ollama is listening. (default: %(default)s)")
    parser.add_argument("--list-writer-models", action="store_true", help="Show which Ollama models are installed, with notes on what suits a 24GB card, then exit")
    parser.add_argument("--no-voice", action="store_true", help="Force the spoken commentary off even when studio.json turns it on.")
    parser.add_argument("--voice", action="store_true", help="Write a spoken commentary script and speak it over the videos. OFF by default -- without it no narration is written at all, which also saves a model call. Speak the narration and lay it over the videos, ducking the game audio under it. Needs qwen-tts (pip install -U qwen-tts soundfile) and a GPU. Without it narration.txt is written but nothing is spoken.")
    parser.add_argument("--voice-model", default=cfg.get("voice_model", tts.DEFAULT_MODEL), help="Qwen3-TTS model for --voice. (default: %(default)s)")
    parser.add_argument("--voice-speaker", default=cfg.get("voice_speaker", tts.DEFAULT_SPEAKER), help="Which preset voice speaks. It must stay FIXED across a run -- Vivian, Serena, Ono_Anna and Sohee are female, Ryan, Eric, Dylan, Aiden and Uncle_Fu male. (default: %(default)s)")
    parser.add_argument("--voice-describe", default=cfg.get("voice_describe", tts.DEFAULT_VOICE), help="How the commentator should sound, in plain words -- Qwen3-TTS designs the voice from this rather than picking a preset. Set it once as voice_describe in studio.json.")
    parser.add_argument("--caption-offset", type=float, default=cfg.get("caption_offset", 0.0), help="Shift every caption by this many seconds. Use it only for a SYSTEMATIC lag -- if one caption is on the wrong moment the model picked the wrong event and this will not help. (default: %(default)s)")
    parser.add_argument("--brief", metavar="DIR", default=None,
                        help="Rebuild UPLOAD_BRIEF.md for an already staged folder and exit. A normal run writes it too.")
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

    if args.brief:
        folder = args.brief
        meta_path = os.path.join(folder, "metadata.json")
        if not os.path.exists(meta_path):
            sys.exit(f"No metadata.json in {folder}")
        with open(meta_path) as handle:
            meta = json.load(handle)
        out = os.path.join(folder, "UPLOAD_BRIEF.md")
        with open(out, "w") as handle:
            handle.write(upload_brief(meta, folder, staged_videos(folder)))
        print(out)
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

    if args.list_games or args.list_all_games:
        try:
            import stable_retro as retro
            all_games = sorted(retro.data.list_games())
            if args.list_all_games:
                print(f"All {len(all_games)} known game definitions in stable-retro:")
                for g in all_games:
                    print(f"  {g}")
            else:
                imported = []
                for g in all_games:
                    try:
                        p = retro.data.get_romfile_path(g)
                        if p and os.path.isfile(p):
                            imported.append(g)
                    except Exception:
                        pass
                print(f"Imported playable games ({len(imported)} of {len(all_games)} definitions):")
                for g in imported:
                    print(f"  {g}")
                if not imported:
                    print("\n(No imported ROMs found. Run: python -m retro.import /path/to/your/roms)")
        except Exception as exc:
            print(f"Could not list games: {exc}")
        return

    if not args.game:
        sys.exit("--game is required (use --list-games to see available games)")

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

    # Checked HERE, before a single frame is played. Every word on the video is
    # written by a model now, so an unreachable one means an unusable run --
    # and finding that out after playing a level would cost the recording.
    if args.writer == "claude-code":
        if not writer.claude_code_available(args.writer_cli):
            sys.exit(
                f"No Claude Code CLI found as {args.writer_cli!r}.\n\n"
                "  It runs against a Claude Pro/Max subscription rather than API\n"
                "  credits, so there is nothing to buy -- but the binary has to be\n"
                "  reachable from THIS shell. A Claude Code installed on Windows is\n"
                "  not on the WSL PATH; WSL needs its own.\n\n"
                "    curl -fsSL https://claude.ai/install.sh | bash\n"
                "    export PATH=" + chr(34) + "$HOME/.local/bin:$PATH" + chr(34) + "   # the usual WSL omission\n"
                "    claude   # log in once\n\n"
                "  Already installed somewhere else?  --writer-cli /path/to/claude\n"
                "  Or skip it:  --writer ollama (local, free) / --writer claude (API).")
    elif args.writer == "claude":
        if not writer.claude_available():
            sys.exit("--writer claude needs the Anthropic SDK:\n"
                     "  pip install anthropic\n"
                     "  then set ANTHROPIC_API_KEY, or run: ant auth login\n\n"
                     "  NOTE this is the Messages API, billed by prepaid credits --\n"
                     "  a Claude Pro subscription does NOT include them. If you have\n"
                     "  Pro and no credits, use --writer claude-code instead.")
    elif not writer.available(args.writer_host):
        sys.exit(
            f"No Ollama server at {args.writer_host}.\n"
            f"  All captions, commentary and descriptions are written by a\n"
            f"  model -- there are no built-in phrases to fall back on.\n\n"
            f"  Start one:   ollama serve\n"
            f"  Get a model: ollama pull {args.writer_model}\n"
            f"  Or use the API instead:  --writer claude")
    elif args.writer_model not in writer.installed_models(args.writer_host):
        sys.exit(
            f"Ollama is running but {args.writer_model!r} is not pulled.\n"
            f"  ollama pull {args.writer_model}\n"
            f"  python studio.py --list-writer-models")

    # OFF by default. The spoken commentary is the least finished part of the
    # pipeline, and leaving it on costs a model call and a synthesis pass on
    # every run for output that is not being used. --voice turns it on;
    # "voice": true in studio.json makes that the default again.
    args.voice = (args.voice or cfg.get("voice", False)) and not args.no_voice
    if args.voice and not tts.available():
        sys.exit("--voice needs Qwen3-TTS:\n"
                 "  pip install -U qwen-tts soundfile\n"
                 "  (about 4GB of weights download on first use)")

    bk2_path = None

    # The staged folder is created BEFORE playing, and the emulator records
    # straight into it, so one run produces exactly one folder holding
    # everything: the .bk2, the capture it renders from, both finished videos
    # and all the social collateral. Nothing is left anywhere else, and nothing
    # is duplicated -- the .bk2 used to live in recordings/ AND be copied here.
    slug = slugify(args.title or pretty_game(args.game))

    # Retrying INTO the folder it came from, rather than beside it. A staged
    # folder is recognisable: a *_source.mp4 with a .bk2 next to it. Without
    # this every retry of a failed write left another dated folder holding a
    # copy of the same recording and no videos, which is not what "retry"
    # means -- and after two failures you are hunting for which one is current.
    reuse = None
    if args.from_mp4:
        parent = os.path.dirname(os.path.abspath(args.from_mp4))
        if (os.path.basename(args.from_mp4).endswith("_source.mp4")
                and glob.glob(os.path.join(parent, "*.bk2"))):
            reuse = parent
    if reuse:
        folder = reuse
        slug = os.path.basename(args.from_mp4)[:-len("_source.mp4")]
        print(f"Retrying in place: {folder}")
    else:
        folder = os.path.join(args.out_dir, f"{datetime.now():%Y%m%d-%H%M%S}_{slug}")
    os.makedirs(folder, exist_ok=True)
    # Only a --record-dir needs creating; the staged folder already exists.
    record_dir = args.record_dir or folder
    if args.record_dir:
        os.makedirs(record_dir, exist_ok=True)

    if args.from_mp4:
        # Someone else's file: copy it in rather than move it.
        native = os.path.join(folder, f"{slug}_source.mp4")
        if os.path.abspath(args.from_mp4) != os.path.abspath(native):
            shutil.copy2(args.from_mp4, native)
        # A staged folder holds the .bk2 next to the mp4, so a retry can still
        # scan the run. Without this, re-running a failed write threw the event
        # timeline away and the model wrote about a run it could not see --
        # which is exactly the case --from-mp4 exists to serve.
        siblings = sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.abspath(args.from_mp4)), "*.bk2")))
        if siblings:
            bk2_path = os.path.join(folder, os.path.basename(siblings[0]))
            if os.path.abspath(siblings[0]) != os.path.abspath(bk2_path):
                shutil.copy2(siblings[0], bk2_path)
            print(f"Reusing the replay beside it: {os.path.basename(bk2_path)}")
    else:
        before = set(glob.glob(os.path.join(record_dir, "*.bk2")))
        started = time.time()
        from play_engine import play_match

        if args.two_human:
            if not args.boot_screen:
                print("NOTE: --two-human without --boot-screen starts from the save")
                print("      state, which for most games is already a ONE-player")
                print("      game -- player 2's pad will do nothing. Add")
                print("      --boot-screen to pick 2 PLAYER GAME at the title.")
            print(f"Starting {args.game} -- TWO human players. "
                  "Close the window when you are done.")
            play_match(args.game, args.state, None, record_dir, players=2,
                       p2_human=True, mode=args.mode, boot_screen=args.boot_screen,
                       scale=args.scale, fullscreen=args.fullscreen,
                       render_mp4=False)
        elif args.players == 1:
            print(f"Starting {args.game} -- single player (gamepad & keyboard enabled). "
                  "Close the window when you are done.\n")
            play_match(args.game, args.state, None, record_dir, players=1,
                       boot_screen=args.boot_screen,
                       scale=args.scale, fullscreen=args.fullscreen,
                       render_mp4=False)
        else:
            if not args.model:
                print("WARNING: --players 2 with no --model means the AI player is "
                      "picking random buttons. Fine for a pipeline test, weak as content.\n")
            play_match(args.game, args.state, args.model, record_dir, players=2,
                       mode=args.mode, boot_screen=args.boot_screen,
                       scale=args.scale, fullscreen=args.fullscreen,
                       render_mp4=False)

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
                                      lead=max(0.0, args.clip_lead))

    title = args.title or auto_title(args.game, args.level)

    # One seed for the whole run, drawn fresh each time. Same footage played
    # twice must not produce the same script twice, and an explicit seed is
    # what guarantees that rather than trusting the server's default.
    seed = random.randrange(1 << 31)
    ai = dict(backend=args.writer, model=args.writer_model,
              gemini_model=args.gemini_model,
              claude_model=args.claude_model, claude_effort=args.claude_effort,
              host=args.writer_host, cli=args.writer_cli, seed=seed,
              think=not args.no_think)
    in_use = {
        "auto": "auto cascade (Claude Code -> Claude API -> Gemini -> Ollama)",
        "gemini": "%s (Google Gemini API)" % args.gemini_model,
        "claude": "%s, effort %s" % (args.claude_model, args.claude_effort),
        "claude-code": "%s via subscription" % args.claude_model,
    }.get(args.writer, args.writer_model)
    print(f"Writing with {in_use} (seed {seed}) ...")

    effective_players = 2 if args.two_human else args.players
    ctx = dict(game=pretty_game(args.game), level=args.level, duration_s=duration,
               players=effective_players, events=events, fps=FPS,
               two_human=args.two_human)

    lines, written_caps = [], []
    if not args.no_captions:
        written_caps = writer.captions(**ctx, **ai)
        if written_caps is None:
            sys.exit(
                "The model returned nothing usable for the captions.\n\n"
                "  Nothing is lost -- the recording and the replay are in\n"
                f"  {folder}\n"
                "  Fix the cause above, then retry the WRITING only (no replaying,\n"
                "  and the event timeline is picked up from the .bk2 beside it):\n\n"
                f"    python studio.py --game {args.game} \\\n"
                f"      --from-mp4 {os.path.join(folder, os.path.basename(native))}\n\n"
                "  If it was an auth error, note that a set ANTHROPIC_API_KEY\n"
                "  overrides any profile from 'ant auth login' -- unset it first.")
        lines = [(max(0.0, c["at"] + args.caption_offset), c["text"])
                 for c in written_caps]
        # Print what each caption was pinned to, so a wrong-moment caption can
        # be told from a wrong-event one without guessing at the video.
        print("Captions (anchored to the event log):")
        for cap, (at, text) in zip(written_caps, lines):
            if cap.get("closing"):
                print("    %s  [the ask]        %s" % (stamp(int(at * FPS)), text))
            elif cap.get("event") is None:
                print("    %s  [opening]        %s" % (stamp(int(at * FPS)), text))
            else:
                logged = events[cap["event"]][0] / FPS
                print("    %s  [%d] %-9s logged %s  %s"
                      % (stamp(int(at * FPS)), cap["event"], cap["kind"],
                         stamp(int(logged * FPS)), text))
    if not args.title:
        print("Title: %s" % title)

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
        "captions": spec_captions(written_caps, lines),
        "style": style,
    }
    if segments:
        spec["segments"] = [[round(a, 3), round(b, 3)] for a, b in segments]
    save_spec(spec, os.path.join(folder, "overlays.json"))

    written = render_spec(spec, out_dir=folder)
    wide, tall, clean = written[0], written[1], written[2]

    # Only asked for when it is going to be used -- this is one of the three
    # model calls a run makes, and writing a script nobody hears is waste.
    written_narr = (writer.narration(**ctx, **ai) or []) if args.voice else []

    def write_narration(placed=None):
        """The script on disk. Real times once the speech exists, order before.

        Narration lines carry an ANCHOR, not a timestamp -- when a line is
        spoken is decided by how long the ones before it turned out to be, and
        that is only known after synthesis. Writing a made-up time here was
        what raised KeyError: 'at' once the anchor replaced it."""
        with open(os.path.join(folder, "narration.txt"), "w") as handle:
            handle.write("# Commentary for %s%s, written by %s (seed %d).\n"
                         % (pretty_game(args.game),
                            ", " + args.level if args.level else "",
                            in_use, seed))
            if placed:
                handle.write("# Times are where each line was actually spoken.\n\n")
                for (at, _path), item in zip(placed, written_narr):
                    handle.write("[%s] %s\n" % (stamp(int(at * FPS)), item["text"]))
            else:
                handle.write("# In order. A line marked (@) waits for that moment;\n"
                             "# the rest run on continuously.\n\n")
                for item in written_narr:
                    anchor = item.get("anchor")
                    mark = ("(@ %s)" % stamp(int(anchor * FPS))) if anchor is not None else "       "
                    handle.write("%s %s\n" % (mark, item["text"]))
    if args.voice:
        write_narration()
        if not written_narr:
            print("  WARNING: the model returned no commentary; "
                  "narration.txt is empty.")

    if args.voice and written_narr:
        # After the videos are rendered, not before: the speech is laid over
        # finished files, so a TTS failure costs the commentary track and
        # nothing else.
        print(f"Speaking {len(written_narr)} lines as {args.voice_speaker} ...")
        try:
            clips = tts.speak_lines(written_narr, os.path.join(folder, "voice"),
                                    model_name=args.voice_model,
                                    voice=args.voice_describe,
                                    speaker=args.voice_speaker)
            track, placed = tts.build_track(
                clips, duration, os.path.join(folder, "narration.wav"))
            write_narration(placed)
            print(f"  narration.wav      the voice alone, {len(placed)} lines")
            for source in (wide, tall):
                stem, ext = os.path.splitext(source)
                spoken = tts.mux(source, track, stem + "_narrated" + ext)
                print(f"  {os.path.basename(spoken)}   voice over the game, music ducked")
        except Exception as exc:                              # noqa: BLE001
            print(f"  WARNING: voice failed ({exc.__class__.__name__}: {exc})")
            print( "           the videos and narration.txt are unaffected.")

    with open(os.path.join(folder, "captions.txt"), "w") as handle:
        for at, text in lines:
            handle.write("%s  %s\n" % (stamp(int(at * FPS)), text))
    written_copy = writer.copy(**ctx, watermark=args.watermark, **ai) or {}
    if not written_copy:
        print("  WARNING: no description came back -- paste.txt will be EMPTY.")
        print(f"           Retry just the writing: python studio.py --game {args.game} \\")
        print(f"             --from-mp4 {os.path.join(folder, os.path.basename(native))}")
        print( "           A smaller model often manages captions and commentary")
        print( "           but not the long description; --writer claude-code is")
        print( "           the fallback that reliably does.")
    meta = build_metadata(args.game, effective_players, events, title, args.watermark,
                          written_copy, level=args.level, duration_s=duration,
                          model=in_use)
    with open(os.path.join(folder, "metadata.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    with open(os.path.join(folder, "UPLOAD.txt"), "w") as handle:
        handle.write(UPLOAD_PLAN)
    with open(os.path.join(folder, "paste.txt"), "w") as handle:
        handle.write(paste_block(meta))
    brief = os.path.join(folder, "UPLOAD_BRIEF.md")
    with open(brief, "w") as handle:
        handle.write(upload_brief(meta, folder, [
            ("YouTube", wide), ("TikTok / Reels / Shorts", tall)]))

    # Two paths, nothing else. Everything that used to be printed here is in
    # the folder: paste.txt holds the three upload forms, UPLOAD_BRIEF.md the
    # same copy written as instructions for a chat window, and --paste-block
    # prints the long version on demand.
    print()
    print(folder)
    print(brief)


if __name__ == "__main__":
    main()
