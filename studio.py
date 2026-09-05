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
    # No {game} here: the title burnt across the top already says which game it
    # is, and repeating it wastes the one line that has to earn the watch.
    #
    # A STARTING TABLE, not the final wording. Every run writes the chosen
    # lines into overlays.json beside the video and restyle.py re-renders after
    # you edit it, so the funniest version of any of these is the one written
    # after actually watching the clip.
    "open": ["watch a grown adult lose to 1988",
             "one take. no edits. no talent.",
             "this is going to be rough for everyone"],
    "death": ["he died. to that. to THAT.",
              "a mushroom with legs ended his run",
              "that goomba has a job and a family. he has neither.",
              "walked into it like it owed him money",
              "he had one job. the job was go right.",
              "death {n}. the goombas have a group chat about him.",
              "thirty years of gaming, everyone. thirty."],
    "shrink": ["small again. like his prospects.",
               "held that mushroom for nine entire seconds",
               "back to factory settings",
               "the mushroom filed a complaint and left"],
    "powerdown": ["the tail is gone. so is the dignity.",
                  "downgraded. deserved.",
                  "briefly had something nice"],
    "powerup": ["a mushroom. this changes nothing.",
                "briefly competent. savour it.",
                "peak. it is all downhill from here.",
                "do not get attached"],
    "1up": ["an extra life. for what, exactly.",
            "1-up. now he can disappoint twice."],
    "coin": ["{n} coins. still cannot jump.",
             "{n} coins and not one good decision",
             "hoarding currency, squandering talent"],
    "clear": ["cleared it. the level went easy on him.",
              "finished. the bar was on the floor.",
              "level complete. nobody clapped."],
}

CAPTION_HOLD = 2.6        # seconds each line stays up
CAPTION_GAP = 5.0         # minimum seconds between lines, so they do not crowd
COIN_EVERY = 10           # only caption coins at milestones


def caption_script(events, duration, game, seed=None):
    """(start_seconds, text) pairs for the whole run.

    Deliberately sparse. A caption on every coin would be a wall of text over
    the gameplay, so coins only speak at milestones and nothing lands within
    CAPTION_GAP of the previous line."""
    rng = random.Random(seed if seed is not None else game)
    lines = []

    def pool(kind, count=None):
        options = CAPTION_LINES.get(kind)
        if not options:
            return None
        text = rng.choice(options)
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
        if lines and at - lines[-1][0] < CAPTION_GAP:
            continue
        lines.append((at, text))
    return lines











# Weighted so the cut lands on what is worth watching. A clear or a 1-Up is the
# payoff, a power-up is the setup, a hit is at least dramatic; coins are common
# enough that they should only break ties.
EVENT_WEIGHTS = {"clear": 5.0, "1up": 5.0, "death": 4.5, "powerup": 4.0,
                 "powerdown": 2.0, "shrink": 2.5, "coin": 1.0, "score": 0.25}

# How far BEFORE the logged frame each kind of moment actually starts.
#
# Events are logged when the underlying value changes, and some values change
# long after the thing you want to watch. A death is the worst case: `lives`
# only decrements at the END of the death sequence, after the death-jump
# animation and the screen fade -- roughly three seconds after the hit that
# caused it. A 1.5s lead there starts the clip on the aftermath and misses
# the kill entirely, which is exactly what it looked like. Coins are the other
# extreme: the counter moves on the frame you touch them.
EVENT_LEAD = {"death": 4.0, "clear": 4.0, "powerdown": 2.5, "shrink": 2.5, "1up": 2.0,
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

    events, frame = [], 0
    prev = {}
    last_pos = None
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
                events.append((frame, "clear", "level ended"))
        if position is not None:
            last_pos = position

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
    return filter_power_noise(events, total_frames=frame)


def narration(events, duration_s, game, players):
    """A timestamped script. Deliberately plain text rather than synthesised
    audio: which TTS to use is a real decision (a local voice is free and
    sounds it, a hosted one costs per character and needs an API key) and
    hardcoding one here would make that choice for you. Pipe this into
    whichever you pick."""
    who = "I play" if players == 1 else "an AI and I play"
    lines = [f"[0:00] {game} -- {who}. {int(duration_s)} seconds of footage."]
    counts = {}
    for _frame, kind, _detail in events:
        counts[kind] = counts.get(kind, 0) + 1
    for frame, kind, detail in events:
        if kind in ("death", "powerup", "powerdown", "shrink", "1up", "clear"):
            lines.append(f"[{stamp(frame)}] {detail}")
    tail = []
    def plural(n, word):
        return f"{n} {word}" if n == 1 else f"{n} {word}s"
    if counts.get("coin"):
        tail.append(plural(counts["coin"], "coin"))
    if counts.get("death"):
        tail.append(plural(counts["death"], "death"))
    if counts.get("powerup"):
        tail.append(plural(counts["powerup"], "power-up"))
    if tail:
        lines.append(f"[end] Final tally: {', '.join(tail)}.")
    return "\n".join(lines)


def build_metadata(game, players, events, title, watermark):
    counts = {}
    for _frame, kind, _detail in events:
        counts[kind] = counts.get(kind, 0) + 1
    tags = ["retrogaming", "nes", "gameplay", "speedrun", "gaming"]
    if players == 2:
        tags += ["ai", "humanvsai", "machinelearning"]
    description = (
        f"{game}, played live.\n\n"
        f"Coins: {counts.get('coin', 0)} | deaths: {counts.get('death', 0)} | "
        f"power-ups: {counts.get('powerup', 0)}\n\n"
        f"{watermark}"
    )
    return {
        "title": title,
        "description": description,
        "game": game,
        "players": players,
        "event_counts": counts,
        "hashtags": {
            "youtube": " ".join("#" + t for t in tags),
            "tiktok": " ".join("#" + t for t in tags[:5] + ["fyp"]),
            "instagram": " ".join("#" + t for t in tags + ["reels", "retro"]),
        },
    }


# Field limits worth catching BEFORE you are standing in the upload form with
# a title the site silently truncates. YouTube's 100-character title and
# Instagram's 30-hashtag cap are the two that actually bite.
LIMITS = {
    "youtube_title": 100,
    "youtube_description": 5000,
    "youtube_tags_total": 500,
    "caption": 2200,
    "instagram_hashtags": 30,
}


def _warn(label, value, limit, unit="characters"):
    if value > limit:
        return f"  !! {label} is {value} {unit}, over the {limit} limit -- it gets cut off\n"
    return ""


def paste_block(meta):
    """Everything the three upload forms ask for, in the order they ask for it.

    The upload stays manual, so the only tedium left is retyping titles and
    hashtags into three different forms. This is that, ready to copy."""
    title = meta.get("title", "")
    description = meta.get("description", "")
    tags = meta.get("hashtags", {})
    yt_tags = ", ".join(t.lstrip("#") for t in tags.get("youtube", "").split())
    ig_caption = title + "\n\n" + tags.get("instagram", "")
    tt_caption = title + " " + tags.get("tiktok", "")

    warn = ""
    warn += _warn("YouTube title", len(title), LIMITS["youtube_title"])
    warn += _warn("YouTube description", len(description), LIMITS["youtube_description"])
    warn += _warn("YouTube tags", len(yt_tags), LIMITS["youtube_tags_total"])
    warn += _warn("Instagram caption", len(ig_caption), LIMITS["caption"])
    warn += _warn("Instagram hashtags", len(tags.get("instagram", "").split()),
                  LIMITS["instagram_hashtags"], "tags")
    warn += _warn("TikTok caption", len(tt_caption), LIMITS["caption"])

    bar = "=" * 70
    rule = lambda label: ("-- " + label + " ").ljust(70, "-")
    out = [
        bar,
        "YOUTUBE  --  studio.youtube.com, upload the _16x9.mp4",
        bar,
        "\n" + rule("Title"),
        title,
        "\n" + rule("Description"),
        description + "\n\n" + tags.get("youtube", ""),
        "\n" + rule("Tags (comma separated)"),
        yt_tags,
        "\n" + rule("Visibility"),
        "Private, or Schedule -- then review it and make it public.",
        "\n" + bar,
        "TIKTOK  --  tiktok.com/upload, upload the _9x16.mp4",
        bar,
        "\n" + rule("Caption"),
        tt_caption,
        "\n" + bar,
        "INSTAGRAM  --  Reels, upload the _9x16.mp4",
        bar,
        "\n" + rule("Caption"),
        ig_caption,
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
    style = merge_style({k: cfg[k] for k in cfg
                         if k in DEFAULT_STYLE or k in ("font", "title_font")})
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", help="stable-retro game id (see list_games.py). Required unless --paste-block or --print-upload-plan.")
    parser.add_argument("--players", type=int, choices=[1, 2], default=1, help="1 = you alone. 2 = you plus an AI player, via play_human_vs_ai. (default: %(default)s)")
    parser.add_argument("--mode", choices=["versus", "coop", "race"], default="versus", help="Two-player match type, ignored when --players 1. (default: %(default)s)")
    parser.add_argument("--model", default=None, help="Checkpoint driving the AI player when --players 2. Without one the AI plays randomly, which makes for a much weaker video.")
    parser.add_argument("--state", default=None)
    parser.add_argument("--title", default=None, help="Overlay text and metadata title. Left out, one is written from what happened in the run -- cleared without dying, died twice, 14 coin run -- so nothing needs typing.")
    parser.add_argument("--watermark", default=cfg.get("watermark", ""), help="Handle burnt into the bottom of both videos. Set it once as \"watermark\" in studio.json instead of passing it every run. (default: from studio.json)")
    parser.add_argument("--record-dir", default="./recordings")
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
    parser.add_argument("--paste-block", metavar="DIR", default=None, help="Print the copy-paste block for an already staged folder (or a metadata.json) and exit. A normal run also writes it to paste.txt.")
    parser.add_argument("--print-upload-plan", action="store_true", help="Explain what can and cannot be automated per platform, then exit")
    args = parser.parse_args()

    if args.print_upload_plan:
        print(UPLOAD_PLAN)
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
            sys.exit(f"{key} not found: {style[key]}\n"
                     f"  Fix the path in studio.json, or drop the key to use\n"
                     f"  {DEFAULT_FONT}")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is not on PATH -- see README step 1.")

    # Real title is written once the events are known; this only names the folder.
    os.makedirs(args.record_dir, exist_ok=True)
    bk2_path = None

    if args.from_mp4:
        native = args.from_mp4
    else:
        before = set(glob.glob(os.path.join(args.record_dir, "*.bk2")))
        started = time.time()
        if args.players == 1:
            from play_and_record import play_human_episode
            print(f"Starting {args.game} -- close the window when you are done.\n")
            play_human_episode(args.game, args.state, args.record_dir)
        else:
            from play_human_vs_ai import play_match
            if not args.model:
                print("WARNING: --players 2 with no --model means the AI player is "
                      "picking random buttons. Fine for a pipeline test, weak as content.\n")
            play_match(args.game, args.state, args.model, args.record_dir, mode=args.mode)

        from play_and_record import find_new_bk2, render_to_mp4
        bk2_path = find_new_bk2(args.record_dir, before, started_at=started)
        if not bk2_path:
            sys.exit("No .bk2 was written -- nothing to render.")
        native = render_to_mp4(bk2_path)
        if not native:
            sys.exit("playback_movie produced no mp4 (ffmpeg missing?). The .bk2 is kept.")

    slug = slugify(args.title or pretty_game(args.game))
    folder = os.path.join(args.out_dir, f"{datetime.now():%Y%m%d-%H%M%S}_{slug}")
    os.makedirs(folder, exist_ok=True)

    # Events first: the short is cut from them, so this has to run before the
    # encodes rather than after.
    events = []
    if bk2_path and not args.no_events:
        print("Scanning the replay for events ...")
        events = read_events(bk2_path, args.game)
        with open(os.path.join(folder, "events.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "timestamp", "kind", "detail"])
            for frame, kind, detail in events:
                writer.writerow([frame, stamp(frame), kind, detail])

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
    lines = [] if args.no_captions else caption_script(events, duration, args.game)
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
        "source": os.path.abspath(native),
        "outputs": [
            {"file": f"{slug}_16x9.mp4", "width": 1920, "height": 1080},
            {"file": f"{slug}_9x16.mp4", "width": 1080, "height": 1920},
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
    wide, tall = written[0], written[1]

    with open(os.path.join(folder, "narration.txt"), "w") as handle:
        handle.write(narration(events, duration, args.game, args.players) + "\n")
    with open(os.path.join(folder, "captions.txt"), "w") as handle:
        for at, text in lines:
            handle.write("%s  %s\n" % (stamp(int(at * FPS)), text))
    meta = build_metadata(args.game, args.players, events, title, args.watermark)
    with open(os.path.join(folder, "metadata.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    with open(os.path.join(folder, "UPLOAD.txt"), "w") as handle:
        handle.write(UPLOAD_PLAN)
    block = paste_block(meta)
    with open(os.path.join(folder, "paste.txt"), "w") as handle:
        handle.write(block)
    if bk2_path:
        shutil.copy2(bk2_path, folder)

    print("\n" + "=" * 66)
    print(f"Staged for review: {folder}")
    print(f"  {os.path.basename(wide)}   -> YouTube (upload by hand until audited)")
    span = ("[%d highlights, %.1fs]" % (len(segments), sum(d for _s, d in segments))
            if segments else "[full length]")
    print(f"  {os.path.basename(tall)} {span}  -> TikTok / Reels / Shorts")
    print(f"  metadata.json, narration.txt, events.csv ({len(events)} events)")
    print("  paste.txt        <- the three upload forms, ready to copy")
    print("\n" + block)


if __name__ == "__main__":
    main()
