#!/usr/bin/env python3
"""
One command: play, then get finished videos staged for review and upload.

    python studio.py --game SuperMarioBros3-Nes-v0 --players 1

Play until you close the window, and this produces, in a dated folder:

    <slug>_16x9.mp4    1920x1080 landscape master (YouTube)
    <slug>_9x16.mp4    1080x1920 vertical short (Shorts / Reels / TikTok)
    metadata.json      title, description and per-platform hashtags
    narration.txt      a timestamped script of what actually happened
    events.csv         the raw event timeline the script was built from

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
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FPS = 60.0988


def stamp(frame):
    secs = frame / FPS
    return f"{int(secs // 60):01d}:{secs % 60:05.2f}"


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "session"


def esc(text):
    """Escape a string for ffmpeg's drawtext, which treats ' : \\ % specially.
    Apostrophes are replaced rather than escaped -- quoting them through a
    filter_complex string survives neither shell nor ffmpeg reliably, and a
    typographic apostrophe renders identically."""
    out = text.replace("\\", "\\\\").replace(":", "\\:").replace("%", "\\%")
    return out.replace("'", "’")


def fill_filter(width, height, title, watermark, blur=20):
    """Blurred-fill letterbox: a zoomed, blurred copy of the frame behind the
    sharp nearest-neighbour gameplay, so the canvas is full at any aspect."""
    title_size = max(28, width // 22)
    mark_size = max(20, width // 38)
    parts = [
        "[0:v]split=2[bg][fg]",
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma={blur}[bgb]",
        f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease:"
        f"flags=neighbor[fgs]",
        "[bgb][fgs]overlay=(W-w)/2:(H-h)/2[v0]",
    ]
    if title:
        parts.append(
            f"[v0]drawtext=fontfile={FONT}:text='{esc(title)}':fontcolor=white:"
            f"fontsize={title_size}:x=(w-text_w)/2:y={height // 24}:"
            f"box=1:boxcolor=black@0.55:boxborderw=14[v1]"
        )
    else:
        parts.append("[v0]null[v1]")
    if watermark:
        parts.append(
            f"[v1]drawtext=fontfile={FONT}:text='{esc(watermark)}':"
            f"fontcolor=white@0.85:fontsize={mark_size}:x=(w-text_w)/2:"
            f"y=h-{height // 14}:box=1:boxcolor=black@0.35:boxborderw=8[vout]"
        )
    else:
        parts.append("[v1]null[vout]")
    return ";".join(parts)


def encode(src_mp4, out_path, width, height, title, watermark, crf=18):
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-i", src_mp4,
            "-filter_complex", fill_filter(width, height, title, watermark),
            "-map", "[vout]", "-map", "0:a?",
            "-r", "60", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out_path,
        ],
        check=True,
    )
    return out_path


# ----------------------------------------------------------------- events --
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

    events, frame = [], 0
    prev = {}
    while movie.step():
        keys = [movie.get_key(i, p) for p in range(movie.players)
                for i in range(env.num_buttons)]
        _obs, _r, term, trunc, info = env.step(keys)
        ram = env.get_ram()
        current = {
            "score": info.get("score"),
            "lives": info.get("lives"),
            "coins": int(ram[coin_addr]) if coin_addr else None,
            "power": int(ram[power_addr]) if power_addr else None,
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
                events.append((frame, "powerup", f"powered up (tier {now})"))
            elif key == "power" and now < was:
                events.append((frame, "powerdown", f"took a hit (tier {now})"))
            elif key == "score" and now > was:
                # info score is one tenth of the HUD value for this game
                events.append((frame, "score", f"+{(now - was) * 10} points"))
        prev = current
        frame += 1
        if term or trunc:
            break
    env.close()
    return events


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
        if kind in ("death", "powerup", "powerdown", "1up"):
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", help="stable-retro game id (see list_games.py). Required unless --paste-block or --print-upload-plan.")
    parser.add_argument("--players", type=int, choices=[1, 2], default=1, help="1 = you alone. 2 = you plus an AI player, via play_human_vs_ai. (default: %(default)s)")
    parser.add_argument("--mode", choices=["versus", "coop", "race"], default="versus", help="Two-player match type, ignored when --players 1. (default: %(default)s)")
    parser.add_argument("--model", default=None, help="Checkpoint driving the AI player when --players 2. Without one the AI plays randomly, which makes for a much weaker video.")
    parser.add_argument("--state", default=None)
    parser.add_argument("--title", default=None, help="Overlay text and metadata title. Defaults to the game id.")
    parser.add_argument("--watermark", default="", help="Handle burnt into the bottom of both videos, e.g. @yourname")
    parser.add_argument("--record-dir", default="./recordings")
    parser.add_argument("--out-dir", default="./studio_out", help="Parent for the dated review folder (default: %(default)s)")
    parser.add_argument("--from-mp4", default=None, help="Skip playing and re-cut an existing mp4 -- for redoing overlays without replaying.")
    parser.add_argument("--no-events", action="store_true", help="Skip the replay scan that builds the event timeline (faster, but narration.txt becomes generic)")
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

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is not on PATH -- see README step 1.")
    if not os.path.exists(FONT):
        sys.exit(f"Font missing: {FONT}\n  sudo apt install -y fonts-dejavu-core")

    title = args.title or args.game
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

    slug = slugify(title)
    folder = os.path.join(args.out_dir, f"{datetime.now():%Y%m%d-%H%M%S}_{slug}")
    os.makedirs(folder, exist_ok=True)

    print("\nEncoding 1920x1080 landscape master ...")
    wide = encode(native, os.path.join(folder, f"{slug}_16x9.mp4"),
                  1920, 1080, title, args.watermark)
    print("Encoding 1080x1920 vertical short ...")
    tall = encode(native, os.path.join(folder, f"{slug}_9x16.mp4"),
                  1080, 1920, title, args.watermark)

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

    with open(os.path.join(folder, "narration.txt"), "w") as handle:
        handle.write(narration(events, duration, args.game, args.players) + "\n")
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
    print(f"  {os.path.basename(tall)}   -> TikTok / Reels / Shorts")
    print(f"  metadata.json, narration.txt, events.csv ({len(events)} events)")
    print("  paste.txt        <- the three upload forms, ready to copy")
    print("\n" + block)


if __name__ == "__main__":
    main()
