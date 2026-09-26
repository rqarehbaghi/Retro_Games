#!/usr/bin/env python3
"""Cut a vertical short out of games that have already been played.

    python tools/progression/short.py --run progression_out/runs/<folder>

Not a faster grid. A GRID compresses one length into another and shows
everything at once; this is an edit -- a handful of moments, chosen, in an
order, with panels arriving and leaving. The long version is the record; this
is the trailer for it.

WHAT IT IS CUT FROM. Every panel carries a timeline: one entry per placement,
with the time in that panel's video, what the placement cleared, the stack
height and the level. That is what makes a cut land on a tetris instead of on
a stopwatch, and it is why this tool cannot run on panels rendered before
timelines existed.

THE GRAMMAR, which is not invented here -- it is how trailers are cut:

    hook          the strongest image first, in the first 3 seconds, because
                  that is how long a viewer gives it
    setup         what this is, who is playing
    turn          something changes -- here, a board tops out and is replaced
    escalation    shot lengths shorten as the stakes rise
    climax        the best moments, cut tight
    button        one last beat: the card, the crown, the challenge

Shot lengths ramp long to short across that arc, and every cut is quantised to
a beat grid so a track dropped on afterwards lands on the cuts. The file ships
SILENT for exactly that reason -- there is no music here to fight with, and
none that could be chosen without a licence.
"""
import argparse
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.progression import video                              # noqa: E402

SIZE = (1080, 1920)
FPS = 60
CARD_SECONDS = 8.0
BPM = 120.0
HOOK = 3.0                       # a viewer decides in about this long
LONG_SHOT, SHORT_SHOT = 3.0, 1.0  # the ramp, in seconds
GUTTER = 44


def beat(seconds, bpm=BPM):
    """Snap to the beat grid, so whatever track goes on top lands on the cuts."""
    step = 60.0 / bpm
    return max(step, round(seconds / step) * step)


def load_panels(cache, keys=None):
    """Every panel in the cache that has numbers AND a timeline."""
    out = []
    for side in sorted(f for f in os.listdir(cache) if f.endswith(".json")
                       and not f.endswith("_timeline.json")):
        key = side[:-5]
        if keys and key not in keys:
            continue
        mp4 = os.path.join(cache, key + ".mp4")
        tl = os.path.join(cache, key + "_timeline.json")
        if not (os.path.exists(mp4) and os.path.exists(tl)):
            continue
        data = json.load(open(os.path.join(cache, side)))
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
             "stream=width,height", "-of", "csv=p=0", mp4],
            capture_output=True, text=True, check=True).stdout.strip()
        w, h = (int(v) for v in probe.split(",")[:2])
        out.append(dict(data, key=key, path=mp4, width=w, height=h,
                        timeline=json.load(open(tl)),
                        seconds=video.duration(mp4),
                        value=(data.get("stats") or [["", 0]])[0][1]))
    return out


def checkpoint_of(panel):
    """'100k_p1_02_bare' -> '100k'. The cache key IS the identity."""
    return panel["key"].split("_", 1)[0]


def pick_games(panels, count=2):
    """The hero and its foil.

    Hybrid, and deliberately so. The best game of the best checkpoint is the
    footage worth watching; an early checkpoint beside it is what makes the
    point of the channel in the first ten seconds. Showing only the winner is
    prettier and says nothing; showing the whole ladder means most of the
    screen is dead boards."""
    if not panels:
        return []
    by_ckpt = {}
    for p in panels:
        by_ckpt.setdefault(checkpoint_of(p), []).append(p)

    def mean(group):
        return sum(g["value"] for g in group) / float(len(group))

    order = sorted(by_ckpt, key=lambda c: mean(by_ckpt[c]))
    hero = max(by_ckpt[order[-1]], key=lambda p: p["value"])
    picked = [hero]
    for name in order[:-1]:                      # weakest first
        if len(picked) >= count:
            break
        picked.append(max(by_ckpt[name], key=lambda p: p["value"]))
    return picked


def moments(panel, top=8):
    """The bits worth cutting to, best first.

    Scored on what the timeline actually recorded: a big clear is worth more
    than a small one, a tall stack is a near-death and reads as danger, and a
    higher level means the piece is falling faster, which is visible."""
    scored = []
    for e in panel["timeline"]:
        if e.get("end"):
            continue
        cleared, height, level = e.get("d") or 0, e.get("h") or 0, e.get("lvl") or 0
        score = cleared * cleared * 10 + max(0, height - 8) * 2 + level
        if score > 0:
            scored.append((score, e["t"], e))
    scored.sort(key=lambda s: (-s[0], s[1]))
    keep, used = [], []
    for score, t, e in scored:
        # Never two cuts from the same few seconds: a montage of one moment
        # from six angles is what happens when nothing spreads them out.
        if any(abs(t - u) < 4.0 for u in used):
            continue
        keep.append({"t": t, "score": score, "cleared": e.get("d") or 0,
                     "level": e.get("lvl") or 0})
        used.append(t)
        if len(keep) >= top:
            break
    return keep


def plan(panels, seconds, card_seconds=CARD_SECONDS, bpm=BPM):
    """The shot list: what is on screen, from where, for how long.

    Two panels while the foil is alive, one once it dies -- that replacement
    IS the turn, and it lands about a third of the way in whatever the footage
    does, because a trailer's shape does not wait for the material."""
    hero = panels[0]
    foil = panels[1] if len(panels) > 1 else None
    body = max(4.0, seconds - card_seconds)
    turn_at = beat(body * 0.35, bpm)
    beats = moments(hero)

    shots, t = [], 0.0
    # HOOK: the single best moment of the best game, before anything is
    # explained. Explaining first is how a short loses half its viewers.
    best = beats[0] if beats else {"t": hero["seconds"] * 0.6}
    shots.append({"sources": [(hero, max(0.0, best["t"] - HOOK + 0.6))],
                  "length": beat(HOOK, bpm), "text": None, "speed": 1.0})
    t += shots[-1]["length"]

    # SETUP: both boards, from their openings, labelled.
    while t < turn_at and foil:
        length = beat(LONG_SHOT, bpm)
        shots.append({"sources": [(hero, t * 1.2), (foil, t * 1.2)],
                      "length": length, "text": None, "speed": 2.0})
        t += length

    # ESCALATION and CLIMAX: the hero alone, shots shortening, each landing on
    # a moment the timeline actually recorded.
    remaining = max(0.0, body - t)
    picks = beats[1:] or beats
    n = max(1, int(remaining / ((LONG_SHOT + SHORT_SHOT) / 2)))
    for i in range(n):
        if t >= body:
            break
        ramp = LONG_SHOT + (SHORT_SHOT - LONG_SHOT) * (i / float(max(1, n - 1)))
        length = beat(min(ramp, body - t), bpm)
        pick = picks[i % len(picks)] if picks else {"t": 0.0}
        shots.append({"sources": [(hero, max(0.0, pick["t"] - length * 0.65))],
                      "length": length, "text": None, "speed": 1.0})
        t += length
    return {"shots": shots, "turn_at": turn_at, "card_seconds": card_seconds,
            "seconds": t + card_seconds, "bpm": bpm,
            "hero": hero["key"], "foil": foil["key"] if foil else None}


def render_shot(shot, out_path, size=SIZE, handle="", text=None):
    """One shot: one or two live panels on a blurred fill, with its caption.

    Rendered on its own rather than as part of one enormous filtergraph. A
    sixty second cut is twenty-odd shots, each with its own sources, offsets
    and speed; as a single graph that is unreadable and fails at run time with
    nothing to point at. Separate files concatenate cleanly and a bad shot can
    be looked at on its own."""
    W, H = size
    sources = shot["sources"]
    panel = sources[0][0]
    cols = video.best_grid(len(sources), panel["width"], panel["height"],
                           W, H - 150, GUTTER)
    rows = int(math.ceil(len(sources) / float(cols)))
    cell_w = (W - (cols - 1) * GUTTER) / float(cols)
    cell_h = (H - 150 - (rows - 1) * GUTTER) / float(rows)
    scale = min(cell_w / panel["width"], cell_h / panel["height"])
    pw = int(panel["width"] * scale) // 2 * 2
    ph = int(panel["height"] * scale) // 2 * 2
    x0 = (W - (cols * pw + (cols - 1) * GUTTER)) // 2
    y0 = 96 + (H - 150 - (rows * ph + (rows - 1) * GUTTER)) // 2

    cmd, chains = ["ffmpeg", "-nostdin", "-y", "-v", "error"], []
    for i, (src, start) in enumerate(sources):
        # Input seek, so a cut forty minutes into a game costs nothing. tpad
        # holds the last frame if the clip would run off the end of that game.
        cmd += ["-ss", "%.3f" % max(0.0, min(start, src["seconds"] - 0.2)),
                "-i", src["path"]]
        chains.append("[%d:v]setpts=PTS-STARTPTS,setpts=PTS/%.3f,"
                      "tpad=stop_mode=clone:stop_duration=%.3f,"
                      "scale=%d:%d:flags=neighbor,setsar=1[p%d]"
                      % (i, shot.get("speed", 1.0), shot["length"], pw, ph, i))
    small_w, small_h = max(2, W // 8), max(2, H // 8)
    chains.append("[0:v]setpts=PTS-STARTPTS,setpts=PTS/%.3f,"
                  "tpad=stop_mode=clone:stop_duration=%.3f,"
                  "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,"
                  "gblur=sigma=2.5,eq=brightness=-0.2:saturation=0.6,"
                  "scale=%d:%d:flags=bicubic,setsar=1[bg]"
                  % (shot.get("speed", 1.0), shot["length"],
                     small_w, small_h, small_w, small_h, W, H))
    stage = "[bg]"
    for i in range(len(sources)):
        cx = x0 + (i % cols) * (pw + GUTTER)
        cy = y0 + (i // cols) * (ph + GUTTER)
        nxt = "[v%d]" % i
        chains.append("%s[p%d]overlay=%d:%d%s" % (stage, i, cx, cy, nxt))
        stage = nxt

    labels = []
    for i, (src, _start) in enumerate(sources):
        name = src.get("label") or checkpoint_of(src)
        labels.append((name, x0 + (i % cols) * (pw + GUTTER) + pw // 2,
                       y0 + (i // cols) * (ph + GUTTER) - 44,
                       video.fit_size(name, pw + GUTTER, 34)))
    if text:
        labels.append((text, W // 2, 40, video.fit_size(text, W - 80, 44)))
    if handle:
        labels.append((handle, "w-text_w-26", H - 44, 20))
    graph = ";".join(chains) + ";" + video.drawtext(stage, labels)

    cmd += ["-filter_complex", graph, "-map", "[final]", "-an",
            "-t", "%.3f" % shot["length"], "-r", str(FPS),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, check=True)
    return out_path


def render(plan_data, panels, out_path, card=None, size=SIZE, handle="",
           work=None, texts=None):
    """Every shot, then the card, joined hard."""
    work = work or os.path.join(os.path.dirname(out_path), "shots")
    os.makedirs(work, exist_ok=True)
    by_key = {p["key"]: p for p in panels}
    parts = []
    for i, shot in enumerate(plan_data["shots"]):
        shot = dict(shot, sources=[(by_key[k] if isinstance(k, str) else k, t)
                                   for k, t in shot["sources"]])
        parts.append(render_shot(shot, os.path.join(work, "shot_%02d.mp4" % i),
                                 size=size, handle=handle,
                                 text=(texts or {}).get(str(i))))
    if card:
        tail = os.path.join(work, "card.mp4")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-loop", "1",
             "-t", "%.3f" % plan_data["card_seconds"], "-i", card,
             "-vf", "scale=%d:%d:force_original_aspect_ratio=decrease,"
                    "pad=%d:%d:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=%d,"
                    "fade=in:st=0:d=0.4" % (size[0], size[1], size[0], size[1], FPS),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-pix_fmt", "yuv420p", tail], check=True)
        parts.append(tail)

    listing = os.path.join(work, "parts.txt")
    with open(listing, "w") as fh:
        for part in parts:
            # SINGLE quotes, with any of its own escaped. The concat demuxer
            # does not understand double quotes -- it takes them as part of
            # the filename and reports the file as missing, quotes included.
            fh.write("file '%s'\n" % os.path.abspath(part).replace("'", "'\\''"))
    # Hard cuts, so the concat demuxer is enough and every cut lands exactly
    # on the beat it was planned for. A crossfade would move them.
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "concat",
                    "-safe", "0", "-i", listing, "-c", "copy",
                    "-movflags", "+faststart", out_path], check=True)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panels", default=os.path.join(ROOT, "progression_out", "panels"),
                   help="the panel cache to cut from (default %(default)s)")
    p.add_argument("--card", default=None, help="the stats card PNG to end on")
    p.add_argument("--games", default=None,
                   help="which games, by panel key, comma separated and HERO FIRST "
                        "(e.g. 200k_level0_1p_bare,10k_p1_01_bare). The default picks "
                        "the best checkpoint's best game and an early checkpoint's")
    p.add_argument("--count", type=int, default=2,
                   help="how many games to draw on when none are named")
    p.add_argument("--seconds", type=float, default=60.0, help="total length")
    p.add_argument("--card-seconds", type=float, default=CARD_SECONDS)
    p.add_argument("--bpm", type=float, default=BPM,
                   help="cuts are quantised to this, so a track at this tempo lands "
                        "on every one of them")
    p.add_argument("--handle", default=None, help="default: studio.json's watermark")
    p.add_argument("--texts", default=None,
                   help="JSON file of {shot index: caption}, to letter the cut")
    p.add_argument("--out", default=os.path.join(ROOT, "progression_out", "short_9x16.mp4"))
    a = p.parse_args()

    keys = [k.strip() for k in a.games.split(",")] if a.games else None
    panels = load_panels(a.panels, keys)
    if not panels:
        sys.exit("no panels with timelines in %s. Panels rendered before timelines "
                 "existed cannot be cut from -- re-render them." % a.panels)
    if keys:
        panels.sort(key=lambda p: keys.index(p["key"]))
    else:
        panels = pick_games(panels, a.count)
    for panel in panels:
        panel["label"] = checkpoint_of(panel)

    shots = plan(panels, a.seconds, a.card_seconds, a.bpm)
    handle = a.handle
    if handle is None:
        handle = json.load(open(os.path.join(ROOT, "studio.json"))).get("watermark", "")

    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    # The edit as data, before it is pixels: every source, offset and length.
    # beats.json is the same cut times for whatever lays music over it.
    json.dump({"shots": [dict(s, sources=[[src["key"], round(t, 3)]
                                          for src, t in s["sources"]])
                         for s in shots["shots"]],
               "hero": shots["hero"], "foil": shots["foil"],
               "bpm": shots["bpm"], "card_seconds": shots["card_seconds"]},
              open(os.path.join(out_dir, "shots.json"), "w"), indent=2)
    cuts, at = [], 0.0
    for s in shots["shots"]:
        cuts.append(round(at, 3))
        at += s["length"]
    json.dump({"bpm": shots["bpm"], "cuts": cuts, "card_at": round(at, 3),
               "seconds": round(at + shots["card_seconds"], 3)},
              open(os.path.join(out_dir, "beats.json"), "w"), indent=2)

    texts = json.load(open(a.texts)) if a.texts else None
    render(shots, panels, a.out, card=a.card, handle=handle,
           work=os.path.join(out_dir, "shots"), texts=texts)
    print("hero %s, foil %s, %d shots, %.1fs + %.0fs card"
          % (shots["hero"], shots["foil"], len(shots["shots"]), at,
             shots["card_seconds"]))
    for name in ("shots.json", "beats.json"):
        print("wrote %s" % os.path.join(out_dir, name))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
