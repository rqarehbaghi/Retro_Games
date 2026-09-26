#!/usr/bin/env python3
"""Turn cached progression panels into a vertical elimination film.

    python tools/progression/short.py --run progression_out/runs/<folder>

Every game remains visible in a bottom roster. The game that will finish next
is enlarged above it; when that game ends, its roster tile dims and the next
game takes the spotlight. All panels share the same source clock and the same
``video.rate_plan`` mapping, so the elimination order is the order that was
actually recorded rather than an edit assembled after the fact.

Panels need ``<key>_timeline.json`` sidecars. A panel recorded before timeline
support cannot be repaired from its MP4: replay it with progression/video.py.
The output is silent by design so licensed music can be added later.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.progression import video                              # noqa: E402

SIZE = (1080, 1920)
FPS = 60
CARD_SECONDS = 8.0
BPM = 120.0
TRANSITION = 0.5
HOLD = 3.0
CATCH_UP = 4.0
TIE_EPSILON = 1.0 / FPS


def beat_transition(seconds, bpm=BPM):
    """Quantise a transition to half-beats, never shorter than one half-beat."""
    step = 30.0 / bpm
    return max(step, round(float(seconds) / step) * step)


def checkpoint_of(panel):
    """``100k_p1_02_bare`` -> ``100k``. The cache key is the identity."""
    return panel["key"].split("_", 1)[0]


def _probe(path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return tuple(int(v) for v in probe.split(",")[:2])


def load_panels(cache, keys=None, strict=False):
    """Load cached panels with both measurement and timeline sidecars."""
    if not os.path.isdir(cache):
        raise ValueError("panel cache does not exist: %s" % cache)
    wanted = list(keys) if keys else sorted(
        name[:-5] for name in os.listdir(cache)
        if name.endswith(".json") and not name.endswith("_timeline.json"))
    out, missing = [], []
    for key in wanted:
        side = os.path.join(cache, key + ".json")
        mp4 = os.path.join(cache, key + ".mp4")
        timeline_path = os.path.join(cache, key + "_timeline.json")
        absent = [name for name, path in (("video", mp4), ("numbers", side),
                                          ("timeline", timeline_path))
                  if not os.path.exists(path)]
        if absent:
            missing.append("%s (%s)" % (key, ", ".join(absent)))
            continue
        with open(side, encoding="utf-8") as fh:
            data = json.load(fh)
        with open(timeline_path, encoding="utf-8") as fh:
            timeline = json.load(fh)
        width, height = _probe(mp4)
        panel = dict(data, key=key, path=mp4, width=width, height=height,
                     timeline=timeline, seconds=video.duration(mp4))
        stats = panel.get("stats") or []
        panel["value"] = stats[0][1] if stats else 0
        state = panel.get("state") or ""
        panel.setdefault("label", "%s%s" %
                         (checkpoint_of(panel), (" / " + state) if state else ""))
        out.append(panel)
    if missing and strict:
        raise ValueError("panels need a fresh render; missing " + "; ".join(missing))
    return out


def panels_from_run(path):
    """Resolve the exact cache entries described by a long-film run folder."""
    run_path = os.path.join(path, "run.json") if os.path.isdir(path) else path
    with open(run_path, encoding="utf-8") as fh:
        run = json.load(fh)
    cache = run.get("panels") or video.PANELS
    if not os.path.isabs(cache):
        cache = os.path.join(ROOT, cache)
    pad = int(run.get("pad", video.CROP_PAD))
    no_stats = bool(run.get("no_stats", False))
    keys, labels = [], {}
    for _checkpoint, name, label in run["columns"]:
        for state in run["states"]:
            key = video.panel_key(name, state, no_stats, pad)
            keys.append(key)
            labels[key] = "%s / %s" % (label, state)
    panels = load_panels(cache, keys, strict=True)
    for panel in panels:
        panel["label"] = labels[panel["key"]]
    chart = os.path.join(os.path.dirname(run_path), "chart.png")
    return panels, run, chart if os.path.exists(chart) else None


def ending(panel):
    """Validated terminal event for one panel.

    A cap and a truncation are deliberately returned as themselves. Calling
    either one a death would turn a censored trajectory into a false winner.
    """
    timeline = panel.get("timeline") or []
    if not timeline:
        raise ValueError("%s has no timeline" % panel["key"])
    times = [float(event.get("t", -1)) for event in timeline]
    if any(t < 0 for t in times) or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("%s has a non-monotonic timeline" % panel["key"])
    final = timeline[-1]
    reason = final.get("end") or panel.get("end_reason")
    if not reason:
        raise ValueError("%s timeline has no terminal reason" % panel["key"])
    side_reason = panel.get("end_reason")
    if side_reason and side_reason != reason:
        raise ValueError("%s disagrees about its ending (%s vs %s)"
                         % (panel["key"], reason, side_reason))
    return {"key": panel["key"], "time": times[-1], "reason": reason,
            "visual_end": min(panel["seconds"], times[-1] + video.END_FRAMES / FPS)}


def _groups(endings):
    """Group true deaths by time and keep all capped survivors together.

    Placement caps censor a game; their wall-clock ordering is not an
    elimination order.  Every capped panel therefore occupies the final tied
    group even if one emulator reached the common decision cap sooner.
    """
    incomplete = [item["key"] for item in endings if item["reason"] == "truncated"]
    if incomplete:
        raise ValueError("truncated panels cannot establish an elimination order: %s"
                         % ", ".join(incomplete))
    deaths = [item for item in endings if item["reason"] == "game_over"]
    censored = [item for item in endings if item["reason"] == "placement_cap"]
    unknown = [item for item in endings
               if item["reason"] not in ("game_over", "placement_cap", "truncated")]
    if unknown:
        raise ValueError("unknown terminal reason on: %s"
                         % ", ".join(item["key"] for item in unknown))
    groups = []
    for item in sorted(deaths, key=lambda e: (e["time"], e["key"])):
        if groups and abs(item["time"] - groups[-1][0]["time"]) <= TIE_EPSILON:
            groups[-1].append(item)
        else:
            groups.append([item])
    if censored:
        groups.append(sorted(censored, key=lambda e: e["key"]))
    return groups


def _outcome(group):
    reasons = {item["reason"] for item in group}
    if len(group) > 1:
        if reasons == {"placement_cap"}:
            return "TIED AT CAP"
        return "TIED SURVIVORS"
    reason = group[0]["reason"]
    if reason == "placement_cap":
        return "SURVIVED TO CAP"
    if reason == "truncated":
        return "RUN TRUNCATED"
    return "LAST SURVIVOR"


def plan(panels, seconds=60.0, card_seconds=CARD_SECONDS, bpm=BPM,
         catch_up=CATCH_UP, hold=HOLD, transition=TRANSITION):
    """Build elimination phases on one shared source/output clock."""
    if not panels:
        raise ValueError("an elimination film needs at least one panel")
    ends = [ending(panel) for panel in panels]
    groups = _groups(ends)
    body_target = max(2.0, float(seconds) - float(card_seconds))
    hold = min(float(hold), max(0.0, body_target - 0.5))
    unit = video.rate_plan([e["time"] for e in ends], 1.0, catch_up, 0.0)[1]
    speed = max(0.1, unit / max(0.5, body_target - hold))
    rates, body_seconds = video.rate_plan(
        [e["time"] for e in ends], speed, catch_up, hold)
    transition = beat_transition(transition, bpm)

    phases, previous = [], 0.0
    for index, group in enumerate(groups):
        start = video.output_time(rates, previous)
        # A capped final group may contain panels that reached the common cap
        # before the preceding true death. It still belongs last because it
        # was censored, so never move the shared source clock backwards.
        terminal = max(previous, max(item["time"] for item in group))
        end = (body_seconds if index == len(groups) - 1
               else video.output_time(rates, terminal))
        reasons = sorted({item["reason"] for item in group})
        phases.append({"keys": [item["key"] for item in group],
                       "source_start": previous, "source_end": terminal,
                       "start": start, "end": end, "show_start": start,
                       "show_end": end, "fade_in": 0.0, "fade_out": 0.0,
                       "reasons": reasons})
        previous = terminal

    # Crossfades straddle the real elimination boundary. Very short phases
    # receive a proportionally shorter dissolve rather than disappearing.
    for index in range(len(phases) - 1):
        left, right = phases[index], phases[index + 1]
        boundary = left["end"]
        dissolve = min(transition, (left["end"] - left["start"]) / 2.0,
                       (right["end"] - right["start"]) / 2.0)
        dissolve = max(0.0, dissolve)
        left["show_end"] = boundary + dissolve / 2.0
        left["fade_out"] = dissolve
        right["show_start"] = boundary - dissolve / 2.0
        right["fade_in"] = dissolve

    final = groups[-1]
    return {"phases": phases, "endings": ends, "rates": rates,
            "body_seconds": body_seconds, "card_seconds": float(card_seconds),
            "seconds": body_seconds + float(card_seconds), "speed": speed,
            "fastest": max(rate for _end, rate in rates), "catch_up": catch_up,
            "hold": hold, "transition": transition, "bpm": bpm,
            "outcome": _outcome(final),
            "survivors": [item["key"] for item in final]}


def _roster_layout(panels, size):
    """A one- or two-row bottom dock; two rows keep 15 boards legible."""
    width, height = size
    count = len(panels)
    rows = 1 if count <= 6 else 2
    cols = int(math.ceil(count / float(rows)))
    margin, gap, label_h = 22, 10, 24
    dock_top, dock_bottom = (height - (300 if rows == 1 else 480), height - 68)
    cell_w = (width - 2 * margin - (cols - 1) * gap) / float(cols)
    cell_h = (dock_bottom - dock_top - (rows - 1) * gap) / float(rows)
    result = {}
    for index, panel in enumerate(panels):
        row, col = divmod(index, cols)
        scale = min((cell_w - 4) / panel["width"],
                    (cell_h - label_h - 4) / panel["height"])
        pw = max(2, int(panel["width"] * scale) // 2 * 2)
        ph = max(2, int(panel["height"] * scale) // 2 * 2)
        cell_x = margin + col * (cell_w + gap)
        cell_y = dock_top + row * (cell_h + gap)
        result[panel["key"]] = {
            "x": int(cell_x + (cell_w - pw) / 2), "y": int(cell_y + label_h),
            "w": pw, "h": ph, "label_x": int(cell_x + cell_w / 2),
            "label_y": int(cell_y + 2), "label_width": int(cell_w)}
    return result, dock_top


def _spot_layout(keys, by_key, size, dock_top):
    """Place one spotlight—or a genuine tied group—in the main stage."""
    width, _height = size
    left, top, right, bottom = 48, 120, width - 48, dock_top - 42
    count = len(keys)
    first = by_key[keys[0]]
    cols = video.best_grid(count, first["width"], first["height"],
                           right - left, bottom - top, gutter=26)
    rows = int(math.ceil(count / float(cols)))
    cell_w = (right - left - (cols - 1) * 26) / float(cols)
    cell_h = (bottom - top - (rows - 1) * 26) / float(rows)
    out = {}
    for index, key in enumerate(keys):
        panel = by_key[key]
        scale = min(cell_w / panel["width"], cell_h / panel["height"])
        pw = max(2, int(panel["width"] * scale) // 2 * 2)
        ph = max(2, int(panel["height"] * scale) // 2 * 2)
        row, col = divmod(index, cols)
        cx = left + col * (cell_w + 26) + cell_w / 2
        cy = top + row * (cell_h + 26) + cell_h / 2
        out[key] = {"x": int(cx - pw / 2), "y": int(cy - ph / 2),
                    "w": pw, "h": ph}
    return out


def _escape_text(text):
    return str(text).replace("\\", "\\\\").replace(":", "\\:").replace("'", "")


def _enabled_label(chain, output, text, start, end, y, size):
    font = video.font_file()
    font_arg = ("fontfile=%s:" % font) if font else ""
    return ("%sdrawtext=%stext='%s':fontcolor=black:fontsize=%d:box=1:"
            "boxcolor=white:boxborderw=10:x=(w-text_w)/2:y=%d:"
            "enable='between(t\\,%.3f\\,%.3f)'%s"
            % (chain, font_arg, _escape_text(text), size, y, start, end, output))


def _phase_heading(phase, final, by_key):
    names = " + ".join(by_key[key]["label"] for key in phase["keys"])
    if final:
        return "%s: %s" % (final, names)
    reasons = set(phase["reasons"])
    if "placement_cap" in reasons:
        return "CAP REACHED: %s" % names
    if "truncated" in reasons:
        return "RUN ENDS: %s" % names
    return ("NEXT OUT (TIE): %s" if len(phase["keys"]) > 1
            else "NEXT OUT: %s") % names


def _progress(start, duration):
    """An ffmpeg expression that rises from zero to one over an interval."""
    return "min(1\\,max(0\\,(t-%.3f)/%.3f))" % (start, max(0.001, duration))


def _scale_filter(tile, spot, phase, grow):
    if not grow or phase["fade_in"] <= 0:
        return "scale=%d:%d:flags=neighbor" % (spot["w"], spot["h"])
    progress = _progress(phase["show_start"], phase["fade_in"])
    width = "trunc((%d+(%d-%d)*%s)/2)*2" % (
        tile["w"], spot["w"], tile["w"], progress)
    height = "trunc((%d+(%d-%d)*%s)/2)*2" % (
        tile["h"], spot["h"], tile["h"], progress)
    return "scale=w='%s':h='%s':flags=neighbor:eval=frame" % (width, height)


def _overlay_position(tile, spot, phase, grow):
    if not grow or phase["fade_in"] <= 0:
        return "%d:%d" % (spot["x"], spot["y"])
    progress = _progress(phase["show_start"], phase["fade_in"])
    x = "%d+(%d-%d)*%s" % (tile["x"], spot["x"], tile["x"], progress)
    y = "%d+(%d-%d)*%s" % (tile["y"], spot["y"], tile["y"], progress)
    return "'%s':'%s':eval=frame" % (x, y)


def render_body(plan_data, panels, path, size=SIZE, handle="", grow=True):
    """Render roster and spotlight branches in one ffmpeg graph."""
    width, height = size
    by_key = {panel["key"]: panel for panel in panels}
    roster, dock_top = _roster_layout(panels, size)
    phase_for = {key: phase for phase in plan_data["phases"]
                 for key in phase["keys"]}
    spots = {key: place for phase in plan_data["phases"]
             for key, place in _spot_layout(phase["keys"], by_key, size,
                                            dock_top).items()}
    max_source = max(end for end, _rate in plan_data["rates"])
    remap = video.remap(plan_data["rates"])
    inputs, filters = [], []
    final_keys = set(plan_data["survivors"])
    endings = {item["key"]: item for item in plan_data["endings"]}

    for index, panel in enumerate(panels):
        key = panel["key"]
        inputs += ["-i", panel["path"]]
        pad = max(0.0, max_source - panel["seconds"])
        filters.append("[%d:v]tpad=stop_mode=clone:stop_duration=%.3f,split=2"
                       "[tile_raw%d][spot_raw%d]" % (index, pad, index, index))
        tile = roster[key]
        dim = ("eq=brightness=-0.30:saturation=0.38:enable='gte(t\\,%.3f)',"
               % endings[key]["visual_end"]) if key not in final_keys else ""
        filters.append("[tile_raw%d]%s%s,scale=%d:%d:flags=neighbor,setsar=1[tile%d]"
                       % (index, dim, remap, tile["w"], tile["h"], index))

        phase, spot = phase_for[key], spots[key]
        branch = ("[spot_raw%d]%s,%s,setsar=1,format=yuva420p"
                  % (index, remap, _scale_filter(tile, spot, phase, grow)))
        if phase["fade_in"] > 0:
            branch += ",fade=t=in:st=%.3f:d=%.3f:alpha=1" % (
                phase["show_start"], phase["fade_in"])
        if phase["fade_out"] > 0:
            branch += ",fade=t=out:st=%.3f:d=%.3f:alpha=1" % (
                phase["end"] - phase["fade_out"] / 2.0, phase["fade_out"])
        filters.append(branch + "[spot%d]" % index)

    filters.append("color=c=#090b10:s=%dx%d:r=%d:d=%.3f[bg]"
                   % (width, height, FPS, plan_data["body_seconds"]))
    stage = "[bg]"
    for index, panel in enumerate(panels):
        tile = roster[panel["key"]]
        output = "[r%d]" % index
        filters.append("%s[tile%d]overlay=%d:%d%s"
                       % (stage, index, tile["x"], tile["y"], output))
        stage = output
    for index, panel in enumerate(panels):
        key, phase, spot = panel["key"], phase_for[panel["key"]], spots[panel["key"]]
        output = "[s%d]" % index
        filters.append("%s[spot%d]overlay=%s:enable='between(t\\,%.3f\\,%.3f)'%s"
                       % (stage, index, _overlay_position(roster[key], spot, phase,
                                                         grow),
                          phase["show_start"], phase["show_end"], output))
        stage = output

    for index, phase in enumerate(plan_data["phases"]):
        output = "[d%d]" % index
        final = plan_data["outcome"] if index == len(plan_data["phases"]) - 1 else ""
        heading = _phase_heading(phase, final, by_key)
        # The pictures overlap during a dissolve, but their claims must not:
        # keep NEXT OUT until the real death boundary, then name the survivor.
        filters.append(_enabled_label(stage, output, heading, phase["start"],
                                      phase["end"], 50,
                                      video.fit_size(heading, width - 60, 34, 14)))
        stage = output

    static = []
    for panel in panels:
        tile = roster[panel["key"]]
        static.append((panel["label"], tile["label_x"], tile["label_y"],
                       video.fit_size(panel["label"], tile["label_width"], 15, 8)))
    speed = ("%sx speed" % video.rate_text(plan_data["speed"]) if
             plan_data["fastest"] <= plan_data["speed"] * 1.01 else
             "%sx speed, up to %sx as games end" %
             (video.rate_text(plan_data["speed"]), video.rate_text(plan_data["fastest"])))
    static.append((speed, width // 2, height - 28, 14))
    if handle:
        static.append((handle, "w-text_w-22", height - 55, 15))
    filters.append(video.drawtext(stage, static))

    cmd = (["ffmpeg", "-nostdin", "-y", "-v", "error"] + inputs +
           ["-filter_complex", ";".join(filters), "-map", "[final]", "-an",
            "-t", "%.3f" % plan_data["body_seconds"], "-r", str(FPS),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", path])
    subprocess.run(cmd, check=True)
    return path


def _plate_center(draw, canvas_width, y, text, font, pad=10):
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    video.plate(draw, ((canvas_width - width) / 2, y), text, font, pad=pad)


def build_crown_card(plan_data, panels, path, chart=None, size=SIZE):
    """Create an honest winner/survivor card, optionally carrying the chart."""
    width, height = size
    image = Image.new("RGB", size, (9, 11, 16))
    draw = ImageDraw.Draw(image)
    title_font = video.house_font(44)
    name_font = video.house_font(30)
    body_font = video.house_font(21)
    by_key = {panel["key"]: panel for panel in panels}
    survivors = [by_key[key] for key in plan_data["survivors"]]
    _plate_center(draw, width, 110, plan_data["outcome"], title_font, 14)
    y = 245
    for panel in survivors:
        _plate_center(draw, width, y, panel["label"], name_font, 10)
        y += 72
        stats = "   ".join("%s %s" % (name, value)
                           for name, value in panel.get("stats") or [])
        if stats:
            _plate_center(draw, width, y, stats, body_font, 8)
            y += 58
    if len(survivors) == 1:
        _plate_center(draw, width, y + 15,
                      "OUTLASTED %d OTHER GAME%s" %
                      (len(panels) - 1, "" if len(panels) == 2 else "S"),
                      body_font, 8)
    if chart and os.path.exists(chart):
        card = Image.open(chart).convert("RGB")
        card.thumbnail((width - 60, max(300, height - 760)), Image.Resampling.LANCZOS)
        image.paste(card, ((width - card.width) // 2, height - card.height - 100))
    image.save(path)
    return path


def _render_card(image, seconds, path, size=SIZE):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-loop", "1",
         "-t", "%.3f" % seconds, "-i", image,
         "-vf", "scale=%d:%d:force_original_aspect_ratio=decrease,"
                "pad=%d:%d:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=%d,"
                "fade=in:st=0:d=0.4" % (size[0], size[1], size[0], size[1], FPS),
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", path], check=True)


def render(plan_data, panels, out_path, chart=None, size=SIZE, handle="", work=None,
           grow=True):
    """Render the elimination body and its crown card, then join them losslessly."""
    work = work or os.path.join(os.path.dirname(out_path), "elimination_work")
    os.makedirs(work, exist_ok=True)
    body = render_body(plan_data, panels, os.path.join(work, "body.mp4"), size,
                       handle, grow=grow)
    if plan_data["card_seconds"] <= 0:
        shutil.copyfile(body, out_path)
        return out_path
    card_png = build_crown_card(plan_data, panels, os.path.join(work, "crown.png"),
                                chart=chart, size=size)
    card_mp4 = os.path.join(work, "crown.mp4")
    _render_card(card_png, plan_data["card_seconds"], card_mp4, size)
    listing = os.path.join(work, "parts.txt")
    with open(listing, "w", encoding="utf-8") as fh:
        for part in (body, card_mp4):
            fh.write("file '%s'\n" % os.path.abspath(part).replace("'", "'\\''"))
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "concat",
                    "-safe", "0", "-i", listing, "-c", "copy",
                    "-movflags", "+faststart", out_path], check=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None,
                        help="long-film run folder (or its run.json); resolves exact panels")
    parser.add_argument("--panels", default=None,
                        help="panel cache when --run is not used")
    parser.add_argument("--games", default=None,
                        help="panel keys, comma separated, in roster order")
    parser.add_argument("--count", type=int, default=0,
                        help="use only the first N loaded panels (default: all)")
    parser.add_argument("--card", default=None,
                        help="optional chart placed beneath the crown on the final card")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="total target length including the card")
    parser.add_argument("--card-seconds", type=float, default=CARD_SECONDS)
    parser.add_argument("--catch-up", type=float, default=None,
                        help="maximum acceleration as games end")
    parser.add_argument("--hold", type=float, default=None,
                        help="real-time hold on the final ending before the card")
    parser.add_argument("--transition", type=float, default=TRANSITION,
                        help="crossfade length, quantised to half-beats")
    parser.add_argument("--no-grow", action="store_true",
                        help="crossfade only; do not grow the next roster tile")
    parser.add_argument("--bpm", type=float, default=BPM)
    parser.add_argument("--handle", default=None,
                        help="default: run handle, then studio.json watermark")
    parser.add_argument("--out", default=os.path.join(ROOT, "progression_out",
                                                       "short_9x16.mp4"))
    args = parser.parse_args()

    run = {}
    inferred_card = None
    try:
        if args.run:
            panels, run, inferred_card = panels_from_run(args.run)
        else:
            cache = args.panels or video.PANELS
            keys = [key.strip() for key in args.games.split(",")] if args.games else None
            panels = load_panels(cache, keys, strict=True)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.games and args.run:
        keys = [key.strip() for key in args.games.split(",")]
        by_key = {panel["key"]: panel for panel in panels}
        missing = [key for key in keys if key not in by_key]
        if missing:
            raise SystemExit("these games are not in the run: %s" % ", ".join(missing))
        panels = [by_key[key] for key in keys]
    if args.count > 0:
        panels = panels[:args.count]
    if not panels:
        raise SystemExit("no panels with timelines were found; render them fresh first")

    catch_up = args.catch_up if args.catch_up is not None else float(run.get("catch_up", CATCH_UP))
    hold = args.hold if args.hold is not None else float(run.get("hold", HOLD))
    handle = args.handle if args.handle is not None else run.get("handle")
    if handle is None:
        with open(os.path.join(ROOT, "studio.json"), encoding="utf-8") as fh:
            handle = json.load(fh).get("watermark", "")
    card = args.card or inferred_card

    try:
        plan_data = plan(panels, args.seconds, args.card_seconds, args.bpm,
                         catch_up, hold, args.transition)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    plan_path = os.path.join(out_dir, "elimination.json")
    with open(plan_path, "w", encoding="utf-8") as fh:
        json.dump(plan_data, fh, indent=2)
    render(plan_data, panels, args.out, chart=card, handle=handle,
           work=os.path.join(out_dir, "elimination_work"), grow=not args.no_grow)
    print("%d games, %d elimination phases, %s" %
          (len(panels), len(plan_data["phases"]), plan_data["outcome"].lower()))
    print("wrote %s" % plan_path)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
