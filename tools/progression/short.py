#!/usr/bin/env python3
"""Turn a progression run into a concise vertical checkpoint race.

    python tools/progression/short.py --run progression_out/runs/<folder>

Exactly ONE game represents each checkpoint: the run with the highest primary
reported metric, with decisions and duration as deterministic tie-breakers.
Every champion remains visible in a bottom roster while the next one to finish
is enlarged above it. The film never exceeds 10x. When the source is too long
to fit, representative source windows are joined with honest jump cuts and a
source-clock timer visibly jumps with them.

Panels need ``<key>_timeline.json`` sidecars. A panel recorded before timeline
support cannot be repaired from its MP4: replay it with progression/video.py.
The output is silent by design. ChatGPT writes sparse on-screen commentary, a
final challenge from the winning AI to its trainer, and upload copy for the
long YouTube film, YouTube Short, Instagram and TikTok. Every model prompt is
saved beside the finished package.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.progression import video                              # noqa: E402
import writer as text_writer                                     # noqa: E402

SIZE = (1080, 1920)
FPS = 60
CARD_SECONDS = 6.0
MAX_SPEED = 10.0
CLIP_SOURCE_SECONDS = 65.0
MIN_PHASE_SOURCE_SECONDS = 35.0
TIE_EPSILON = 1.0 / FPS
OVERLAY_FONT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "fonts", "BlackOpsOne-Regular.ttf")


PLATFORM_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "description", "tags"],
    "additionalProperties": False,
}
SHORT_COPY_SCHEMA = {
    "type": "object",
    "properties": {
        "overlays": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cut": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["cut", "text"],
                "additionalProperties": False,
            },
        },
        "challenge": {"type": "string"},
        "youtube_long": PLATFORM_SCHEMA,
        "youtube_short": PLATFORM_SCHEMA,
        "instagram": PLATFORM_SCHEMA,
        "tiktok": PLATFORM_SCHEMA,
    },
    "required": ["overlays", "challenge", "youtube_long", "youtube_short",
                 "instagram", "tiktok"],
    "additionalProperties": False,
}


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
            labels[key] = label
    panels = load_panels(cache, keys, strict=True)
    for panel in panels:
        panel["label"] = labels[panel["key"]]
        panel["checkpoint"] = checkpoint_of(panel)
    chart = os.path.join(os.path.dirname(run_path), "chart.png")
    return panels, run, chart if os.path.exists(chart) else None


def _performance_key(panel):
    """Primary reported result, then work survived, then source duration.

    The first stat is the game's declared headline metric (LINES for this
    Tetris integration), not a Tetris-specific field in the video tool.
    Lexical key order is reversed separately so a complete tie is stable.
    """
    return (float(panel.get("value", 0)), int(panel.get("decisions", 0)),
            float(ending(panel)["time"]))


def checkpoint_champions(panels):
    """Choose one best recorded state for every checkpoint, in run order."""
    groups, order = {}, []
    for panel in panels:
        checkpoint = checkpoint_of(panel)
        if checkpoint not in groups:
            groups[checkpoint] = []
            order.append(checkpoint)
        groups[checkpoint].append(panel)
    champions = []
    for checkpoint in order:
        ranked = sorted(groups[checkpoint],
                        key=lambda p: (-_performance_key(p)[0],
                                       -_performance_key(p)[1],
                                       -_performance_key(p)[2], p["key"]))
        champion = ranked[0]
        champion["candidates"] = len(ranked)
        champions.append(champion)
    return champions


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


def _allocate_phase_budgets(durations, total):
    """Share a finite source-time budget without erasing short phases."""
    durations = [max(0.0, float(value)) for value in durations]
    total = min(sum(durations), max(0.0, float(total)))
    if total >= sum(durations) - 1e-6:
        return durations
    count = max(1, sum(1 for value in durations if value > 0))
    floor = min(MIN_PHASE_SOURCE_SECONDS, total / count)
    budgets = [min(value, floor) for value in durations]
    remaining = total - sum(budgets)
    for _unused in range(20):
        open_slots = [i for i, value in enumerate(durations)
                      if value - budgets[i] > 1e-6]
        if remaining <= 1e-6 or not open_slots:
            break
        weights = [math.sqrt(durations[i]) for i in open_slots]
        weight_total = sum(weights) or float(len(weights))
        spent = 0.0
        for index, weight in zip(open_slots, weights):
            share = remaining * weight / weight_total
            add = min(share, durations[index] - budgets[index])
            budgets[index] += add
            spent += add
        if spent <= 1e-9:
            break
        remaining -= spent
    return budgets


def _phase_windows(start, end, budget, phase_index):
    """Evenly sample a phase, always retaining its actual ending.

    Long stretches become several readable 4--7 output-second excerpts rather
    than one high-speed blur. The first phase also keeps the true opening.
    """
    start, end = float(start), float(end)
    duration = max(0.0, end - start)
    budget = min(duration, max(0.0, float(budget)))
    if budget <= 1e-6:
        return []
    if budget >= duration - 1e-6:
        return [(start, end)]
    count = max(1, int(math.ceil(budget / CLIP_SOURCE_SECONDS)))
    if phase_index == 0 and count == 1 and duration > budget + 1.0:
        count = 2
    length = budget / count
    if count == 1:
        return [(end - length, end)]
    travel = duration - length
    return [(start + travel * i / (count - 1),
             start + travel * i / (count - 1) + length)
            for i in range(count)]


def source_at(plan_data, output_second):
    """Source clock displayed at an output timestamp, including jump cuts."""
    target = max(0.0, float(output_second))
    cuts = plan_data.get("cuts") or []
    for cut in cuts:
        if target <= cut["out_end"] + 1e-6:
            within = max(0.0, target - cut["out_start"])
            return min(cut["source_end"],
                       cut["source_start"] + within * plan_data["speed"])
    return cuts[-1]["source_end"] if cuts else 0.0


def plan(panels, seconds=60.0, card_seconds=CARD_SECONDS,
         max_speed=MAX_SPEED, **_ignored):
    """Build a <=10x race from selected source windows on one shared clock."""
    if not panels:
        raise ValueError("a checkpoint race needs at least one panel")
    if max_speed <= 0:
        raise ValueError("maximum short speed must be greater than zero")
    ends = [ending(panel) for panel in panels]
    groups = _groups(ends)
    body_target = max(2.0, float(seconds) - float(card_seconds))

    bounds, previous = [], 0.0
    for group in groups:
        terminal = max(previous, max(item["time"] for item in group))
        bounds.append((previous, terminal, group))
        previous = terminal
    total_source = bounds[-1][1]
    needed = total_source / body_target if body_target else max_speed
    speed = min(float(max_speed), max(1.0, needed))
    source_budget = min(total_source, body_target * speed)
    budgets = _allocate_phase_budgets(
        [end - start for start, end, _group in bounds], source_budget)

    cuts, cursor = [], 0.0
    for phase_index, ((start, end, _group), budget) in enumerate(zip(bounds, budgets)):
        for source_start, source_end in _phase_windows(start, end, budget, phase_index):
            length = (source_end - source_start) / speed
            cuts.append({"id": len(cuts), "phase": phase_index,
                         "source_start": round(source_start, 3),
                         "source_end": round(source_end, 3),
                         "out_start": round(cursor, 3),
                         "out_end": round(cursor + length, 3)})
            cursor += length

    phases, ending_output = [], {}
    for index, (_start, _end, group) in enumerate(bounds):
        owned = [cut for cut in cuts if cut["phase"] == index]
        if not owned:
            continue
        phase = {"keys": [item["key"] for item in group],
                 "source_start": owned[0]["source_start"],
                 "source_end": owned[-1]["source_end"],
                 "start": owned[0]["out_start"], "end": owned[-1]["out_end"],
                 "show_start": owned[0]["out_start"],
                 "show_end": owned[-1]["out_end"],
                 "reasons": sorted({item["reason"] for item in group})}
        phases.append(phase)
        for item in group:
            ending_output[item["key"]] = phase["end"]
    for item in ends:
        item["output_time"] = ending_output.get(item["key"], cursor)

    final = groups[-1]
    removed = max(0.0, total_source - sum(
        cut["source_end"] - cut["source_start"] for cut in cuts))
    return {"phases": phases, "endings": ends, "cuts": cuts,
            "body_seconds": cursor, "card_seconds": float(card_seconds),
            "seconds": cursor + float(card_seconds), "speed": speed,
            "max_speed": float(max_speed), "source_seconds": total_source,
            "removed_source_seconds": removed, "outcome": _outcome(final),
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


def overlay_font_file():
    """Bundled display font used only for model-written short-form copy."""
    if not os.path.isfile(OVERLAY_FONT_FILE):
        raise RuntimeError("missing bundled overlay font: %s" % OVERLAY_FONT_FILE)
    return OVERLAY_FONT_FILE


def overlay_font(size):
    return ImageFont.truetype(overlay_font_file(), size)


def _filter_path(path):
    """Escape a local font path for FFmpeg's filtergraph parser."""
    value = os.path.abspath(path).replace("\\", "/")
    for char in (":", ",", "[", "]", ";", "'"):
        value = value.replace(char, "\\" + char)
    return value


def _overlay_size(text, width, base=48, minimum=24):
    """Fit proportional display lettering instead of fixed-cell house type."""
    font = overlay_font(base)
    measured = max(1.0, float(font.getlength(str(text))))
    return max(minimum, min(base, int(base * (width - 24) / measured)))


def _enabled_label(chain, output, text, start, end, y, size, font=None):
    return _enabled_text(chain, output, text, start, end,
                         "(w-text_w)/2", y, size, font=font)


def _enabled_text(chain, output, text, start, end, x, y, size, font=None):
    font_path = video.font_file() if font is None else font
    font_arg = ("fontfile=%s:" % _filter_path(font_path)) if font_path else ""
    return ("%sdrawtext=%stext='%s':fontcolor=black:fontsize=%d:box=1:"
            "boxcolor=white:boxborderw=10:x=%s:y=%d:"
            "enable='between(t\\,%.3f\\,%.3f)'%s"
            % (chain, font_arg, _escape_text(text), size, x, y, start, end, output))


def _phase_heading(phase, final, by_key):
    names = " + ".join(by_key[key]["label"] for key in phase["keys"])
    if final:
        return "%s  %s" % (names, final)
    return names


def _clock(seconds):
    value = max(0, int(seconds))
    hours, value = divmod(value, 3600)
    minutes, secs = divmod(value, 60)
    return ("%d:%02d:%02d" % (hours, minutes, secs) if hours
            else "%02d:%02d" % (minutes, secs))


def render_body(plan_data, panels, path, size=SIZE, handle="", **_ignored):
    """Render champion roster, one stable spotlight, jump cuts and source clock."""
    width, height = size
    by_key = {panel["key"]: panel for panel in panels}
    roster, dock_top = _roster_layout(panels, size)
    phase_for = {key: phase for phase in plan_data["phases"]
                 for key in phase["keys"]}
    spots = {key: place for phase in plan_data["phases"]
             for key, place in _spot_layout(phase["keys"], by_key, size,
                                            dock_top).items()}
    cuts = plan_data["cuts"]
    max_source = max(cut["source_end"] for cut in cuts)
    inputs, filters = [], []
    final_keys = set(plan_data["survivors"])
    endings = {item["key"]: item for item in plan_data["endings"]}

    for index, panel in enumerate(panels):
        key = panel["key"]
        inputs += ["-i", panel["path"]]
        pad = max(0.0, max_source - panel["seconds"])
        raw = ["[p%d_%d]" % (index, i) for i in range(len(cuts))]
        if len(cuts) == 1:
            filters.append("[%d:v]tpad=stop_mode=clone:stop_duration=%.3f%s"
                           % (index, pad, raw[0]))
        else:
            filters.append("[%d:v]tpad=stop_mode=clone:stop_duration=%.3f,"
                           "split=%d%s" % (index, pad, len(cuts), "".join(raw)))
        pieces = []
        for cut_index, cut in enumerate(cuts):
            label = "[pc%d_%d]" % (index, cut_index)
            filters.append("%strim=start=%.3f:end=%.3f,"
                           "setpts=(PTS-STARTPTS)/%.6f%s"
                           % (raw[cut_index], cut["source_start"], cut["source_end"],
                              plan_data["speed"], label))
            pieces.append(label)
        joined = "[joined%d]" % index
        if len(pieces) == 1:
            filters.append("%snull%s" % (pieces[0], joined))
        else:
            filters.append("%sconcat=n=%d:v=1:a=0%s"
                           % ("".join(pieces), len(pieces), joined))
        filters.append("%ssplit=2[tile_raw%d][spot_raw%d]"
                       % (joined, index, index))
        tile = roster[key]
        dim = ("eq=brightness=-0.30:saturation=0.38:enable='gte(t\\,%.3f)',"
               % endings[key]["output_time"]) if key not in final_keys else ""
        filters.append("[tile_raw%d]%sscale=%d:%d:flags=neighbor,setsar=1[tile%d]"
                       % (index, dim, tile["w"], tile["h"], index))

        phase, spot = phase_for[key], spots[key]
        filters.append("[spot_raw%d]scale=%d:%d:flags=neighbor,setsar=1[spot%d]"
                       % (index, spot["w"], spot["h"], index))

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
        filters.append("%s[spot%d]overlay=%d:%d:enable='between(t\\,%.3f\\,%.3f)'%s"
                       % (stage, index, spot["x"], spot["y"],
                          phase["show_start"], phase["show_end"], output))
        stage = output

    for index, phase in enumerate(plan_data["phases"]):
        output = "[d%d]" % index
        final = plan_data["outcome"] if index == len(plan_data["phases"]) - 1 else ""
        heading = _phase_heading(phase, final, by_key)
        # The pictures overlap during a dissolve, but their claims must not:
        # keep NEXT OUT until the real death boundary, then name the survivor.
        filters.append(_enabled_label(stage, output, heading, phase["start"],
                                      phase["end"], 42,
                                      video.fit_size(heading, width - 60, 34, 14)))
        stage = output

    # The timer is SOURCE time, not output time. It visibly jumps whenever an
    # edit removes minutes of play, which is what keeps a 70-minute run honest
    # without turning it into an unreadable 70x blur.
    for tick in range(int(math.ceil(plan_data["body_seconds"]))):
        output = "[tm%d]" % tick
        text = "GAME CLOCK  %s" % _clock(source_at(plan_data, tick + 0.001))
        filters.append(_enabled_text(stage, output, text, tick,
                                     min(plan_data["body_seconds"], tick + 1.01),
                                     "w-text_w-22", 24, 17))
        stage = output

    previous_source = cuts[0]["source_start"]
    for cut in cuts:
        skipped = cut["source_start"] - previous_source
        previous_source = cut["source_end"]
        if cut["id"] == 0 or skipped < 1.0:
            continue
        output = "[jump%d]" % cut["id"]
        text = "TIME CUT  +%s" % _clock(skipped)
        filters.append(_enabled_text(stage, output, text, cut["out_start"],
                                     min(cut["out_end"], cut["out_start"] + 0.9),
                                     22, 24, 17))
        stage = output

    by_cut = {cut["id"]: cut for cut in cuts}
    for index, overlay in enumerate(plan_data.get("overlays") or []):
        cut = by_cut.get(int(overlay.get("cut", -1)))
        if not cut:
            continue
        start = cut["out_start"] + min(0.25, (cut["out_end"] - cut["out_start"]) / 4)
        end = cut["out_end"] - 0.12
        if end <= start:
            continue
        output = "[copy%d]" % index
        text = overlay["text"]
        filters.append(_enabled_label(stage, output, text, start, end,
                                      dock_top - 110,
                                      _overlay_size(text, width - 70),
                                      font=overlay_font_file()))
        stage = output

    static = []
    for panel in panels:
        tile = roster[panel["key"]]
        static.append((panel["label"], tile["label_x"], tile["label_y"],
                       video.fit_size(panel["label"], tile["label_width"], 15, 8)))
    static.append(("%sx MAX  /  JUMP CUTS SHOWN ON CLOCK" %
                   video.rate_text(plan_data["speed"]), width // 2, height - 28, 14))
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


def _value_at(panel, second):
    current = 0
    for event in panel.get("timeline") or []:
        if float(event.get("t", 0)) > second:
            break
        current = event.get("v", current)
    return current


def _short_prompt(plan_data, panels, run):
    by_key = {panel["key"]: panel for panel in panels}
    phase_key = {index: phase["keys"][0]
                 for index, phase in enumerate(plan_data["phases"])}
    champions = []
    for panel in panels:
        metric, value = (panel.get("stats") or [["RESULT", panel.get("value", 0)]])[0]
        champions.append(
            "- %s chose state %s: %s %s, %s decisions, %s source time."
            % (panel["label"], panel.get("state") or "unknown", value, metric,
               panel.get("decisions", 0), _clock(ending(panel)["time"])))
    cuts = []
    for cut in plan_data["cuts"]:
        panel = by_key[phase_key[cut["phase"]]]
        before = _value_at(panel, cut["source_start"])
        after = _value_at(panel, cut["source_end"])
        clears = [int(event.get("d", 0)) for event in panel.get("timeline") or []
                  if cut["source_start"] <= float(event.get("t", -1))
                  <= cut["source_end"]]
        cuts.append(
            "- cut %d, %s-%s on the game clock, spotlight %s: headline metric "
            "%s to %s; biggest single gain %s."
            % (cut["id"], _clock(cut["source_start"]), _clock(cut["source_end"]),
               panel["label"], before, after, max(clears or [0])))
    winner = " + ".join(by_key[key]["label"] for key in plan_data["survivors"])
    game = run.get("title") or run.get("game") or "this game"
    channel = run.get("channel") or run.get("handle") or "the trainer's channel"
    return """You are writing the SILENT vertical companion to a checkpoint-progression film.

The game is %s. The channel is %s.

It is a race between the best recorded game from each training checkpoint. It
is not a statistical report. A small roster stays at the bottom, one game is
large, and the source GAME CLOCK visibly jumps whenever uneventful minutes are
cut. The footage never exceeds %.1fx. The final card crowns %s.

CHECKPOINT CHAMPIONS
%s

THE EDITED WINDOWS
%s

Write sparse, strong on-screen copy that makes this entertaining with the sound
off. Return at most 10 overlays and no more than one for any cut. `cut` must be
one of the numbered cuts above. Each overlay is at most 42 characters, readable
in one glance, specific to this experiment, and either explains the stakes,
notices a real reversal, or makes one dry joke. Do not narrate falling pieces.
Do not invent numbers. Do not use emoji, hashtags, quotation marks, stage
directions, or generic hype. Leave some cuts text-free.

`challenge` is the final card's one sentence, spoken AS THE WINNING AI directly
to the human who trained it. At most 70 characters. Confident and funny, not
sentient, threatening, or melodramatic. It challenges the human to the promised
head-to-head match.

Also write upload copy for four distinct posts:
- youtube_long: the narrated 16:9 checkpoint comparison.
- youtube_short: this silent, jump-cut checkpoint race.
- instagram: this vertical Reel.
- tiktok: this vertical video.

Every platform gets a title, description, and tags. YouTube long description is
500-1200 characters and explains the fair comparison, one champion per
checkpoint, and that the winner will face its trainer. YouTube Short is under
500 characters. Instagram and TikTok descriptions are under 300 characters.
Tags contain plain search terms without `#`; descriptions may end with a small,
relevant hashtag set. Do not claim monotonic improvement unless the facts above
show it. Do not claim all states were displayed: one champion was selected from
each checkpoint.
""" % (game, channel, plan_data["speed"], winner,
       "\n".join(champions), "\n".join(cuts))


def _writer_options(run, override=None):
    return {
        "backend": override or run.get("writer") or "chatgpt",
        "cascade_order": run.get("writer_cascade") or ["chatgpt", "ollama"],
        "cli": run.get("writer_cli") or "codex",
        "chatgpt_model": run.get("chatgpt_model"),
        "chatgpt_timeout": int(run.get("chatgpt_timeout") or 300),
        "model": run.get("writer_model") or text_writer.DEFAULT_MODEL,
        "host": run.get("writer_host") or text_writer.DEFAULT_HOST,
        "think": True,
        "verbose": True,
    }


def _clean_package(data, plan_data):
    valid = {cut["id"] for cut in plan_data["cuts"]}
    overlays, used = [], set()
    for item in data.get("overlays") or []:
        cut = int(item.get("cut", -1))
        text = text_writer.clean_spoken(item.get("text", ""))
        text = text_writer.trim_words(text, 42)
        if cut not in valid or cut in used or not text:
            continue
        overlays.append({"cut": cut, "text": text})
        used.add(cut)
        if len(overlays) >= 10:
            break
    challenge = text_writer.trim_words(
        text_writer.clean_spoken(data.get("challenge", "")), 70)
    if not challenge:
        challenge = "You trained me. Now try to beat me."
    clean = {"overlays": overlays, "challenge": challenge}
    for name in ("youtube_long", "youtube_short", "instagram", "tiktok"):
        source = data.get(name) or {}
        clean[name] = {
            "title": text_writer.clean_spoken(source.get("title", "")),
            "description": text_writer.clean_spoken(source.get("description", "")),
            "tags": [str(tag).strip().lstrip("#") for tag in source.get("tags", [])
                     if str(tag).strip()],
        }
    return clean


def write_short_package(plan_data, panels, run, out_dir, writer_backend=None):
    """Ask once for overlays plus all upload copy, preserving the exact prompt."""
    prompts = os.path.join(out_dir, "prompts")
    os.makedirs(prompts, exist_ok=True)
    prompt = _short_prompt(plan_data, panels, run)
    prompt_path = os.path.join(prompts, "short_overlays_and_upload_copy.txt")
    with open(prompt_path, "w", encoding="utf-8") as handle:
        handle.write(prompt)
    with open(os.path.join(prompts, "short_output_schema.json"), "w",
              encoding="utf-8") as handle:
        json.dump(SHORT_COPY_SCHEMA, handle, indent=2)
    data = text_writer.write(prompt, SHORT_COPY_SCHEMA,
                             **_writer_options(run, writer_backend))
    if not data:
        raise RuntimeError("the short writer returned nothing; its prompt is saved at %s"
                           % prompt_path)
    package = _clean_package(data, plan_data)
    with open(os.path.join(out_dir, "short_copy.json"), "w", encoding="utf-8") as handle:
        json.dump(package, handle, indent=2)
    return package


def _fenced(label, value):
    return "**%s**\n\n````\n%s\n````\n" % (label, str(value).strip())


def save_upload_package(out_dir, package, plan_data, panels):
    """Progression equivalent of studio.py's upload brief and paste block."""
    long_video = ("narrated.mp4" if os.path.exists(os.path.join(out_dir, "narrated.mp4"))
                  else "grid.mp4")
    short_video = "short_9x16.mp4"
    order = ("youtube_long", "youtube_short", "instagram", "tiktok")
    labels = {"youtube_long": "YouTube long", "youtube_short": "YouTube Short",
              "instagram": "Instagram Reel", "tiktok": "TikTok"}
    files = {"youtube_long": long_video, "youtube_short": short_video,
             "instagram": short_video, "tiktok": short_video}
    sections, plain = [], []
    for name in order:
        item = package[name]
        tags = ", ".join(item["tags"])
        sections.append("## %s\n\nAttach `%s`.\n\n%s\n%s\n%s" % (
            labels[name], os.path.abspath(os.path.join(out_dir, files[name])),
            _fenced("Title", item["title"]),
            _fenced("Description", item["description"]),
            _fenced("Tags", tags)))
        plain.extend(["=" * 72, labels[name].upper(), "FILE: " + files[name],
                      "TITLE:", item["title"], "DESCRIPTION:", item["description"],
                      "TAGS:", tags, ""])
    winner = " + ".join(next(p["label"] for p in panels if p["key"] == key)
                        for key in plan_data["survivors"])
    brief = """# Upload brief — checkpoint progression

The long film and vertical short are two edits of the same deterministic runs.
The short selects one best recorded state per checkpoint, never exceeds %.1fx,
and visibly labels every jump in source time. Its crowned winner is **%s**.

Review every post privately before publishing. The model-written fields below
are preserved verbatim; the prompts that produced them are in `prompts/`.

%s
""" % (plan_data["speed"], winner, "\n\n".join(sections))
    with open(os.path.join(out_dir, "UPLOAD_BRIEF.md"), "w", encoding="utf-8") as handle:
        handle.write(brief)
    with open(os.path.join(out_dir, "paste.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(plain))
    metadata = {"winner": winner, "challenge": package["challenge"],
                "selected_games": [panel["key"] for panel in panels],
                "source_seconds": plan_data["source_seconds"],
                "removed_source_seconds": plan_data["removed_source_seconds"],
                "speed": plan_data["speed"],
                "uploads": {name: package[name] for name in order}}
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def preserve_long_prompts(out_dir):
    """Extract every narration prompt embedded in script*.json into prompts/."""
    prompts = os.path.join(out_dir, "prompts")
    os.makedirs(prompts, exist_ok=True)
    for name in sorted(os.listdir(out_dir)):
        if not name.startswith("script") or not name.endswith(".json"):
            continue
        try:
            data = json.load(open(os.path.join(out_dir, name), encoding="utf-8"))
        except (OSError, ValueError):
            continue
        prompt = data.get("_prompt")
        if prompt:
            target = os.path.join(prompts, name.replace("script", "long_narration")
                                  .replace(".json", ".txt"))
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(prompt)


def _plate_center(draw, canvas_width, y, text, font, pad=10):
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    video.plate(draw, ((canvas_width - width) / 2, y), text, font, pad=pad)


def _wrap_words(text, width=30):
    lines, current = [], ""
    for word in str(text).split():
        trial = (current + " " + word).strip()
        if current and len(trial) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def build_crown_card(plan_data, panels, path, chart=None, size=SIZE):
    """One crown, one checkpoint name, one challenge. Never a stats page."""
    width, height = size
    image = Image.new("RGB", size, (9, 11, 16))
    draw = ImageDraw.Draw(image)
    title_font = video.house_font(58)
    body_font = overlay_font(42)
    by_key = {panel["key"]: panel for panel in panels}
    survivors = [by_key[key] for key in plan_data["survivors"]]
    cx = width // 2
    gold, edge = (255, 200, 45), (255, 238, 150)
    crown = [(cx - 190, 340), (cx - 155, 155), (cx - 62, 260),
             (cx, 105), (cx + 62, 260), (cx + 155, 155), (cx + 190, 340)]
    draw.polygon(crown, fill=gold, outline=edge)
    draw.rounded_rectangle((cx - 195, 315, cx + 195, 390), radius=14,
                           fill=gold, outline=edge, width=4)
    winner = " + ".join(panel["label"] for panel in survivors)
    _plate_center(draw, width, 500, winner.upper(), title_font, 18)
    challenge = plan_data.get("challenge") or "You trained me. Now try to beat me."
    y = 720
    for line in _wrap_words(challenge, 28):
        _plate_center(draw, width, y, line, body_font, 12)
        y += 76
    footer = "THE WINNER PLAYS ITS TRAINER NEXT"
    _plate_center(draw, width, height - 230, footer, video.house_font(20), 9)
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
           grow=False):
    """Render the elimination body and its crown card, then join them losslessly."""
    work = work or os.path.join(os.path.dirname(out_path), "elimination_work")
    os.makedirs(work, exist_ok=True)
    body = render_body(plan_data, panels, os.path.join(work, "body.mp4"), size,
                       handle)
    if plan_data["card_seconds"] <= 0:
        shutil.copyfile(body, out_path)
        return out_path
    card_png = build_crown_card(plan_data, panels, os.path.join(work, "crown.png"),
                                chart=None, size=size)
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
                        help="use only the first N checkpoint champions (default: all)")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="total target length including the card")
    parser.add_argument("--card-seconds", type=float, default=CARD_SECONDS)
    parser.add_argument("--max-speed", type=float, default=MAX_SPEED,
                        help="hard playback ceiling, never more than 10x")
    parser.add_argument("--writer", choices=text_writer.BACKENDS, default=None,
                        help="override the long run's writer for overlays/upload copy")
    parser.add_argument("--handle", default=None,
                        help="default: run handle, then studio.json watermark")
    parser.add_argument("--out", default=os.path.join(ROOT, "progression_out",
                                                       "short_9x16.mp4"))
    args = parser.parse_args()

    run = {}
    try:
        if args.run:
            panels, run, _inferred_card = panels_from_run(args.run)
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
    if not panels:
        raise SystemExit("no panels with timelines were found; render them fresh first")
    panels = checkpoint_champions(panels)
    if args.count > 0:
        panels = panels[:args.count]
    if not 0 < args.max_speed <= MAX_SPEED:
        raise SystemExit("--max-speed must be greater than zero and no more than 10")

    handle = args.handle if args.handle is not None else run.get("handle")
    if handle is None:
        with open(os.path.join(ROOT, "studio.json"), encoding="utf-8") as fh:
            handle = json.load(fh).get("watermark", "")
    try:
        plan_data = plan(panels, args.seconds, args.card_seconds,
                         max_speed=args.max_speed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    try:
        package = write_short_package(plan_data, panels, run, out_dir, args.writer)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    plan_data["overlays"] = package["overlays"]
    plan_data["challenge"] = package["challenge"]
    plan_path = os.path.join(out_dir, "elimination.json")
    with open(plan_path, "w", encoding="utf-8") as fh:
        json.dump(plan_data, fh, indent=2)
    render(plan_data, panels, args.out, chart=None, handle=handle,
           work=os.path.join(out_dir, "elimination_work"))
    preserve_long_prompts(out_dir)
    save_upload_package(out_dir, package, plan_data, panels)
    print("%d checkpoint champions, %d elimination phases, %s" %
          (len(panels), len(plan_data["phases"]), plan_data["outcome"].lower()))
    print("%.1fx maximum, %s removed with %d visible jump cuts" %
          (plan_data["speed"], _clock(plan_data["removed_source_seconds"]),
           max(0, len(plan_data["cuts"]) - 1)))
    print("wrote %s" % plan_path)
    print("wrote %s" % args.out)
    print("wrote %s" % os.path.join(out_dir, "UPLOAD_BRIEF.md"))


if __name__ == "__main__":
    main()
