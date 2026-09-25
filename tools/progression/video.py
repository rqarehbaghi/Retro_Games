#!/usr/bin/env python3
"""Render a grid of agents playing: N checkpoints across, M start states down.

    # Tetris: three checkpoints, two states, played until they lose
    python tools/progression/video.py --game TetrisTime-Nes-v0 \
        --folder checkpoints/tetris_v54 --columns 5000,100000,200000 \
        --states p1_01,p1_02 --player 1

    # Mario: the same picture, a different game
    python tools/progression/video.py --game SuperMarioBros3-Nes-v0 \
        --folder checkpoints/SuperMarioBros3-Nes-v0 \
        --columns iter_1.zip,final_iter_6000.zip --states level1,level2 \
        --cap 4000 --crop full

Nothing here knows what game it is showing. How a move is chosen comes from
`tools/progression/runners.py`, dispatched on the game's declared algorithm;
the numbers burnt onto each panel are the game's own `report_stats`; the crop
is derived from the ACTUAL frame size and, when the game declares one, its
`observation.grid`. Add `--still` to render one frame and look before
committing to a full render.

The video is an ILLUSTRATION, never the evidence: which states it shows was
decided before any checkpoint ran, and any claim about a checkpoint comes from
the sweep, not from six panels.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.progression import chart as chart_card, runners, speech   # noqa: E402

OUT = os.path.join(ROOT, "progression_out")
PANELS = os.path.join(OUT, "panels")
SPEED = 3.0
HEADINGS = {"control": "Untrained control"}
METRICS = 2                          # how the panel's numbers were measured;
                                     # 2 counts HUD counter wraps. A panel
                                     # measured by an older rule is re-rendered
                                     # rather than compared against a newer one
PANEL_SCALE = 2                      # nearest-neighbour upscale of the crop
MARGIN = 6                           # game pixels kept around a declared grid
CROP_PAD = 64                        # pixels of readouts kept beside it
END_FRAMES = 30                      # how long the end label is held
END_LABELS = {"game_over": "GAME OVER", "placement_cap": "CAP REACHED",
              "truncated": "TRUNCATED"}


def house_font(size):
    for path in (os.path.expanduser("~/.local/share/fonts/PressStart2P-Regular.ttf"),
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:                                     # noqa: BLE001
                pass
    return ImageFont.load_default()


def font_file():
    for path in (os.path.expanduser("~/.local/share/fonts/PressStart2P-Regular.ttf"),
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(path):
            return path
    return None


def plate(draw, xy, text, font, pad=4):
    """Black on an opaque white plate -- the house rule for every overlay."""
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    draw.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill=(255, 255, 255))
    draw.text((x, y), text, font=font, fill=(0, 0, 0))


def crop_box(frame, grid, crop, pad=CROP_PAD):
    """Which rectangle of the screen this panel shows.

    Derived from the frame the emulator actually produced and, for the two
    grid-relative modes, the game's declared well -- never from NES constants,
    so a game with a different resolution or a different layout still lands on
    its own board. A game with no grid block can only be shown whole."""
    screen_h, screen_w = np.asarray(frame).shape[:2]
    if crop == "full" or not grid:
        return 0, 0, screen_w, screen_h
    x, y = int(grid["x"]), int(grid["y"])
    cell = int(grid["cell"])
    right, bottom = x + int(grid["cols"]) * cell, y + int(grid["rows"]) * cell
    if crop == "well":
        # The board alone. Tight, and it cuts off NEXT / SCORE / LINES, which
        # is why it is not the default.
        return (max(0, x - MARGIN), max(0, y - 2 * cell),
                min(screen_w, right + MARGIN), min(screen_h, bottom + MARGIN))
    # "half": the agent's SIDE of the screen, full height -- its board plus
    # `pad` pixels of the readouts beside it. 64 was chosen by rendering
    # candidates side by side on TetrisTime: it is the narrowest crop that
    # keeps the SCORE / LINES / LEVEL labels whole. It cannot be narrowed
    # without cutting them, because on that screen the OTHER player's well
    # sits directly BELOW those labels -- no rectangle keeps one and drops
    # the other. --pad is the knob for a game where the trade goes the other
    # way.
    if x < screen_w // 2:
        return 0, 0, min(screen_w, right + pad), screen_h
    return max(0, x - pad), 0, screen_w, screen_h


class PanelWriter:
    """Turns the frames a runner emits into one panel .mp4.

    Counters are drawn per frame here rather than as ffmpeg drawtext filters: a
    trajectory has hundreds of decisions, and one enable-window filter per
    change would be an unusable filter graph.

    Sizing waits for the first frame, because only the emulator knows how big
    its screen is."""

    def __init__(self, path, crop, grid, pad=CROP_PAD, show_stats=True):
        self.path, self.crop, self.grid, self.pad = path, crop, grid, pad
        self.show_stats = show_stats
        self.ff = self.font = None
        self.box = None
        self.w = self.h = 0

    def _open(self, frame):
        x0, y0, x1, y1 = crop_box(frame, self.grid, self.crop, self.pad)
        self.box = (x0, y0, x1, y1)
        self.w, self.h = (x1 - x0) * PANEL_SCALE, (y1 - y0) * PANEL_SCALE
        self.font = house_font(max(8, self.w // 20))
        self.ff = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "%dx%d" % (self.w, self.h), "-r", "60", "-i", "-", "-an",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-pix_fmt", "yuv420p", self.path],
            stdin=subprocess.PIPE)

    def __call__(self, frame, state, end=None):
        if self.ff is None:
            self._open(frame)
        x0, y0, x1, y1 = self.box
        panel = Image.fromarray(np.asarray(frame)[y0:y1, x0:x1]).resize(
            (self.w, self.h), Image.NEAREST)
        draw = ImageDraw.Draw(panel)
        # Counters sit a quarter of the way down, in the empty top of the
        # board: at the bottom they covered the stack, which is the one thing
        # the video is about. One per line -- "LINES 147  PIECES 350" on one
        # line is wider than the panel and lost its end. --no-stats turns them
        # off entirely; the game draws its own score row anyway, and the end
        # chart carries the numbers that are actually being compared.
        for i, (label, value) in enumerate(
                (state.get("stats") or []) if self.show_stats else []):
            plate(draw, (6, int(self.h * 0.25) + i * (self.font.size + 12)),
                  "%s %s" % (label, value), self.font)
        if end:
            width = draw.textlength(end, font=self.font)
            plate(draw, ((self.w - width) / 2, self.h // 2 - self.font.size), end, self.font)
        self.ff.stdin.write(panel.tobytes())

    def finish(self, frame, state, end_reason):
        """A few frames of the final board carrying the end label, so the
        freeze that follows in the composite says WHY it stopped -- a capped
        trajectory was censored, a game over was not."""
        label = END_LABELS.get(end_reason, end_reason.upper())
        for _ in range(END_FRAMES):
            self(frame, state, end=label)
        self.ff.stdin.close()
        self.ff.wait()
        return {"path": self.path, "width": self.w, "height": self.h}


def capture_panel(game, checkpoint, state, cap, player, path, crop, pad=CROP_PAD,
                  show_stats=True):
    """Play one (checkpoint, state) pair and write its panel to `path`.

    The numbers come back as well as going into the pixels, and main() saves
    them beside the mp4: a cached panel is a video and nothing else, so
    without the sidecar the end chart could only be built on a run that
    replayed everything."""
    spec = runners.spec_for(game, player, state)
    writer = PanelWriter(path, crop, spec.grid, pad, show_stats)
    result = runners.run(game, checkpoint, state, cap, player, writer)
    panel = writer.finish(result["last"], {"stats": result["stats"]}, result["end_reason"])
    panel.update({"end_reason": result["end_reason"], "decisions": result["decisions"],
                  "stats": result["stats"]})
    return panel


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def _clock(t):
    return "%d:%05.2f" % (int(t // 60), t % 60)


def rate_text(x):
    """A speed as a viewer would write it: 1.8x, 8x, 48x.

    --speed auto solves for whatever makes the games last as long as the
    narration, which is a number like 1.76534, and printing that on screen
    says "a machine wrote this caption" rather than telling anyone anything."""
    return ("%.1f" % x).rstrip("0").rstrip(".")


def fit_size(text, width, base, minimum=11):
    """Shrink a label until it fits the panel it names.

    Press Start 2P is fixed-cell, about one em per character, so the LENGTH of
    a label decides its rendered width almost exactly -- and a 2x3 grid of tall
    boards makes each panel narrow enough that two neighbouring labels ran into
    each other and read as one word."""
    if not text:
        return base
    return max(minimum, min(base, int((width - 16) / len(text))))


def drawtext(chain, labels):
    """A drawtext per label, threaded through one filter chain.

    House style, same as the studio pipeline: black glyphs on an OPAQUE white
    plate. An outline or a shadow disappears into a bright frame."""
    path = font_file()
    out, src = "", chain
    for k, (body, cx, cy, size) in enumerate(labels):
        dst = "[t%d]" % (k + 1) if k < len(labels) - 1 else "[final]"
        # A string cx is an ffmpeg x EXPRESSION, used as-is -- that is how the
        # handle sits against the right edge without knowing how wide it is.
        x = cx if isinstance(cx, str) else "%d-text_w/2" % cx
        out += ("%sdrawtext=%stext='%s':fontcolor=black:fontsize=%d:box=1:boxcolor=white:"
                "boxborderw=8:x=%s:y=%d%s;"
                % (src, ("fontfile=%s:" % path) if path else "",
                   body.replace(":", "\\:").replace("'", ""), size, x, cy, dst))
        src = dst
    return out.rstrip(";")


def rate_plan(lengths, speed, catch_up, hold=0.0):
    """How fast to play, as fewer and fewer games are still going.

    Uncapped, the last survivor holds the film open while everything else sits
    frozen: with ten panels, nine of them finished, the viewer watches nine
    still images for as long as the tenth takes. So the rate rises as panels
    end -- when a tenth of the grid is still playing there is a tenth as much
    to look at, and it can be played proportionally faster.

    Returns [(source_end, rate)] covering 0..max(lengths), and the total output
    length. The boundaries are exactly the moments a game ends, so the rate
    only ever changes when the picture changes."""
    total = max(lengths)
    marks = sorted({round(x, 3) for x in lengths} | {round(total, 3)})
    plan, previous = [], 0.0
    for mark in marks:
        if mark <= previous:
            continue
        # Still playing THROUGH this segment, not at its edge.
        alive = sum(1 for x in lengths if round(x, 3) >= mark)
        boost = min(catch_up, len(lengths) / float(max(1, alive)))
        plan.append((mark, speed * max(1.0, boost)))
        previous = mark
    if hold > 0:
        # A beat at REAL time on the final board before the card replaces it.
        # Cutting away the instant the last game ends means nobody sees how it
        # ended, and at 40x the difference between "ended" and "gone" is two
        # frames. Rate 1.0 means these seconds are exactly these seconds.
        plan.append((total + hold, 1.0))
    out = sum((end - start) / rate
              for (end, rate), start in zip(plan, [0.0] + [p[0] for p in plan[:-1]]))
    return plan, out


def remap(plan):
    """The rate plan as ONE setpts expression, in output seconds.

    Every panel gets the same expression, which is what keeps them in step --
    a per-panel speed would drift them apart. Written as a sum of clipped
    segments rather than nested ifs: out(T) = SUM over segments of the time
    spent in that segment divided by its rate."""
    # The commas inside max() and min() MUST be escaped: a comma is what
    # separates one filter from the next in a filtergraph, so an unescaped one
    # made ffmpeg look for a filter called 'min(T' and fail.
    terms, start = [], 0.0
    for end, rate in plan:
        terms.append("max(0\\,min(T\\,%.3f)-%.3f)/%.4f" % (end, start, rate))
        start = end
    return "setpts=(%s)/TB" % "+".join(terms)


def compose(cells, out_path, ncols=None, headers=None, still=False, still_at=8.0,
            speed=SPEED, chart=None, chart_seconds=12.0, catch_up=1.0, hold=0.0,
            handle=""):
    """Every panel into one 1920x1080 frame, each frozen once its game ends.

    `cells` is a flat list in READING ORDER, each with a path, a size and a
    `label`. The shape is `ncols` wide and however many rows that needs, so
    six panels are 3x2 or 2x3 or 6x1 without any of this knowing what a
    checkpoint or a state is.

    `headers` is the older per-column/per-row labelling, used when the grid IS
    checkpoints across by states down; pass None and each panel carries its own
    caption instead, which is the only thing that still reads correctly once
    the panels wrap.
    """
    ncols = ncols or len(cells)
    nrows = int(math.ceil(len(cells) / float(ncols)))
    longest = max(duration(c["path"]) for c in cells)
    top = 96 if headers else 70
    cell_w, cell_h = 1920 // ncols, (1080 - top - 24) // nrows
    pw = min(cell_w - 40, int((cell_h - 34) * cells[0]["width"] / cells[0]["height"]))
    ph = int(pw * cells[0]["height"] / cells[0]["width"])
    # Panels are usually limited by HEIGHT, not width -- a tall Tetris well in
    # a wide cell. Centring each panel in its own cell then leaves a column of
    # empty screen down the middle, so the whole block is packed to a fixed
    # gutter and centred as one thing.
    gut_x, gut_y = 48, 44
    block_w = ncols * pw + (ncols - 1) * gut_x
    block_h = nrows * ph + (nrows - 1) * gut_y
    x0 = max(0, (1920 - block_w) // 2)
    y0 = max(top, top + ((1080 - top - 24) - block_h) // 2)

    lengths = [duration(c["path"]) for c in cells]
    plan, out_seconds = rate_plan(lengths, speed, catch_up, hold)
    pad_to = longest + hold
    inputs, filters, overlays = [], [], "[bg]"
    for i, cell in enumerate(cells):
        inputs += ["-i", cell["path"]]
        # A finished panel is DIMMED, so the eye goes to the games still being
        # played rather than to a wall of identical frozen boards. It stays on
        # screen -- how a game ended is the thing worth seeing -- it just stops
        # competing. Applied in SOURCE time, before the remap, so the moment it
        # dims is the moment that game actually ended.
        # The LAST game to finish is never dimmed: its ending is the moment
        # the whole film has been building to, and dimming it at the instant
        # it lands is exactly how you miss it.
        dim = ("eq=brightness=-0.28:saturation=0.45:enable='gte(t\\,%.3f)',"
               % lengths[i]) if len(cells) > 1 and lengths[i] < longest - 0.01 else ""
        # Freeze FIRST, in source time, then remap: every panel is then the
        # same length before the speed is applied, so one shared expression
        # keeps them all on the same clock.
        filters.append(
            "[%d:v]tpad=stop_mode=clone:stop_duration=%.3f,%s%s,"
            "scale=%d:%d:flags=neighbor[p%d]"
            % (i, max(0.0, pad_to - lengths[i]), dim, remap(plan), pw, ph, i))
    for i in range(len(cells)):
        cx = x0 + (i % ncols) * (pw + gut_x)
        cy = y0 + (i // ncols) * (ph + gut_y)
        nxt = "[s%d]" % i if i < len(cells) - 1 else "[vout]"
        overlays += "[p%d]overlay=%d:%d%s;" % (i, cx, cy, nxt)
        if i < len(cells) - 1:
            overlays += "[s%d]" % i
    # The background MUST carry a duration. color= is an infinite source, and
    # overlaying onto it made the render run for as long as ffmpeg was left
    # alive -- a 29-minute game at 3x produced 6776 seconds of output and was
    # mistaken for a slow encode rather than an unbounded one. out_seconds comes
    # from the rate plan, which knows what the varying speed adds up to.
    graph = (";".join(filters) + ";"
             + "color=c=black:s=1920x1080:r=60:d=%.3f[bg];" % out_seconds + overlays)
    graph = graph.rstrip(";")

    # Labels go on the composite, not inside the panels: a checkpoint name is
    # wider than one board.
    if headers:
        columns, rows = headers
        labels = [(HEADINGS.get(col, col), x0 + i * (pw + gut_x) + pw // 2, 34,
                   fit_size(HEADINGS.get(col, col), pw + gut_x, 26))
                  for i, col in enumerate(columns)]
        labels += [(row, 1920 // 2, y0 + j * (ph + gut_y) - 30, 20)
                   for j, row in enumerate(rows)]
    else:
        labels = [(cell["label"], x0 + (i % ncols) * (pw + gut_x) + pw // 2,
                   y0 + (i // ncols) * (ph + gut_y) - 34,
                   fit_size(cell["label"], pw + gut_x, 22))
                  for i, cell in enumerate(cells)]
    # Say the rate honestly: with catch-up it is not one number, and a viewer
    # watching the last board suddenly move faster deserves to know why.
    fastest = max(rate for _end, rate in plan)
    labels.append(("%sx speed" % rate_text(speed) if fastest <= speed * 1.01 else
                   "%sx speed, up to %sx as games end"
                   % (rate_text(speed), rate_text(fastest)),
                   1920 // 2, 1054, 16))
    if handle:
        # The channel, bottom right, on the same white plate as everything
        # else. Right-aligned by expression so a longer handle does not run
        # off the frame.
        labels.append((handle, "w-text_w-26", 1046, 18))
    graph += ";" + drawtext("[vout]", labels)

    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error"] + inputs
    last = "[final]"
    if chart and not still:
        # The chart is a still image held for a few seconds, concatenated onto
        # the end in the SAME pass -- writing the grid and then re-encoding it
        # to append a tail would cost a second full encode of the whole video.
        inputs_len = len(cells)
        cmd += ["-loop", "1", "-t", "%.3f" % chart_seconds, "-i", chart]
        # BOTH segments must be TRIMMED, not merely expected to end.
        #
        # Without the trim the grid segment ran far past the background's own
        # duration -- overlay repeats a finished input rather than stopping --
        # and concat happily played that overrun instead of ever reaching the
        # card. It looked right, because the -t on the output cut the result
        # back to exactly the length the card should have made it: a 232s file
        # that was 232s of grid with the card silently never shown. concat also
        # refuses segments that disagree on size, pixel format, sample aspect
        # or frame rate, and it fails at RUN time rather than when the graph is
        # built, so those are pinned here too.
        graph += (";[final]trim=duration=%.3f,setpts=PTS-STARTPTS,fps=60,setsar=1,"
                  "format=yuv420p[gridv];"
                  "[%d:v]scale=1920:1080,trim=duration=%.3f,setpts=PTS-STARTPTS,"
                  "fps=60,setsar=1,format=yuv420p,fade=in:st=0:d=0.5[chartv];"
                  "[gridv][chartv]concat=n=2:v=1:a=0[outv]"
                  % (out_seconds, inputs_len, chart_seconds))
        last = "[outv]"
        out_seconds += chart_seconds
    cmd += ["-filter_complex", graph, "-map", last, "-an"]
    if still:
        # A few seconds in, not frame 1: at frame 1 every board is empty and
        # the still says nothing about the layout in use.
        cmd += ["-ss", str(still_at), "-frames:v", "1", out_path]
    else:
        cmd += ["-t", "%.3f" % out_seconds,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    subprocess.run(cmd, check=True)
    return out_path


def game_title(game):
    """'TetrisTime-Nes-v0' is an integration id, not a name to put on a card."""
    return game.split("-")[0] if "-" in game else game


def build_chart(cells, game, path, order, handle=""):
    """The closing card, built from the panels' own sidecar numbers.

    The headline is whatever the game reports FIRST -- lines for Tetris, score
    for Mario -- and the second figure is the work it took, which is placements
    for one algorithm and env steps for the other. Neither is named here."""
    rows, metric, work, missing = [], "", "decisions", []
    for cell in cells:
        stats = cell.get("stats") or []
        if not stats:
            missing.append("%s / %s" % (cell["column"], cell["state"]))
            continue
        metric = metric or stats[0][0]
        if len(stats) > 1:
            work = stats[1][0].lower()
        rows.append({"column": cell["column"], "value": stats[0][1],
                     "decisions": cell.get("decisions") or 0,
                     "end_reason": cell.get("end_reason")})
    if missing:
        # Loudly. A card that silently averages fewer games than the film
        # showed is worse than no card.
        print("  WARNING: %d of %d games have no numbers and are NOT on the "
              "card: %s" % (len(missing), len(cells), ", ".join(missing)))
    if not rows:
        print("  (no panel numbers on disk: these panels were cached before the "
              "sidecars existed. Delete them to rebuild, or drop --chart)")
        return None
    card = chart_card.build(rows, path, game=game_title(game),
                            metric="%s per game" % metric.title(),
                            decision_word=work, order=order, handle=handle)
    # What the numbers ARE, for anyone downstream that has to say them out
    # loud. Without this the narration called a lines-cleared mean "the
    # average placement count", which is a different number on the same card.
    card.update({"metric_label": metric.lower(), "work_label": work})
    return card


def resolve_column(name, folder):
    """A column is the untrained control, a step count, or a checkpoint file.

    Step counts are the afterstate trainer's naming (ckpt_100000_steps.zip); a
    PPO run names its checkpoints whatever it likes, so a filename or a path is
    accepted as-is."""
    if name == runners.CONTROL:
        return runners.CONTROL, runners.CONTROL
    if name.isdigit():
        return os.path.join(folder, "ckpt_%s_steps.zip" % name), "%dk" % (int(name) // 1000)
    path = name if os.path.isabs(name) else os.path.join(folder, name)
    return path, os.path.splitext(os.path.basename(name))[0]




def panel_key(name, state, no_stats, pad):
    """The crop and the burnt-in counters are baked into a panel, so they
    belong in its NAME: otherwise a --no-stats run silently reuses panels with
    the counters drawn on."""
    return "%s_%s%s%s" % (name, state, "_bare" if no_stats else "",
                          "" if pad == CROP_PAD else "_pad%d" % pad)


def render_one(job):
    """Play one pair and write its mp4 and its sidecar. The unit of work.

    Called in this process when there is one panel to do, and in a worker
    subprocess when there are several -- stable-retro allows ONE emulator per
    process, so panels can only overlap by being separate processes."""
    panel = capture_panel(job["game"], job["checkpoint"], job["state"], job["cap"],
                          job["player"], job["mp4"], job["crop"], job["pad"],
                          not job["no_stats"])
    json.dump({"cap": job["cap"], "player": job["player"], "crop": job["crop"],
               "metrics": METRICS,
               "end_reason": panel["end_reason"], "decisions": panel["decisions"],
               "stats": panel["stats"], "state": job["state"],
               "checkpoint": os.path.basename(job["checkpoint"])},
              open(job["side"], "w"), indent=2)
    return panel


def render_parallel(jobs, workers):
    """Several panels at once, one subprocess each.

    A subprocess per panel rather than a thread: the emulator is one instance
    per PROCESS, and it also means a crash in one game cannot take the rest of
    the run with it. Output is captured and reported per panel as it lands,
    because a dozen emulators writing to one terminal interleaves into noise."""
    started = time.time()
    queue, running, done = list(jobs), [], 0
    print("rendering %d panels, %d at a time" % (len(queue), workers))
    while queue or running:
        while queue and len(running) < workers:
            job = queue.pop(0)
            proc = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--render-one",
                 json.dumps(job)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            running.append((job, proc, time.time()))
        for item in list(running):
            job, proc, began = item
            if proc.poll() is None:
                continue
            running.remove(item)
            done += 1
            out = (proc.stdout.read() or b"").decode("utf-8", "replace")
            if proc.returncode != 0:
                print("  [%d/%d] FAILED %s\n%s"
                      % (done, len(jobs), job["key"], out[-2000:]))
                continue
            side = json.load(open(job["side"])) if os.path.exists(job["side"]) else {}
            print("  [%d/%d] %-26s %s after %s decisions   %.1f min"
                  % (done, len(jobs), job["key"], side.get("end_reason", "?"),
                     side.get("decisions", "?"), (time.time() - began) / 60.0))
        if running:
            time.sleep(0.5)
    print("  all panels done in %.1f min" % ((time.time() - started) / 60.0))


def render_panels(game, columns, rows, cap, player, cache, crop, pad, no_stats,
                  workers=1):
    """Play every (checkpoint, state) pair, reusing anything already rendered.

    The cache is SHARED between runs -- a panel is minutes of emulator time --
    and each run copies the ones it used into its own folder afterwards."""
    os.makedirs(cache, exist_ok=True)
    order, todo = [], []
    for path, name, label in columns:
        for state in rows:
            key = panel_key(name, state, no_stats, pad)
            mp4 = os.path.join(cache, key + ".mp4")
            side = os.path.join(cache, key + ".json")
            settings = {"cap": cap, "player": player, "crop": crop,
                        "metrics": METRICS}
            cached = json.load(open(side)) if os.path.exists(side) else None
            stale = cached and [k for k, v in settings.items()
                                if k in cached and cached[k] != v]
            if cached and cached.get("metrics", 1) < METRICS:
                # Panels measured by an older rule. Version 2 counts the wrap
                # of a fixed-width HUD counter: without it a game that cleared
                # 1057 lines was recorded as 57 and lost a tryout it had won.
                stale = (stale or []) + ["measurement rule"]
            order.append((key, mp4, side, label, state))
            if os.path.exists(mp4) and not os.path.exists(side):
                # An mp4 with no numbers beside it is not a usable panel: it
                # contributes a game to the film and nothing to the card, which
                # is how a five-column chart quietly got built from twelve
                # games instead of fifteen. It is also what a killed render
                # leaves behind, so the video itself may be truncated.
                print("re-rendering %s -- it has no numbers beside it" % key)
            elif os.path.exists(mp4) and not stale:
                print("cached panel %s" % key)
                if cached and not all(k in cached for k in settings):
                    # Panels rendered before these were recorded. The numbers
                    # in them are still whatever rule produced them, and there
                    # is no way to find out from the file.
                    print("   (predates the settings check -- delete it to be sure "
                          "it was played under cap %d, player %d, crop %s)"
                          % (cap, player, crop))
                continue
            if stale:
                # The cap decides when a game is STOPPED and the player decides
                # whose board it is: a panel made under a different one is a
                # different experiment, and reusing it would put two rules in
                # the same grid and the same average.
                print("re-rendering %s -- it was made with a different %s"
                      % (key, ", ".join(stale)))
            todo.append({"key": key, "game": game, "checkpoint": path, "state": state,
                         "cap": cap, "player": player, "mp4": mp4, "side": side,
                         "crop": crop, "pad": pad, "no_stats": bool(no_stats)})

    if len(todo) > 1 and workers > 1:
        render_parallel(todo, min(workers, len(todo)))
    else:
        for job in todo:
            print("rendering panel %s ..." % job["key"])
            panel = render_one(job)
            print("   %s after %d decisions" % (panel["end_reason"], panel["decisions"]))

    panels = {}
    for key, mp4, side, label, state in order:
        if not os.path.exists(mp4):
            sys.exit("panel %s was not rendered -- see the failure above" % key)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
             "stream=width,height", "-of", "csv=p=0", mp4],
            capture_output=True, text=True, check=True).stdout.strip()
        w, h = (int(v) for v in probe.split(",")[:2])
        panel = {"path": mp4, "width": w, "height": h}
        if os.path.exists(side):
            panel.update(json.load(open(side)))
        panel["label"] = panel["column"] = label
        panel["state"] = state
        panels[(label, state)] = panel
    return panels


def slug(text):
    return "".join(c.lower() if c.isalnum() else "-" for c in text).strip("-")


def run_folder(base, game):
    """One self-contained folder per run, like the studio pipeline's.

    Everything that run produced lands in it -- the individual games, the
    card, the silent film, the voice on its own, the combined file and the
    timed script -- so editing one piece never means regenerating the rest."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = os.path.join(base, "%s_%s" % (stamp, slug(game_title(game))))
    os.makedirs(os.path.join(folder, "gameplays"), exist_ok=True)
    return folder


def copy_gameplays(cells, folder):
    """The standalone games, named for what they are rather than for the cache."""
    for cell in cells:
        dest = os.path.join(folder, "gameplays", "%s__%s.mp4"
                            % (slug(cell["column"]), slug(cell["state"])))
        shutil.copyfile(cell["path"], dest)


def say(game, card, states, cap, body_seconds, folder, writer_backend,
        voice_model, voice_name, channel, wpm, take="", card_budget=22.0):
    """Write the script and render it to audio. Returns (body, card, texts).

    Deliberately BEFORE the video is composed: the closing card is then held
    for exactly as long as the part written about it, which is the only way
    the numbers are never spoken over a game still in progress."""
    from tools.progression import narrate

    try:
        algorithm = runners.spec_for(game).algorithm
    except Exception:                                             # noqa: BLE001
        algorithm = ""
    print("  [voice] asking %s for a script ..." % writer_backend)
    script = narrate.write(game_title(game), body_seconds, card, states, cap,
                           algorithm=algorithm, channel=channel, wpm=wpm,
                           card_seconds_budget=card_budget, backend=writer_backend)
    if not script:
        print("  (no script: the writer returned nothing)")
        return None
    json.dump(script, open(os.path.join(folder, "script%s.json" % take), "w"), indent=2)
    return speak_script(script, folder, voice_model, voice_name, take)


def speak_script(script, folder, voice_model, voice_name, take=""):
    """Render a script to audio: the body, then the card part and the closing."""
    print("  [voice] speaking %d body and %d card paragraphs with %s ..."
          % (len(script["body"]), len(script["card"]) + bool(script.get("closing")),
             voice_model))
    blocks = os.path.join(folder, "narration_blocks%s" % take)
    body = speech.speak(script["body"], blocks, backend=voice_model, voice=voice_name)
    card_text = list(script["card"]) + ([script["closing"]] if script.get("closing") else [])
    card_clips = speech.speak(card_text, os.path.join(blocks, "card"),
                              backend=voice_model, voice=voice_name)
    return body, card_clips, list(script["body"]) + card_text


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default=None,
                   help="the integration id, e.g. TetrisTime-Nes-v0. Required")
    p.add_argument("--folder", default=None,
                   help="where the checkpoints are (default: checkpoints/<game>)")
    p.add_argument("--columns", default="control",
                   help="one per column: 'control', a step count, or a checkpoint filename")
    p.add_argument("--states", default=None,
                   help="start states, one per row. Required")
    p.add_argument("--manifest", default=None,
                   help="OPTIONAL. A pre-registered state suite from the evaluation "
                        "side (tools/progression/suite.py). It only supplies defaults "
                        "for --game, --states, --cap and --player, and it is NOT read "
                        "unless you name it -- inheriting a cap from a file nobody "
                        "mentioned is how two stopping rules end up in one grid")
    p.add_argument("--cap", type=int, default=0,
                   help="decisions per game -- placements for Tetris, env steps for a "
                        "policy. 0, the default, means PLAY TO GAME OVER: no budget, "
                        "the game ends when the agent loses. That is the honest "
                        "ending, and it is unbounded -- an agent that does not lose "
                        "does not stop")
    p.add_argument("--player", type=int, default=0, help="which player the agent drives")
    p.add_argument("--crop", choices=("half", "full", "well"), default="half",
                   help="half the screen on the agent's side (default), the whole screen, "
                        "or the declared board alone; a game with no grid is always whole")
    p.add_argument("--pad", type=int, default=CROP_PAD,
                   help="pixels of readouts kept beside the board in --crop half "
                        "(default %(default)s: the narrowest that keeps TetrisTime's "
                        "SCORE / LINES / LEVEL labels whole)")
    p.add_argument("--speed", default=str(SPEED),
                   help="playback speed multiplier, or 'auto' to set it so the gameplay "
                        "lasts exactly as long as the narration written for it "
                        "(default %(default)s)")
    p.add_argument("--labels", default=None,
                   help="what to CALL each column, comma separated, in the order of "
                        "--columns. The default is the checkpoint's own name")
    p.add_argument("--row-labels", default=None,
                   help="what to CALL each row, comma separated, in the order of "
                        "--states. The default is the state's own name; pass an empty "
                        "string to print no row labels at all, which is usually what "
                        "you want -- a start state is an internal name, not something "
                        "the viewer needs")
    p.add_argument("--no-stats", action="store_true",
                   help="no counters burnt into the panels -- just the games and the "
                        "labels you chose. The card still carries the numbers")
    p.add_argument("--cols", type=int, default=0,
                   help="panels per row. The default puts one column per checkpoint, so "
                        "3 checkpoints x 2 states is 3 wide; --cols 4 wraps the same six "
                        "panels 4 then 2, and --cols 2 makes them 2x3")
    p.add_argument("--chart", action="store_true",
                   help="close the film with a stats card built from these games")
    p.add_argument("--chart-seconds", type=float, default=28.0,
                   help="the MAXIMUM the results card is held (default %(default)s). "
                        "With --voice it is held for as long as the words written about "
                        "it, up to this -- anything longer is a still image with a "
                        "voice over it, which is not a video")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="cap the FINISHED file at this many seconds. The card's share "
                        "comes off first and the gameplay speed is solved so the whole "
                        "thing lands on it; the script is written to that length too. "
                        "Off by default, and it overrides --speed when given")
    p.add_argument("--hold", type=float, default=3.0,
                   help="seconds held on the final board, at real speed, after the last "
                        "game ends and before the card (default %(default)s). At 40x, "
                        "the difference between a game ending and the cut is two frames")
    p.add_argument("--title", default=None,
                   help="headline on the stats card (default: the game's name)")
    p.add_argument("--voice", action="store_true",
                   help="write and speak a narration: the game's history, what the agent "
                        "is, and -- only over the card -- what the numbers say")
    p.add_argument("--writer", default="auto",
                   help="which LLM writes the narration "
                        "(auto|claude-code|claude|gemini|ollama)")
    p.add_argument("--voice-model", default=speech.DEFAULT,
                   help="which TTS speaks it: kokoro (default, local, free, one fixed "
                        "voice), qwen (the studio pipeline's), piper, elevenlabs (PAID, "
                        "and it sends the script to a third party)")
    p.add_argument("--voice-name", default=None,
                   help="the speaker: a Kokoro voice such as %s, a Qwen preset, a piper "
                        ".onnx path, or an ElevenLabs voice id"
                        % ", ".join(speech.KOKORO_VOICES[:3]))
    p.add_argument("--wpm", type=int, default=0,
                   help="words per minute used to size the script. The default is the "
                        "measured rate of the voice you picked (kokoro 170, qwen 130), "
                        "because a wrong figure here is silence at the end or lines cut "
                        "off it")
    p.add_argument("--handle", default=None,
                   help="the channel handle burnt into the bottom right of the film "
                        "and onto the results card (default: the watermark in "
                        "studio.json). Pass an empty string to leave it off")
    p.add_argument("--channel", default=None,
                   help="the channel the narration is written for (default: the "
                        "watermark in studio.json)")
    p.add_argument("--catch-up", type=float, default=6.0,
                   help="how much faster the film may run once games start ending "
                        "(default %(default)s). The rate rises in proportion to how "
                        "many panels are still playing, so the last survivor does not "
                        "hold the film open while everything else sits frozen. 1 "
                        "disables it and plays the whole thing at --speed")
    p.add_argument("--still", action="store_true", help="render one frame and stop")
    p.add_argument("--jobs", type=int, default=4,
                   help="how many panels to play AT ONCE (default %(default)s). Each "
                        "one is its own process because stable-retro allows a single "
                        "emulator per process; they are independent games, so this is "
                        "the whole difference between minutes and hours. Raise it "
                        "towards your core count, lower it if memory is tight")
    p.add_argument("--render-one", default=None,
                   help=argparse.SUPPRESS)      # worker mode, not for humans
    p.add_argument("--panels", default=PANELS,
                   help="shared panel cache, reused across runs (default %(default)s)")
    p.add_argument("--out-dir", default=None,
                   help="the run folder (default: a new timestamped one under "
                        "progression_out/runs)")
    p.add_argument("--respeak", default=None, metavar="FOLDER",
                   help="say the SAME script again -- another take, or another voice -- "
                        "and remux. Nothing is replayed and no video is re-encoded")
    p.add_argument("--renarrate", default=None, metavar="FOLDER",
                   help="write a NEW script for a finished run, speak it and remux. "
                        "The gameplay and the card are reused as they are")
    a = p.parse_args()

    if a.render_one:
        # A worker: play exactly one panel and exit. The parent reads the
        # sidecar this writes rather than anything printed here.
        render_one(json.loads(a.render_one))
        return
    if a.respeak or a.renarrate:
        return regenerate(a)

    manifest = json.load(open(a.manifest)) if a.manifest else {}
    game = a.game or manifest.get("game")
    if not game:
        sys.exit("no game: pass --game")
    rows = (a.states.split(",") if a.states
            else [s["name"] for s in manifest.get("states", [])[:2]])
    if not rows:
        sys.exit("no start states: pass --states")
    cap = a.cap or int(manifest.get("placement_cap", 0))
    player = a.player or int(manifest.get("player", 0))
    folder = a.folder or os.path.join(ROOT, "checkpoints", game)
    a.wpm = a.wpm or speech.words_per_minute(a.voice_model)
    auto_speed = str(a.speed).lower() == "auto"
    speed = SPEED if auto_speed else float(a.speed)

    names = a.columns.split(",")
    given = a.labels.split(",") if a.labels else []
    if given and len(given) != len(names):
        sys.exit("--labels has %d names for %d columns" % (len(given), len(names)))
    if a.row_labels is None:
        row_labels = list(rows)
    elif a.row_labels.strip():
        row_labels = [r.strip() for r in a.row_labels.split(",")]
        if len(row_labels) != len(rows):
            sys.exit("--row-labels has %d names for %d states"
                     % (len(row_labels), len(rows)))
    else:
        row_labels = []
    # Everything a model or a GPU could refuse is checked BEFORE the emulator
    # runs: a missing voice must never cost an hour of replay.
    if a.voice and not a.still and not speech.available(a.voice_model):
        sys.exit("voice backend %r is not installed here.\n"
                 "kokoro:  pip install \"kokoro>=0.9.4\" soundfile   "
                 "(and: sudo apt install espeak-ng)" % a.voice_model)

    columns = []
    for i, spec_name in enumerate(names):
        path, name = resolve_column(spec_name, folder)
        columns.append((path, name, given[i].strip() if given else HEADINGS.get(name, name)))
    labels = [label for _p, _n, label in columns]

    panels = render_panels(game, columns, rows, cap, player, a.panels,
                           a.crop, a.pad, a.no_stats, a.jobs)
    cells = [panels[(c, r)] for r in rows for c in labels]
    ncols = a.cols or len(labels)
    if a.cols and len(rows) > 1:
        # Once the panels wrap, "this column is 100k" and "this row is state 2"
        # stop being true, so each panel has to say what it is itself -- with
        # the row's name only if there is one to say.
        by_state = dict(zip(rows, row_labels))
        for cell in cells:
            name = by_state.get(cell["state"], "")
            cell["label"] = ("%s  %s" % (cell["column"], name)) if name else cell["column"]
    headers = (labels, row_labels) if not a.cols else None

    watermark = json.load(
        open(os.path.join(ROOT, "studio.json"))).get("watermark", "")
    handle = watermark if a.handle is None else a.handle

    if a.still:
        still_path = os.path.join(OUT, "progression_still.png")
        # The still exists to preview the film, so it is composed with the
        # SAME plan -- otherwise its footer quotes a speed the film will not
        # actually run at, and its panels are not dimmed where the film's are.
        compose(cells, still_path, ncols=ncols, headers=headers, still=True,
                speed=speed, catch_up=a.catch_up, hold=a.hold, handle=handle)
        print("wrote %s" % still_path)
        return

    out_dir = a.out_dir or run_folder(os.path.join(OUT, "runs"), game)
    os.makedirs(os.path.join(out_dir, "gameplays"), exist_ok=True)
    copy_gameplays(cells, out_dir)

    card = None
    if a.chart:
        card = build_chart(cells, a.title or game,
                           os.path.join(out_dir, "chart.png"), labels, handle)

    lengths = [duration(c["path"]) for c in cells]
    # At speed 1 this is what the catch-up plan adds up to for the GAMES, and
    # every other speed divides it -- which is what makes a target length
    # solvable in one step. The hold is deliberately excluded: it plays at real
    # time and does not scale, so dividing it by the speed made a film asked to
    # be 60 seconds come out 62.4.
    unit_seconds = rate_plan(lengths, 1.0, a.catch_up, 0.0)[1]
    chart_seconds = a.chart_seconds if card else 0.0
    spoken = None
    if a.voice:
        channel = a.channel or handle or watermark
        # What the script is written TO: the length the gameplay will run.
        # With --max-seconds that is decided in advance, so the words are
        # sized for it rather than for a speed that is about to change.
        body_target = (max(2.0, a.max_seconds - a.chart_seconds) if a.max_seconds
                       else unit_seconds / speed + a.hold)
        spoken = say(game, card, rows, cap, body_target, out_dir, a.writer,
                     a.voice_model, a.voice_name, channel, a.wpm,
                     card_budget=a.chart_seconds)
    if spoken:
        from tools.progression import narrate
        body, card_clips, _texts = spoken
        body_seconds = (sum(s for _p, s in body)
                        + narrate.BLOCK_GAP * max(0, len(body) - 1) + 0.6)
        if auto_speed and not a.max_seconds:
            # The GAMEPLAY is cut to the narration, not the other way round.
            # This is what removes the long silences -- spacing sentences
            # further apart to cover the gap is what made them unnatural.
            speed = max(1.0, unit_seconds / max(1.0, body_seconds - a.hold))
            print("  [voice] speed set to %.2fx, so the games last %s -- the length "
                  "of the spoken part" % (speed, narrate.clock(body_seconds)))
        elif not a.max_seconds and body_seconds > unit_seconds / speed + a.hold:
            print("  (the spoken part runs %s but the games last %s at %gx, so the "
                  "end will be cut. --speed auto fits them to each other)"
                  % (narrate.clock(body_seconds),
                     narrate.clock(unit_seconds / speed + a.hold), speed))
        if card:
            # A CEILING on how much is SAID over the card, not a guillotine
            # across the end of it. One run held the card 108 seconds because
            # the writer had 108 seconds of things to say; the next capped the
            # hold at 22 and dropped every card block for not fitting, so the
            # winner was never announced. fit_card keeps what fits, always
            # keeps the closing, and the hold is then exactly what those need.
            kept, index, chart_seconds = narrate.fit_card(card_clips, a.chart_seconds)
            if len(kept) < len(card_clips):
                print("  (voice: %d of %d card lines did not fit the %gs card)"
                      % (len(card_clips) - len(kept), len(card_clips), a.chart_seconds))
            body_n = len(body)
            texts = list(_texts[:body_n]) + [_texts[body_n + i] for i in index]
            card_clips = kept
            spoken = (body, card_clips, texts)

    if a.max_seconds:
        # The FINISHED FILE is what is being limited, so the card's share comes
        # off first and the games are fitted into what is left. Solved after
        # the card's real length is known, which is why this is not simply
        # another --speed.
        target = max(2.0, a.max_seconds - chart_seconds - a.hold)
        speed = max(1.0, unit_seconds / target)
        if unit_seconds / speed < target - 1.0:
            print("  (the games only fill %s of the %s asked for -- they are not "
                  "long enough, and slower than real time is not on offer)"
                  % (_clock(unit_seconds / speed + a.hold), _clock(a.max_seconds)))
        else:
            print("  length capped at %s: %sx speed for %s of games, plus %s of card"
                  % (_clock(a.max_seconds), rate_text(speed),
                     _clock(unit_seconds / speed + a.hold), _clock(chart_seconds)))

    grid_seconds = rate_plan(lengths, speed, a.catch_up, a.hold)[1]
    silent = os.path.join(out_dir, "grid.mp4")
    compose(cells, silent, ncols=ncols, headers=headers, speed=speed,
            chart=card and card["path"], chart_seconds=chart_seconds,
            catch_up=a.catch_up, hold=a.hold, handle=handle)
    print("wrote %s" % silent)

    if spoken:
        finish_voice(out_dir, silent, spoken, grid_seconds)
    json.dump({"game": game, "columns": [list(c) for c in columns], "states": rows,
               "cap": cap, "player": player, "speed": speed, "crop": a.crop,
               "pad": a.pad, "no_stats": bool(a.no_stats), "cols": ncols,
               "chart": bool(card), "chart_seconds": chart_seconds,
               "catch_up": a.catch_up, "hold": a.hold, "handle": handle,
               "grid_seconds": grid_seconds, "title": a.title,
               "channel": a.channel, "voice_model": a.voice_model,
               "voice_name": a.voice_name, "writer": a.writer, "wpm": a.wpm,
               "panels": a.panels},
              open(os.path.join(out_dir, "run.json"), "w"), indent=2)
    print("\nrun folder: %s" % out_dir)


def next_take(out_dir):
    """Takes are NUMBERED, not overwritten.

    Re-speaking is meant to be done repeatedly -- another voice, another
    reading -- and comparing takes is the entire point, so take 2 must not
    destroy take 1. The first is unsuffixed; after that it is _take2, _take3."""
    if not os.path.exists(os.path.join(out_dir, "narration.wav")):
        return ""
    n = 2
    while os.path.exists(os.path.join(out_dir, "narration_take%d.wav" % n)):
        n += 1
    return "_take%d" % n


def finish_voice(out_dir, silent, spoken, grid_seconds, take=""):
    """Lay the spoken blocks over the film and write the timed script."""
    from tools.progression import narrate
    body, card_clips, texts = spoken
    total = duration(silent)
    placed = narrate.schedule(body, card_clips, grid_seconds)
    kept = [b for b in placed if b[1] <= total + 0.05]
    if len(kept) < len(placed):
        print("  (voice: %d block%s ran past the end of the film and were dropped. "
              "--renarrate asks for a shorter script; the film itself is unchanged)"
              % (len(placed) - len(kept), "" if len(placed) - len(kept) == 1 else "s"))
    wav = os.path.join(out_dir, "narration%s.wav" % take)
    narrate.build_track(kept, total, wav)
    narrate.mux(silent, wav, os.path.join(out_dir, "narrated%s.mp4" % take))
    narrate.save_schedule(kept, texts, out_dir, "grid.mp4", total, grid_seconds, take)
    for name in ("narration%s.wav", "narrated%s.mp4", "narration%s.txt"):
        print("wrote %s" % os.path.join(out_dir, name % take))


def regenerate(a):
    """--respeak / --renarrate: redo the audio for a finished run.

    The film and the card are left exactly as they are, so this costs one
    model call and one TTS pass -- no emulator, no video encode. The card's
    length is already fixed by the first run, so a much longer new script can
    overrun it; anything that does not fit is dropped and reported."""
    out_dir = a.respeak or a.renarrate
    run = json.load(open(os.path.join(out_dir, "run.json")))
    take = next_take(out_dir)
    silent = os.path.join(out_dir, "grid.mp4")
    grid_seconds = float(run["grid_seconds"])
    voice_model = a.voice_model or run.get("voice_model") or speech.DEFAULT
    voice_name = a.voice_name or run.get("voice_name")
    a.wpm = a.wpm or speech.words_per_minute(voice_model)
    if not speech.available(voice_model):
        sys.exit("voice backend %r is not installed here." % voice_model)

    if a.renarrate:
        cache = run.get("panels") or PANELS
        cells = []
        for _path, name, label in [tuple(c) for c in run["columns"]]:
            for state in run["states"]:
                side = os.path.join(cache, panel_key(name, state, run["no_stats"],
                                                     run["pad"]) + ".json")
                cells.append(dict(json.load(open(side)), column=label, state=state))
        card = build_chart(cells, run.get("title") or run["game"],
                           os.path.join(out_dir, "chart.png"),
                           [c[2] for c in run["columns"]]) if run.get("chart") else None
        spoken = say(run["game"], card, run["states"], run["cap"], grid_seconds,
                     out_dir, a.writer, voice_model, voice_name,
                     a.channel or run.get("channel") or "", a.wpm, take,
                     card_budget=run.get("chart_seconds") or a.chart_seconds)
        if not spoken:
            sys.exit("the writer returned nothing; the existing film is untouched")
    else:
        # The SAME words, said again: another take, or another voice. The
        # script is read back from the run rather than rewritten, so nothing
        # about the wording can change underneath you -- and it is the NEWEST
        # script, so re-speaking after a --renarrate says the new words rather
        # than silently resurrecting the first ones.
        scripts = sorted(f for f in os.listdir(out_dir)
                         if f.startswith("script") and f.endswith(".json"))
        if not scripts:
            sys.exit("no script.json in %s -- that run was rendered without --voice, "
                     "so there is nothing to say again. Use --renarrate." % out_dir)
        print("  [voice] re-speaking %s" % scripts[-1])
        spoken = speak_script(json.load(open(os.path.join(out_dir, scripts[-1]))),
                              out_dir, voice_model, voice_name, take)

    finish_voice(out_dir, silent, spoken, grid_seconds, take)
    print("\nrun folder: %s" % out_dir)


if __name__ == "__main__":
    main()
