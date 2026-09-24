#!/usr/bin/env python3
"""Render a grid of agents playing: N checkpoints across, M start states down.

    # Tetris: three checkpoints, two states
    python tools/progression/video.py --game TetrisTime-Nes-v0 \
        --folder checkpoints/tetris_v54 --columns 5000,100000,200000 \
        --states p1_01,p1_02 --manifest progression_out/suite_1p/suite_manifest.json

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
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.progression import chart as chart_card, runners        # noqa: E402

OUT = os.path.join(ROOT, "progression_out")
PANELS = os.path.join(OUT, "panels")
SPEED = 3.0
HEADINGS = {"control": "Untrained control"}
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
        out += ("%sdrawtext=%stext='%s':fontcolor=black:fontsize=%d:box=1:boxcolor=white:"
                "boxborderw=8:x=%d-text_w/2:y=%d%s;"
                % (src, ("fontfile=%s:" % path) if path else "",
                   body.replace(":", "\\:").replace("'", ""), size, cx, cy, dst))
        src = dst
    return out.rstrip(";")


def compose(cells, out_path, ncols=None, headers=None, still=False, still_at=8.0,
            speed=SPEED, chart=None, chart_seconds=12.0):
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

    inputs, filters, overlays = [], [], "[bg]"
    for i, cell in enumerate(cells):
        inputs += ["-i", cell["path"]]
        pad = max(0.0, longest - duration(cell["path"]))
        filters.append(
            "[%d:v]setpts=PTS/%s,tpad=stop_mode=clone:stop_duration=%.3f,"
            "scale=%d:%d:flags=neighbor[p%d]" % (i, speed, pad / speed, pw, ph, i))
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
    # mistaken for a slow encode rather than an unbounded one.
    out_seconds = longest / speed
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
    labels.append(("%gx speed" % speed, 1920 // 2, 1054, 16))
    graph += ";" + drawtext("[vout]", labels)

    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error"] + inputs
    last = "[final]"
    if chart and not still:
        # The chart is a still image held for a few seconds, concatenated onto
        # the end in the SAME pass -- writing the grid and then re-encoding it
        # to append a tail would cost a second full encode of the whole video.
        inputs_len = len(cells)
        cmd += ["-loop", "1", "-t", "%.3f" % chart_seconds, "-i", chart]
        # concat refuses inputs that disagree on size, pixel format, sample
        # aspect or rate, and it fails at RUN time rather than when the graph
        # is built -- so both sides are pinned explicitly.
        graph += (";[final]fps=60,setsar=1,format=yuv420p[gridv];"
                  "[%d:v]scale=1920:1080,fps=60,setsar=1,format=yuv420p,"
                  "fade=in:st=0:d=0.5[chartv];"
                  "[gridv][chartv]concat=n=2:v=1:a=0[outv]" % inputs_len)
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


def build_chart(cells, game, path, order):
    """The closing card, built from the panels' own sidecar numbers.

    The headline is whatever the game reports FIRST -- lines for Tetris, score
    for Mario -- and the second figure is the work it took, which is placements
    for one algorithm and env steps for the other. Neither is named here."""
    rows, metric, work = [], "", "decisions"
    for cell in cells:
        stats = cell.get("stats") or []
        if not stats:
            continue
        metric = metric or stats[0][0]
        if len(stats) > 1:
            work = stats[1][0].lower()
        rows.append({"column": cell["column"], "value": stats[0][1],
                     "decisions": cell.get("decisions") or 0,
                     "end_reason": cell.get("end_reason")})
    if not rows:
        print("  (no panel numbers on disk: these panels were cached before the "
              "sidecars existed. Delete them to rebuild, or drop --chart)")
        return None
    return chart_card.build(rows, path, game=game,
                            metric="%s per game" % metric.title(),
                            decision_word=work, order=order)


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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default=None, help="default: the manifest's game")
    p.add_argument("--folder", default=None,
                   help="where the checkpoints are (default: checkpoints/<game>)")
    p.add_argument("--columns", default="control",
                   help="one per column: 'control', a step count, or a checkpoint filename")
    p.add_argument("--states", default=None,
                   help="one per row (default: the first two in the manifest)")
    p.add_argument("--manifest", default=os.path.join(OUT, "suite_manifest.json"),
                   help="a generated state suite; optional if --states and --game are given")
    p.add_argument("--cap", type=int, default=0,
                   help="decisions per game -- placements, or env steps for a policy")
    p.add_argument("--player", type=int, default=0, help="which player the agent drives")
    p.add_argument("--crop", choices=("half", "full", "well"), default="half",
                   help="half the screen on the agent's side (default), the whole screen, "
                        "or the declared board alone; a game with no grid is always whole")
    p.add_argument("--pad", type=int, default=CROP_PAD,
                   help="pixels of readouts kept beside the board in --crop half "
                        "(default %(default)s: the narrowest that keeps TetrisTime's "
                        "SCORE / LINES / LEVEL labels whole)")
    p.add_argument("--speed", type=float, default=SPEED,
                   help="playback speed multiplier (default %(default)s). The grid lasts as "
                        "long as its LONGEST game: a checkpoint that plays 29 minutes makes "
                        "a 10-minute video at 3x, so raise this for long games")
    p.add_argument("--labels", default=None,
                   help="what to CALL each column, comma separated, in the order of "
                        "--columns. The default is the checkpoint's own name")
    p.add_argument("--no-stats", action="store_true",
                   help="no counters burnt into the panels -- just the games and the "
                        "labels you chose. The end chart still carries the numbers")
    p.add_argument("--cols", type=int, default=0,
                   help="panels per row. The default puts one column per checkpoint, so "
                        "3 checkpoints x 2 states is 3 wide; --cols 4 wraps the same six "
                        "panels 4 then 2, and --cols 2 makes them 2x3")
    p.add_argument("--chart", action="store_true",
                   help="close the video with a stats card built from these games")
    p.add_argument("--chart-seconds", type=float, default=12.0)
    p.add_argument("--voice", action="store_true",
                   help="write and speak a narration over the video: the game's history, "
                        "how the agent was trained, and the numbers on the chart. OFF by "
                        "default, like the studio pipeline's")
    p.add_argument("--writer", default="auto",
                   help="which LLM writes the narration (auto|claude-code|claude|gemini|ollama)")
    p.add_argument("--still", action="store_true", help="render one frame and stop")
    p.add_argument("--panels", default=PANELS, help="where panel mp4s are cached")
    p.add_argument("--out", default=None,
                   help="default: progression_out/progression_grid.mp4, or "
                        "progression_still.png with --still")
    a = p.parse_args()

    manifest = json.load(open(a.manifest)) if os.path.exists(a.manifest) else {}
    game = a.game or manifest.get("game")
    if not game:
        sys.exit("no game: pass --game, or a --manifest that names one")
    rows = (a.states.split(",") if a.states
            else [s["name"] for s in manifest.get("states", [])[:2]])
    if not rows:
        sys.exit("no start states: pass --states, or a --manifest that lists them")
    cap = a.cap or int(manifest.get("placement_cap", 0)) or 500
    player = a.player or int(manifest.get("player", 0))
    folder = a.folder or os.path.join(ROOT, "checkpoints", game)

    names = a.columns.split(",")
    given = a.labels.split(",") if a.labels else []
    if given and len(given) != len(names):
        sys.exit("--labels has %d names for %d columns" % (len(given), len(names)))

    os.makedirs(a.panels, exist_ok=True)
    columns, panels = [], {}
    for i, spec_name in enumerate(names):
        path, name = resolve_column(spec_name, folder)
        label = given[i].strip() if given else HEADINGS.get(name, name)
        columns.append(label)
        for state in rows:
            # The crop and the burnt-in counters are baked into a panel, so
            # they belong in its NAME: otherwise a --no-stats run silently
            # reuses panels that have the counters drawn on.
            key = "%s_%s%s%s" % (name, state, "_bare" if a.no_stats else "",
                                 "" if a.pad == CROP_PAD else "_pad%d" % a.pad)
            mp4 = os.path.join(a.panels, key + ".mp4")
            side = os.path.join(a.panels, key + ".json")
            if os.path.exists(mp4):
                print("cached panel %s" % key)
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                     "stream=width,height", "-of", "csv=p=0", mp4],
                    capture_output=True, text=True, check=True).stdout.strip()
                w, h = (int(v) for v in probe.split(",")[:2])
                panel = {"path": mp4, "width": w, "height": h}
                if os.path.exists(side):
                    panel.update(json.load(open(side)))
            else:
                print("rendering panel %s ..." % key)
                panel = capture_panel(game, path, state, cap, player, mp4,
                                      a.crop, a.pad, not a.no_stats)
                json.dump({"end_reason": panel["end_reason"],
                           "decisions": panel["decisions"], "stats": panel["stats"],
                           "checkpoint": os.path.basename(path), "state": state},
                          open(side, "w"), indent=2)
                print("   %s after %d decisions" % (panel["end_reason"], panel["decisions"]))
            panel["label"] = panel["column"] = label
            panel["state"] = state
            panels[(label, state)] = panel

    # Reading order: each state's row of checkpoints, then the next state's.
    # --cols only decides where that sequence wraps.
    cells = [panels[(c, r)] for r in rows for c in columns]
    ncols = a.cols or len(columns)
    if a.cols and len(rows) > 1:
        # Once the panels wrap, "this column is 100k" and "this row is state
        # 2" stop being true, so each panel has to say what it is itself.
        for cell in cells:
            cell["label"] = "%s  %s" % (cell["label"], cell["state"])
    headers = (columns, rows) if not a.cols else None

    out = a.out or os.path.join(
        OUT, "progression_still.png" if a.still else "progression_grid.mp4")
    card = None
    if a.chart and not a.still:
        card = build_chart(cells, game, os.path.join(OUT, "progression_chart.png"), columns)
        if card:
            print("wrote %s" % card["path"])
    compose(cells, out, ncols=ncols, headers=headers, still=a.still, speed=a.speed,
            chart=card and card["path"], chart_seconds=a.chart_seconds)
    print("wrote %s" % out)

    # Speech is LAST and never fatal: the video already exists by this point,
    # and a model or a GPU being unavailable must not cost the render.
    if a.voice and not a.still:
        from tools.progression import narrate
        try:
            spoken = narrate.add(out, game, card, backend=a.writer, out_dir=OUT,
                                 states=rows, cap=cap,
                                 chart_seconds=a.chart_seconds if card else 0.0)
            if spoken:
                print("wrote %s" % spoken)
        except Exception as exc:                                  # noqa: BLE001
            print("  (narration failed, the silent video is unaffected: %s)" % exc)


if __name__ == "__main__":
    main()
