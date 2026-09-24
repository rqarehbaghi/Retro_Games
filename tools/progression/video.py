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
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.progression import runners                             # noqa: E402

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

    def __init__(self, path, crop, grid, pad=CROP_PAD):
        self.path, self.crop, self.grid, self.pad = path, crop, grid, pad
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
        # line is wider than the panel and lost its end.
        for i, (label, value) in enumerate(state.get("stats") or []):
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


def capture_panel(game, checkpoint, state, cap, player, path, crop, pad=CROP_PAD):
    """Play one (checkpoint, state) pair and write its panel to `path`."""
    spec = runners.spec_for(game, player, state)
    writer = PanelWriter(path, crop, spec.grid, pad)
    result = runners.run(game, checkpoint, state, cap, player, writer)
    panel = writer.finish(result["last"], {"stats": result["stats"]}, result["end_reason"])
    panel.update({"end_reason": result["end_reason"], "decisions": result["decisions"],
                  "stats": result["stats"]})
    return panel


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def compose(panels, columns, rows, out_path, still=False, still_at=8.0, speed=SPEED):
    """Every panel into one 1920x1080 frame, each frozen once its game ends."""
    ncols, nrows = len(columns), len(rows)
    longest = max(duration(p["path"]) for p in panels.values())
    cell_w, cell_h = 1920 // ncols, (1080 - 120) // nrows
    first = panels[(columns[0], rows[0])]
    pw = min(cell_w - 40, int((cell_h - 34) * first["width"] / first["height"]))
    ph = int(pw * first["height"] / first["width"])

    order = [(c, r) for r in rows for c in columns]
    inputs, filters, overlays = [], [], "[bg]"
    for i, key in enumerate(order):
        panel = panels[key]
        inputs += ["-i", panel["path"]]
        pad = max(0.0, longest - duration(panel["path"]))
        filters.append(
            "[%d:v]setpts=PTS/%s,tpad=stop_mode=clone:stop_duration=%.3f,"
            "scale=%d:%d:flags=neighbor[p%d]" % (i, speed, pad / speed, pw, ph, i))
    for i in range(len(order)):
        cx = (i % ncols) * cell_w + (cell_w - pw) // 2
        cy = 96 + (i // ncols) * cell_h + 10
        nxt = "[s%d]" % i if i < len(order) - 1 else "[vout]"
        overlays += "[p%d]overlay=%d:%d%s;" % (i, cx, cy, nxt)
        if i < len(order) - 1:
            overlays += "[s%d]" % i
    # The background MUST carry a duration. color= is an infinite source, and
    # overlaying onto it made the render run for as long as ffmpeg was left
    # alive -- a 29-minute game at 3x produced 6776 seconds of output and was
    # mistaken for a slow encode rather than an unbounded one.
    out_seconds = longest / speed
    graph = (";".join(filters) + ";"
             + "color=c=black:s=1920x1080:r=60:d=%.3f[bg];" % out_seconds + overlays)
    graph = graph.rstrip(";")
    # Headers go on the composite, not inside the panels: a checkpoint name is
    # wider than one board. Same house style as every other overlay in this
    # repo -- black on an opaque white plate.
    path_font = font_file()
    labels = [(HEADINGS.get(col, col), (i % ncols) * cell_w + cell_w // 2, 34, 26)
              for i, col in enumerate(columns)]
    labels += [(row, 1920 // 2, 96 + j * cell_h - 26, 20) for j, row in enumerate(rows)]
    labels.append(("%gx speed" % speed, 1920 // 2, 1050, 16))
    text = ""
    for k, (body, cx, cy, size) in enumerate(labels):
        src = "[vout]" if k == 0 else "[t%d]" % k
        dst = "[t%d]" % (k + 1) if k < len(labels) - 1 else "[final]"
        text += ("%sdrawtext=%stext='%s':fontcolor=black:fontsize=%d:box=1:boxcolor=white:"
                 "boxborderw=8:x=%d-text_w/2:y=%d%s;"
                 % (src, ("fontfile=%s:" % path_font) if path_font else "",
                    body.replace(":", "\\:").replace("'", ""), size, cx, cy, dst))
    graph += ";" + text.rstrip(";")

    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error"] + inputs + \
          ["-filter_complex", graph, "-map", "[final]", "-an"]
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

    os.makedirs(a.panels, exist_ok=True)
    columns, panels = [], {}
    for spec_name in a.columns.split(","):
        path, name = resolve_column(spec_name, folder)
        columns.append(name)
        for state in rows:
            key = "%s_%s" % (name, state)
            mp4 = os.path.join(a.panels, key + ".mp4")
            if os.path.exists(mp4):
                print("cached panel %s" % key)
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                     "stream=width,height", "-of", "csv=p=0", mp4],
                    capture_output=True, text=True, check=True).stdout.strip()
                w, h = (int(v) for v in probe.split(",")[:2])
                panels[(name, state)] = {"path": mp4, "width": w, "height": h}
            else:
                print("rendering panel %s ..." % key)
                panels[(name, state)] = capture_panel(
                    game, path, state, cap, player, mp4, a.crop, a.pad)
                print("   %s after %d decisions" % (panels[(name, state)]["end_reason"],
                                                    panels[(name, state)]["decisions"]))

    out = a.out or os.path.join(
        OUT, "progression_still.png" if a.still else "progression_grid.mp4")
    compose(panels, columns, rows, out, still=a.still, speed=a.speed)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
