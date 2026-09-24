#!/usr/bin/env python3
"""Render the 3x2 progression grid: three checkpoints, two states, side by side.

    python tools/progression/video.py --still      # one frame, to look at first
    python tools/progression/video.py              # the full grid

The video is an ILLUSTRATION of the sweep, never the evidence: which states it
shows was decided from the zero-value control alone (tools/progression/select.py)
before any checkpoint ran, and the numbers that support any claim come from
results.json over the whole suite.

Panels are the agent's own well, cropped out of the two-player screen. The
agent was trained as player 2 on that screen, so evaluating it there keeps the
measurement in distribution; cropping is presentation only and also removes the
idle half-screen.
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

from tools.progression import ids, report                        # noqa: E402

OUT = os.path.join(ROOT, "progression_out")
PANELS = os.path.join(OUT, "panels")
SPEED = 3.0
HEADINGS = {"control": "Zero-value control"}
PANEL_SCALE = 3                      # nearest-neighbour upscale of the cropped well
MARGIN = 6                           # game pixels kept around the well


class FrameTap:
    """Wraps the raw retro env so EVERY emulator frame is seen.

    The training env steps the emulator internally while skipping transition
    animations, and those frames never come back from env.step -- so a video
    built from step returns would jump over every level change. Tapping the
    inner env catches them."""

    def __init__(self, inner, sink):
        self._inner, self._sink = inner, sink

    def step(self, action):
        out = self._inner.step(action)
        self._sink(out[0])
        return out

    def reset(self, **kwargs):
        out = self._inner.reset(**kwargs)
        self._sink(out[0])
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def house_font(size):
    for path in (os.path.expanduser("~/.local/share/fonts/PressStart2P-Regular.ttf"),
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:                                     # noqa: BLE001
                pass
    return ImageFont.load_default()


def plate(draw, xy, text, font, pad=4):
    """Black on an opaque white plate -- the house rule for every overlay."""
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    draw.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill=(255, 255, 255))
    draw.text((x, y), text, font=font, fill=(0, 0, 0))


def capture_panel(checkpoint, state, cap, player, path, label):
    """Play one pair and write its well, with burnt-in counters, to `path`.

    Counters are drawn per frame here rather than as ffmpeg drawtext filters: a
    trajectory has hundreds of placements, and one enable-window filter per
    change would be an unusable filter graph."""
    import custom_integrations                                    # noqa: F401
    from rl.env import make_env, TrainingSpec
    from rl.vars import GameVars
    from rl.simulators import get_simulator
    from rl.afterstate import AfterstateAgent

    overrides = {"player": player, "state": state}
    spec = TrainingSpec(json.load(open(os.path.join(OUT, "suite_manifest.json")))["game"],
                        overrides)
    cfg = spec.afterstate_config
    sim = get_simulator(cfg.get("simulator") or spec.features_name or spec.game, config=cfg)
    agent = AfterstateAgent(input_dim=sim.feature_dim, device="cpu",
                            model_type=cfg.get("model_type", "mlp"),
                            zero_init_value=(checkpoint == "control"))
    if checkpoint != "control":
        agent.load(checkpoint)
    gv = GameVars(spec.game, entry=spec.entry)
    lines_var = cfg.get("lines_var", "lines_p{player}").replace("{player}", str(player))

    grid = spec.grid
    x0 = max(0, int(grid["x"]) - MARGIN)
    x1 = min(256, int(grid["x"]) + int(grid["cols"]) * int(grid["cell"]) + MARGIN)
    y0 = max(0, int(grid["y"]) - 2 * int(grid["cell"]))
    y1 = min(240, int(grid["y"]) + int(grid["rows"]) * int(grid["cell"]) + MARGIN)
    w, h = (x1 - x0) * PANEL_SCALE, (y1 - y0) * PANEL_SCALE
    font = house_font(max(8, w // 14))

    ff = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "%dx%d" % (w, h), "-r", "60", "-i", "-", "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", path],
        stdin=subprocess.PIPE)

    state_box = {"lines": 0, "placements": 0, "end": None}

    def write(frame):
        # Only the counters go INSIDE the panel, and abbreviated: the well is
        # ~276px wide, so a label like "Zero-value control" simply runs off the
        # edge. Checkpoint and state names are drawn as headers by compose(),
        # where there is room for them.
        panel = Image.fromarray(np.asarray(frame)[y0:y1, x0:x1]).resize(
            (w, h), Image.NEAREST)
        draw = ImageDraw.Draw(panel)
        plate(draw, (6, 4), "L %d  P %d" % (state_box["lines"], state_box["placements"]), font)
        if state_box["end"]:
            plate(draw, (6, h - font.size - 14), state_box["end"], font)
        ff.stdin.write(panel.tobytes())

    env = make_env(spec.game, overrides)
    core = env.unwrapped
    core.env = FrameTap(core.env, write)
    obs, info = env.reset()
    lines0 = int(info.get(lines_var, 0) or 0)
    pending, end = False, "placement_cap"
    while state_box["placements"] < cap:
        action = None
        if not pending:
            cands = sim.get_candidates(obs=obs, ram=getattr(core, "ram", None),
                                       info=info, vars=gv, spec=spec)
            if cands:
                action = agent.select_action(cands, epsilon=0.0)[0]
                pending = True
                state_box["placements"] += 1
        obs, _r, term, trunc, info = env.step(action)
        state_box["lines"] = int(info.get(lines_var, 0) or 0) - lines0
        if info.get("afterstate_ready", True):
            pending = False
        if term:
            end = "game_over"
            break
        if trunc:
            end = "truncated"
            break
    # A few frames of the final board carrying the end label, so the freeze
    # that follows in the composite says WHY it stopped.
    # Short enough to fit the panel, and the two outcomes stay distinct: a
    # capped trajectory was censored, not finished.
    state_box["end"] = {"game_over": "GAME OVER",
                        "placement_cap": "CAP REACHED",
                        "truncated": "TRUNCATED"}[end]
    last = core.env._inner.render() if hasattr(core.env, "_inner") else None
    for _ in range(30):
        write(last if last is not None else obs)
    env.close()
    ff.stdin.close()
    ff.wait()
    return {"end_reason": end, "delta_lines": state_box["lines"],
            "placements": state_box["placements"], "path": path,
            "width": w, "height": h}


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def compose(panels, columns, rows, out_path, still=False, still_at=8.0):
    """Six panels into 1920x1080, each frozen on its last frame once it ends."""
    longest = max(duration(p["path"]) for p in panels.values())
    cell_w, cell_h = 1920 // 3, (1080 - 120) // 2
    pw = min(cell_w - 40, int((cell_h - 34) * panels[(columns[0], rows[0])]["width"]
                              / panels[(columns[0], rows[0])]["height"]))
    ph = int(pw * panels[(columns[0], rows[0])]["height"]
             / panels[(columns[0], rows[0])]["width"])
    font_head = 26

    inputs, filters, overlays = [], [], "[bg]"
    for i, ((col, row), panel) in enumerate(
            [((c, r), panels[(c, r)]) for r in rows for c in columns]):
        inputs += ["-i", panel["path"]]
        pad = max(0.0, longest - duration(panel["path"]))
        filters.append(
            "[%d:v]setpts=PTS/%s,tpad=stop_mode=clone:stop_duration=%.3f,"
            "scale=%d:%d:flags=neighbor[p%d]" % (i, SPEED, pad / SPEED, pw, ph, i))
    for i, (col, row) in enumerate([(c, r) for r in rows for c in columns]):
        cx = (i % 3) * cell_w + (cell_w - pw) // 2
        cy = 96 + (i // 3) * cell_h + 10
        nxt = "[s%d]" % i if i < len(panels) - 1 else "[vout]"
        overlays += "[p%d]overlay=%d:%d%s;" % (i, cx, cy, nxt)
        if i < len(panels) - 1:
            overlays += "[s%d]" % i
    graph = ";".join(filters) + ";" + "color=c=black:s=1920x1080:r=60[bg];" + overlays
    graph = graph.rstrip(";")
    # Headers go on the composite, not inside the panels: a checkpoint name is
    # wider than the well. Same house style as every other overlay here --
    # black on an opaque white plate.
    font_file = None
    for candidate in (os.path.expanduser("~/.local/share/fonts/PressStart2P-Regular.ttf"),
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(candidate):
            font_file = candidate
            break
    labels = []
    for i, col in enumerate(columns):
        labels.append((HEADINGS.get(col, col), (i % 3) * cell_w + cell_w // 2, 34, font_head))
    for j, row in enumerate(rows):
        tag = "median-difficulty state" if j == 0 else "hard state"
        labels.append(("%s  (%s)" % (row, tag), 1920 // 2, 96 + j * cell_h - 26, 20))
    labels.append(("%gx speed   -   states chosen from the control alone"
                   % SPEED, 1920 // 2, 1050, 16))
    text = ""
    for k, (body, cx, cy, size) in enumerate(labels):
        src = "[vout]" if k == 0 else "[t%d]" % k
        dst = "[t%d]" % (k + 1) if k < len(labels) - 1 else "[final]"
        text += ("%sdrawtext=%stext='%s':fontcolor=black:fontsize=%d:box=1:boxcolor=white:"
                 "boxborderw=8:x=%d-text_w/2:y=%d%s;"
                 % (src, ("fontfile=%s:" % font_file) if font_file else "",
                    body.replace(":", "\\:").replace("'", ""), size, cx, cy, dst))
    graph += ";" + text.rstrip(";")

    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error"] + inputs + \
          ["-filter_complex", graph, "-map", "[final]", "-an"]
    if still:
        # A few seconds in, not frame 1: at frame 1 every board is empty and
        # the still says nothing about the layout in use.
        cmd += ["-ss", str(still_at), "-frames:v", "1", out_path]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    subprocess.run(cmd, check=True)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default=os.path.join(ROOT, "checkpoints", "tetris_v54"))
    p.add_argument("--columns", default="control,100000,200000")
    p.add_argument("--still", action="store_true", help="render one frame and stop")
    p.add_argument("--out", default=os.path.join(OUT, "progression_grid.mp4"))
    a = p.parse_args()

    results = json.load(open(os.path.join(OUT, "results.json")))
    if "selection" not in results:
        sys.exit("results.json has no selection: run report.py --select first")
    chosen = results["selection"]
    rows = [chosen["median_state"], chosen["hard_state"]]
    manifest = json.load(open(os.path.join(OUT, "suite_manifest.json")))
    cap, player = int(manifest["placement_cap"]), int(manifest["player"])

    os.makedirs(PANELS, exist_ok=True)
    columns, panels = [], {}
    for spec_name in a.columns.split(","):
        if spec_name == "control":
            path, label, name = "control", "Zero-value control", "control"
        else:
            steps = int(spec_name)
            path = os.path.join(a.folder, "ckpt_%d_steps.zip" % steps)
            label, name = "%dk steps" % (steps // 1000), "%dk" % (steps // 1000)
        columns.append(name)
        for state in rows:
            key = "%s_%s" % (name, state)
            mp4 = os.path.join(PANELS, key + ".mp4")
            if os.path.exists(mp4):
                print("cached panel %s" % key)
                probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                                        "-show_entries", "stream=width,height", "-of", "csv=p=0", mp4],
                                       capture_output=True, text=True, check=True).stdout.strip()
                w, h = (int(v) for v in probe.split(",")[:2])
                panels[(name, state)] = {"path": mp4, "width": w, "height": h}
            else:
                print("rendering panel %s ..." % key)
                panels[(name, state)] = capture_panel(path, state, cap, player, mp4, label)
                print("   %s" % panels[(name, state)]["end_reason"])

    out = a.out if not a.still else os.path.join(OUT, "progression_still.png")
    compose(panels, columns, rows, out, still=a.still)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
