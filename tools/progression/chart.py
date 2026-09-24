"""The card that closes the video: what the numbers on screen actually were.

One 1920x1080 PNG, drawn from the panel results and nothing else. Every figure
on it is measured from the games in the video -- there is no model output here
and no number that was not produced by a run.

What it shows, and why these and not "accuracy": a policy is not a classifier.
There is no label to be right or wrong about, so the honest summary of a
trained agent is the DISTRIBUTION of what it achieved: the mean, the spread
around it, the worst and best run, how much work it took to get there, and
whether the game ended on its own or was stopped by the cap. A capped run is
censored -- the agent had not finished -- and is counted separately rather than
folded into the average, because pooling them understates exactly the
checkpoints that are doing well.

The sample size is printed next to every figure. With a handful of start states
the spread is indicative, not a confidence interval, and the card says so
rather than implying precision it does not have.
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG = (10, 10, 14)
INK = (245, 245, 245)
DIM = (150, 152, 160)
RULE = (48, 50, 58)
BAR = (86, 196, 172)             # the trained columns
BAR_CONTROL = (110, 112, 124)    # the untrained control, deliberately grey
WHISKER = (232, 200, 108)


def _font(size, mono=False):
    for path in (("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",) if mono else
                 ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",)) + (
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:                                     # noqa: BLE001
                pass
    return ImageFont.load_default()


def _title_font(size):
    """Press Start 2P for the headline only -- it is fixed-cell, so a long
    line of it is enormous, but it is the channel's face."""
    path = os.path.expanduser("~/.local/share/fonts/PressStart2P-Regular.ttf")
    if os.path.exists(path):
        try:
            return ImageFont.truetype(path, size)
        except Exception:                                         # noqa: BLE001
            pass
    return _font(size)


def summarise(rows):
    """Per column: n, mean, sample standard deviation, range, work, endings.

    Population vs sample matters at these sizes: with n runs the sample sd
    (n-1) is the one that does not flatter a small sample, and at n=1 it is
    undefined and is reported as undefined rather than as zero."""
    values = [float(r["value"]) for r in rows]
    n = len(values)
    mean = sum(values) / n if n else 0.0
    sd = (math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
          if n > 1 else None)
    work = [float(r.get("decisions") or 0) for r in rows]
    per_work = (100.0 * sum(values) / sum(work)) if sum(work) else 0.0
    return {"n": n, "mean": mean, "sd": sd,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "median": sorted(values)[n // 2] if n else 0.0,
            "decisions": sum(work) / n if n else 0.0,
            "per_100": per_work,
            "finished": sum(1 for r in rows if r.get("end_reason") == "game_over"),
            "capped": sum(1 for r in rows if r.get("end_reason") == "placement_cap")}


def build(results, out_path, game="", metric="", decision_word="decisions",
          subtitle="", order=None):
    """`results` is one dict per panel: column, value, decisions, end_reason."""
    groups = {}
    for row in results:
        groups.setdefault(row["column"], []).append(row)
    names = [c for c in (order or sorted(groups)) if c in groups]
    stats = {name: summarise(groups[name]) for name in names}

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_title, f_head = _title_font(34), _font(30)
    f_body, f_small, f_num = _font(26), _font(20), _font(30, mono=True)

    d.text((80, 64), (game or "Training progression").upper(), font=f_title, fill=INK)
    d.text((80, 124), subtitle or "Every figure measured from the games in this video",
           font=f_body, fill=DIM)
    d.line((80, 176, W - 80, 176), fill=RULE, width=2)

    # ------------------------------------------------------------- chart --
    top, bottom, left = 290, 830, 130
    width = 880
    step = width // max(1, len(names))
    bar_w = min(150, int(step * 0.5))
    ceiling = max([stats[n]["max"] for n in names] + [1.0]) * 1.15
    d.text((left, top - 74), metric or "Result per game", font=f_head, fill=INK)
    d.text((left, top - 36),
           "bar = mean     line = min to max     dot = one game", font=f_small, fill=DIM)
    for i in range(5):                       # gridlines, drawn under the bars
        y = bottom - int((bottom - top) * i / 4.0)
        d.line((left, y, left + width, y), fill=RULE, width=1)
        d.text((left - 78, y - 13), "%d" % round(ceiling * i / 4.0), font=f_small, fill=DIM)
    for i, name in enumerate(names):
        s = stats[name]
        cx = left + i * step + step // 2
        h_mean = int((bottom - top) * s["mean"] / ceiling)
        colour = BAR_CONTROL if name.lower().startswith("untrained") or name == "control" else BAR
        d.rectangle((cx - bar_w // 2, bottom - h_mean, cx + bar_w // 2, bottom), fill=colour)
        # The range, as a whisker, and every game as a dot: with a handful of
        # start states the spread IS the story, and a bar alone hides that two
        # runs of the same checkpoint can differ several times over.
        y_lo = bottom - int((bottom - top) * s["min"] / ceiling)
        y_hi = bottom - int((bottom - top) * s["max"] / ceiling)
        d.line((cx, y_lo, cx, y_hi), fill=WHISKER, width=3)
        for y in (y_lo, y_hi):
            d.line((cx - 18, y, cx + 18, y), fill=WHISKER, width=3)
        for j, row in enumerate(groups[name]):
            y = bottom - int((bottom - top) * float(row["value"]) / ceiling)
            dx = (j - (len(groups[name]) - 1) / 2.0) * 16      # spread, not stacked
            d.ellipse((cx + dx - 5, y - 5, cx + dx + 5, y + 5), fill=INK)
        label = "%.0f" % s["mean"]
        d.text((cx - d.textlength(label, font=f_num) / 2, y_hi - 48), label,
               font=f_num, fill=INK)
        wrapped = name if len(name) < 16 else name.replace(" ", "\n", 1)
        d.multiline_text((cx, bottom + 18), wrapped, font=f_body, fill=INK,
                         anchor="ma", align="center", spacing=6)
        note = "n=%d" % s["n"]
        d.text((cx - d.textlength(note, font=f_small) / 2,
                bottom + (58 if "\n" not in wrapped else 92)), note, font=f_small, fill=DIM)

    # ------------------------------------------------------------- table --
    # Names are as long as the user's --labels, so the numeric columns start
    # well clear of them rather than at a guessed offset.
    tx, ty = 1060, 290
    name_w = 300
    cols = [("mean", 0), ("sd", 120), ("range", 240), (decision_word[:9], 420)]
    for head, dx in cols:
        d.text((tx + name_w + dx, ty - 34), head, font=f_small, fill=DIM)
    d.line((tx, ty - 2, W - 80, ty - 2), fill=RULE, width=1)
    for i, name in enumerate(names):
        s = stats[name]
        y = ty + 24 + i * 56
        d.text((tx, y), name if len(name) < 20 else name[:19] + "-", font=f_body, fill=INK)
        d.text((tx + name_w, y), "%.0f" % s["mean"], font=f_num, fill=INK)
        d.text((tx + name_w + 120, y), "-" if s["sd"] is None else "%.0f" % s["sd"],
               font=f_num, fill=INK if s["sd"] is not None else DIM)
        d.text((tx + name_w + 240, y), "%.0f-%.0f" % (s["min"], s["max"]),
               font=f_num, fill=INK)
        d.text((tx + name_w + 420, y), "%.0f" % s["decisions"], font=f_num, fill=INK)

    ey = ty + 24 + len(names) * 56 + 64
    d.text((tx, ey - 34), "how each game ended", font=f_small, fill=DIM)
    d.line((tx, ey - 2, W - 80, ey - 2), fill=RULE, width=1)
    for i, name in enumerate(names):
        s = stats[name]
        y = ey + 24 + i * 48
        d.text((tx, y), name if len(name) < 20 else name[:19] + "-", font=f_body, fill=INK)
        d.text((tx + name_w, y), "%d finished" % s["finished"], font=f_body, fill=INK)
        if s["capped"]:
            d.text((tx + name_w + 220, y), "%d hit the cap" % s["capped"],
                   font=f_body, fill=WHISKER)

    # ------------------------------------------------------------ footer --
    n_total = sum(stats[n]["n"] for n in names)
    d.line((80, H - 130, W - 80, H - 130), fill=RULE, width=2)
    d.text((80, H - 108),
           "%d games, %d start states per column. Spread is across START STATES, "
           "not repeats of one game." % (n_total, stats[names[0]]["n"] if names else 0),
           font=f_small, fill=DIM)
    d.text((80, H - 74),
           "A run stopped by the cap had not finished, so it is counted apart from "
           "the ones that played themselves out.", font=f_small, fill=DIM)
    d.text((80, H - 40),
           "An illustration of the training, at this sample size -- not a "
           "measurement of how good the agent is.", font=f_small, fill=DIM)

    img.save(out_path)
    return {"path": out_path, "stats": stats, "order": names}
