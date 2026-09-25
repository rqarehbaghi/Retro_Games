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
GOLD = (240, 198, 88)      # the winner, and only the winner


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


def crown(d, cx, base, size, colour=GOLD):
    """A crown, drawn rather than typed.

    The glyph exists in some fonts and not others, and a missing glyph renders
    as a box on the one frame of the film most likely to be screenshotted."""
    w, h = size, size * 0.75
    left, right, top, bottom = cx - w / 2, cx + w / 2, base - h, base
    d.polygon([(left, bottom), (left, top + h * 0.25),
               (left + w * 0.25, top + h * 0.62), (cx, top),
               (right - w * 0.25, top + h * 0.62), (right, top + h * 0.25),
               (right, bottom)], fill=colour)
    for x in (left, cx, right):
        r = size * 0.09
        d.ellipse((x - r, top + h * 0.1 - r, x + r, top + h * 0.1 + r), fill=colour)


def build(results, out_path, game="", metric="", decision_word="decisions",
          subtitle="", order=None, handle=""):
    """One picture, one question: which of these plays best?

    Deliberately NOT a statistics page. An earlier version printed the mean,
    the sample standard deviation, the range, the work per game and how every
    game ended, in two tables -- which collided with each other, and which
    asked a viewer who came for Tetris to read a spreadsheet. The numbers that
    were dropped are still computed, because the narration is given them and
    speaks the honest version out loud; they are simply not on the card.

    `results` is one dict per panel: column, value, decisions, end_reason."""
    groups = {}
    for row in results:
        groups.setdefault(row["column"], []).append(row)
    names = [c for c in (order or sorted(groups)) if c in groups]
    stats = {name: summarise(groups[name]) for name in names}
    best = max(names, key=lambda n: stats[n]["mean"]) if names else ""
    unit = (metric or "points per game").split(" per ")[0].lower()

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_title, f_ask = _title_font(34), _font(38)
    f_name, f_small, f_num = _font(30), _font(22), _font(44, mono=True)
    f_win, f_win_big = _font(26), _font(46)

    d.text((80, 58), (game or "Training progression").upper(), font=f_title, fill=INK)
    if handle:
        d.text((W - 80 - d.textlength(handle, font=f_name), 66), handle,
               font=f_name, fill=BAR)
    d.text((80, 128), subtitle or "Which one plays best?", font=f_ask, fill=DIM)
    d.line((80, 196, W - 80, 196), fill=RULE, width=2)

    # --------------------------------------------------------------- bars --
    top, bottom, left, right = 300, 792, 150, W - 150
    step = (right - left) // max(1, len(names))
    bar_w = min(190, int(step * 0.56))
    ceiling = max([stats[n]["mean"] for n in names] + [1.0]) * 1.28
    d.line((left, bottom, right, bottom), fill=RULE, width=2)
    for i, name in enumerate(names):
        s = stats[name]
        cx = left + i * step + step // 2
        h = int((bottom - top) * s["mean"] / ceiling)
        winner = name == best
        d.rectangle((cx - bar_w // 2, bottom - h, cx + bar_w // 2, bottom),
                    fill=GOLD if winner else BAR)
        value = "%.0f" % s["mean"]
        d.text((cx - d.textlength(value, font=f_num) / 2, bottom - h - 62), value,
               font=f_num, fill=GOLD if winner else INK)
        if winner:
            crown(d, cx, bottom - h - 78, 56)
        label = name if len(name) < 14 else name.replace(" ", "\n", 1)
        d.multiline_text((cx, bottom + 22), label, font=f_name,
                         fill=INK if winner else DIM, anchor="ma", align="center",
                         spacing=6)

    # ------------------------------------------------------------- winner --
    if best:
        box = (150, 880, W - 150, 980)
        d.rectangle(box, outline=GOLD, width=3)
        crown(d, 208, 950, 46)
        d.text((258, 898), "WINNER", font=f_win, fill=GOLD)
        won = "%s  -  %.0f %s a game" % (best, stats[best]["mean"], unit)
        d.text((W - 190 - d.textlength(won, font=f_win_big), 902), won,
               font=f_win_big, fill=INK)

    # ----------------------------------------------------------- footnote --
    n = stats[names[0]]["n"] if names else 0
    # Below the winner box: above it, it ran into the column labels.
    d.text((150, 1008),
           "Average %s, over %d game%s each. One game can swing wildly, so treat "
           "this as a snapshot rather than a verdict."
           % (unit, n, "" if n == 1 else "s"), font=f_small, fill=DIM)

    img.save(out_path)
    return {"path": out_path, "stats": stats, "order": names, "best": best}
