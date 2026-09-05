#!/usr/bin/env python3
"""
Rendering, driven entirely by a JSON spec.

studio.py decides WHAT to say and when; everything about how it looks lives
here and, at runtime, in the overlays.json written beside each video. That
split is the point: text, size, colour, position and timing become data you can
edit and re-render, instead of constants buried in a filter string.

    studio.py   ->  overlays.json  ->  restyle.py
                        (edit it)

Both studio.py and restyle.py render through render_spec() below, so a video
re-rendered after an edit goes through exactly the same path as the original
and the two cannot drift.
"""
import json
import os
import subprocess

DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

TRANSITIONS = ("none", "fade", "dissolve", "pixelize")
OUT_FPS = 60

# The console draws its status bar across the bottom of the picture -- roughly
# the last fifth on an NES. Text dropped on top of it makes both unreadable.
HUD_TOP = 0.82

# Average advance width as a fraction of font size, used only when Pillow is
# missing and text cannot be measured. ~0.55 suits a proportional sans; a
# fixed-cell pixel face such as Press Start 2P is nearer 1.0.
CHAR_WIDTH_RATIO = 0.55

# Every default the renderer has. Anything here can be overridden per video in
# overlays.json, and anything in "caption" can be overridden per line.
DEFAULT_STYLE = {
    "font": DEFAULT_FONT,
    "title_font": DEFAULT_FONT,
    "char_width_ratio": CHAR_WIDTH_RATIO,
    "transition": "fade",
    "transition_seconds": 0.25,
    "blur": 20,
    "crf": 18,
    "title": {
        "size_div": 40,          # font size = frame width / size_div
        "color": "white@0.92",
        "border_color": "black@0.9",
        "y_frac": 0.045,
    },
    "caption": {
        "size_div": 30,
        "color": "white",
        "border_color": "black@0.92",
        "shadow": True,
        "hold": 2.6,
        "y_frac_vertical": 0.80,   # into the blurred band, clear of the game
        "y_frac_wide": HUD_TOP - 0.08,   # lower third, above the status bar
    },
    "watermark": {
        "size_div": 55,
        "color": "white@0.6",
        "border_color": "black@0.7",
        "y_frac_vertical": 0.925,
        "y_frac_wide": 0.94,
    },
}


def merge_style(user):
    """User style over DEFAULT_STYLE, one level deep."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_STYLE.items()}
    for key, value in (user or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update(value)
        else:
            out[key] = value
    return out


# ------------------------------------------------------------------ text --
def esc(text):
    """Escape for ffmpeg drawtext, which treats ' : \\ % specially.

    Apostrophes are replaced rather than escaped: quoting them through a
    filter_complex string survives neither the shell nor ffmpeg reliably, and a
    typographic apostrophe renders identically."""
    out = str(text).replace("\\", "\\\\").replace(":", "\\:").replace("%", "\\%")
    return out.replace("'", "’")


def esc_path(path):
    """Escape a path for a drawtext fontfile= option.

    Unescaped this works right up until a font lives somewhere with a space or
    a colon in the name, at which point ffmpeg reads the rest of the path as
    further filter options and fails with something unrelated-looking."""
    return str(path).replace(chr(92), chr(92) * 2).replace(":", chr(92) + ":")


def text_width(text, font_path, size, ratio=CHAR_WIDTH_RATIO):
    try:
        from PIL import ImageFont
        return ImageFont.truetype(font_path, size).getlength(text)
    except Exception:
        return len(text) * size * ratio


def fit_size(text, font_path, size, max_width, ratio=CHAR_WIDTH_RATIO, floor=14):
    """Largest size at or below `size` whose line fits max_width.

    drawtext neither wraps nor shrinks -- an overlong line just draws past both
    edges of the frame with no error. Easy to hit the moment the font changes,
    since a fixed-cell pixel face is nearly twice the width of a sans at the
    same point size."""
    while size > floor and text_width(text, font_path, size, ratio) > max_width:
        size -= 2
    return size


def draw(label, out_label, text, font, size, y, color, border_color,
         border_w=None, shadow=False, x="(w-text_w)/2", enable=None):
    """One drawtext stage.

    Outlined rather than boxed: a filled box takes a rectangle of the frame for
    as long as the line is up, while an outline covers only the glyphs and
    stays readable over anything."""
    parts = [
        "%sdrawtext=fontfile=%s" % (label, esc_path(font)),
        "text='%s'" % esc(text),
        "fontcolor=%s" % color,
        "fontsize=%d" % size,
        "borderw=%d" % (border_w if border_w is not None else max(2, size // 12)),
        "bordercolor=%s" % border_color,
        "x=%s" % x,
        "y=%d" % y,
    ]
    if shadow:
        parts += ["shadowcolor=black@0.55", "shadowx=2", "shadowy=2"]
    if enable:
        parts.append("enable='%s'" % enable)
    return ":".join(parts) + out_label


# --------------------------------------------------------------- filters --
def cut_filter(segments, with_audio=True, transition="fade", trans_s=0.25):
    """trim/concat prefix stitching the chosen segments into one stream, so the
    whole render stays a single ffmpeg pass.

    Hard cuts between unrelated moments read as a glitch rather than an edit,
    so segments are separated by a transition. `fade` needs no xfade chaining
    and so survives any segment count; `dissolve` and `pixelize` chain xfade
    pairwise and cost trans_s of overlap at every join."""
    if not segments:
        return "", "[0:v]", "0:a?"

    n = len(segments)
    chains, vlabels, alabels = [], [], []
    for i, seg in enumerate(segments):
        start, duration = seg[0], seg[1]
        vchain = ("[0:v]trim=start=%.3f:duration=%.3f,setpts=PTS-STARTPTS"
                  % (start, duration))
        if transition in ("dissolve", "pixelize"):
            # xfade refuses a stream whose frame rate it cannot determine, and
            # trim+setpts leaves it undefined ("current rate of 1/0 is invalid").
            vchain += ",fps=%d" % OUT_FPS
        if transition == "fade":
            # In on all but the first, out on all but the last, so the reel
            # opens and closes on picture rather than on black.
            if i > 0:
                vchain += ",fade=t=in:st=0:d=%.3f" % trans_s
            if i < n - 1:
                vchain += ",fade=t=out:st=%.3f:d=%.3f" % (
                    max(0.0, duration - trans_s), trans_s)
        chains.append(vchain + "[cv%d]" % i)
        vlabels.append("[cv%d]" % i)
        if with_audio:
            achain = ("[0:a]atrim=start=%.3f:duration=%.3f,asetpts=PTS-STARTPTS"
                      % (start, duration))
            if transition == "fade":
                if i > 0:
                    achain += ",afade=t=in:st=0:d=%.3f" % trans_s
                if i < n - 1:
                    achain += ",afade=t=out:st=%.3f:d=%.3f" % (
                        max(0.0, duration - trans_s), trans_s)
            chains.append(achain + "[ca%d]" % i)
            alabels.append("[ca%d]" % i)

    if transition in ("dissolve", "pixelize") and n > 1:
        kind = "fade" if transition == "dissolve" else "pixelize"
        prev_v = vlabels[0]
        prev_a = alabels[0] if with_audio else None
        elapsed = segments[0][1]
        for i in range(1, n):
            out_v = "[xv%d]" % i
            chains.append("%s%sxfade=transition=%s:duration=%.3f:offset=%.3f%s"
                          % (prev_v, vlabels[i], kind, trans_s,
                             max(0.0, elapsed - trans_s), out_v))
            prev_v = out_v
            if with_audio:
                out_a = "[xa%d]" % i
                chains.append("%s%sacrossfade=d=%.3f%s"
                              % (prev_a, alabels[i], trans_s, out_a))
                prev_a = out_a
            elapsed += segments[i][1] - trans_s
        chains.append("%snull[vcut]" % prev_v)
        if with_audio:
            chains.append("%sanull[acut]" % prev_a)
    else:
        chains.append("".join(vlabels) + "concat=n=%d:v=1:a=0[vcut]" % n)
        if with_audio:
            chains.append("".join(alabels) + "concat=n=%d:v=0:a=1[acut]" % n)

    return ";".join(chains), "[vcut]", ("[acut]" if with_audio else None)


def remap_captions(captions, segments):
    """Move caption times onto the cut timeline.

    Times in the spec are SOURCE seconds. When the video is a highlight cut the
    output timeline differs, so a line would otherwise land over the wrong
    moment. Lines whose moment was cut out are dropped."""
    if not segments:
        return captions
    out = []
    for cap in captions:
        elapsed = 0.0
        for seg in segments:
            start, duration = seg[0], seg[1]
            if start <= cap["at"] < start + duration:
                moved = dict(cap)
                moved["at"] = elapsed + (cap["at"] - start)
                out.append(moved)
                break
            elapsed += duration
    return out


def build_filter(spec, width, height, src_label="[0:v]"):
    """The full video chain: blurred fill, title, watermark, captions."""
    style = merge_style(spec.get("style"))
    ratio = style["char_width_ratio"]
    vertical = height > width
    parts = [
        "%ssplit=2[bg][fg]" % src_label,
        "[bg]scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,"
        "gblur=sigma=%s[bgb]" % (width, height, width, height, style["blur"]),
        "[fg]scale=%d:%d:force_original_aspect_ratio=decrease:flags=neighbor[fgs]"
        % (width, height),
        "[bgb][fgs]overlay=(W-w)/2:(H-h)/2[v0]",
    ]

    stage = "[v0]"
    title = spec.get("title")
    if title:
        cfg = style["title"]
        size = fit_size(title, style["title_font"],
                        max(18, width // cfg["size_div"]), int(width * 0.9), ratio)
        parts.append(draw(stage, "[v1]", title, style["title_font"], size,
                          int(height * cfg["y_frac"]), cfg["color"],
                          cfg["border_color"]))
        stage = "[v1]"

    mark = spec.get("watermark")
    if mark:
        cfg = style["watermark"]
        size = max(14, width // cfg["size_div"])
        # Centred in the bottom band on vertical; tucked into the blurred
        # pillarbox on wide, where centring would put it on the status bar.
        x = "(w-text_w)/2" if vertical else str(int(width * 0.02))
        y = cfg["y_frac_vertical"] if vertical else cfg["y_frac_wide"]
        parts.append(draw(stage, "[v2]", mark, style["font"], size,
                          int(height * y), cfg["color"], cfg["border_color"],
                          border_w=2, x=x))
        stage = "[v2]"

    caps = remap_captions(spec.get("captions", []), spec.get("segments"))
    if not caps:
        parts.append("%snull[vout]" % stage)
        return ";".join(parts)

    cfg = style["caption"]
    # Placement depends on the shape of the canvas: scaled into 9:16 the frame
    # leaves deep bands and the caption drops below the picture entirely;
    # scaled into 16:9 the picture fills the height and the caption has to
    # overlay, so it goes just above the status bar rather than through it.
    y = int(height * (cfg["y_frac_vertical"] if vertical else cfg["y_frac_wide"]))
    for i, cap in enumerate(caps):
        nxt = "[vout]" if i == len(caps) - 1 else "[cap%d]" % i
        font = cap.get("font", style["font"])
        size = cap.get("size") or max(20, width // cfg["size_div"])
        size = fit_size(cap["text"], font, size, int(width * 0.92), ratio)
        hold = cap.get("hold", cfg["hold"])
        parts.append(draw(stage, nxt, cap["text"], font, size,
                          cap.get("y", y), cap.get("color", cfg["color"]),
                          cap.get("border_color", cfg["border_color"]),
                          shadow=cfg.get("shadow", True),
                          enable="between(t,%.2f,%.2f)" % (cap["at"], cap["at"] + hold)))
        stage = nxt
    return ";".join(parts)


# ---------------------------------------------------------------- render --
def has_audio_stream(path):
    """Assumes audio when it cannot tell: a cut built with no audio chain drops
    the soundtrack silently, whereas guessing wrong the other way fails loudly
    at encode time, which is the better error."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                              "-show_entries", "stream=index", "-of", "csv=p=0", path],
                             capture_output=True, text=True, check=True)
        return bool(out.stdout.strip())
    except Exception:
        return True


def render_spec(spec, out_dir=None, only=None, verbose=True):
    """Render every output the spec lists. Returns the paths written."""
    style = merge_style(spec.get("style"))
    source = spec["source"]
    if not os.path.exists(source):
        raise SystemExit("Source video missing: %s" % source)
    with_audio = has_audio_stream(source)
    segments = spec.get("segments")
    written = []

    for out in spec["outputs"]:
        if only and out["file"] != only:
            continue
        width, height = int(out["width"]), int(out["height"])
        path = os.path.join(out_dir or ".", out["file"])
        cut, vlabel, alabel = cut_filter(segments, with_audio,
                                         style["transition"],
                                         style["transition_seconds"])
        graph = build_filter(spec, width, height, src_label=vlabel)
        if cut:
            graph = cut + ";" + graph
        audio_map = ["-map", alabel] if alabel else []
        if verbose:
            print("Rendering %s (%dx%d) ..." % (out["file"], width, height))
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", source,
             "-filter_complex", graph, "-map", "[vout]", *audio_map,
             "-r", str(OUT_FPS), "-c:v", "libx264", "-preset", "slow",
             "-crf", str(style["crf"]), "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", path],
            check=True)
        written.append(path)
    return written


def load_spec(path):
    with open(path) as handle:
        return json.load(handle)


def save_spec(spec, path):
    with open(path, "w") as handle:
        json.dump(spec, handle, indent=2)
    return path
