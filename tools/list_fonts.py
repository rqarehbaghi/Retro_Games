#!/usr/bin/env python3
"""
See what fonts this machine has, and what they actually look like.

Picking a font from a list of family names is guesswork, and a name tells you
nothing about whether a face suits pixel-art gameplay. This renders a real
sample sheet -- every font drawn with your own caption text, through the same
ffmpeg drawtext the videos use -- so what you see on the sheet is exactly what
lands on the video.

    python list_fonts.py                       # sample sheet of every font
    python list_fonts.py --list                # just the paths, no rendering
    python list_fonts.py --grep mono           # only families matching
    python list_fonts.py --sample "He died. To that. To THAT."

Then put the path you like in studio.json:

    "font":       "/usr/share/fonts/truetype/.../Whatever.ttf",
    "title_font": "/usr/share/fonts/truetype/.../Whatever.ttf"

or try it on an already staged video without committing to it:

    python restyle.py ./studio_out/<folder> --set-font /path/to/Whatever.ttf

NOTHING PIXEL-Y INSTALLED? Nothing in Debian or Ubuntu ships one -- there is
no fonts-press-start-2p package. Fetch one by hand:

    mkdir -p ~/.local/share/fonts
    curl -L -o ~/.local/share/fonts/PressStart2P-Regular.ttf \\
      https://github.com/google/fonts/raw/main/ofl/pressstart2p/PressStart2P-Regular.ttf
    fc-cache -f

Use a pixel face for "title_font" and keep a proportional one for captions:
Press Start 2P is fixed-cell at roughly one em per character, so caption lines
get shrunk hard to fit the frame.
"""
import argparse
import os
import shutil
import subprocess
import sys

from overlays import esc, esc_path

ROW_HEIGHT = 76
SHEET_WIDTH = 1500
DEFAULT_SAMPLE = "He died. To that. To THAT. 0123"


def system_fonts():
    """(path, family, style) for every font fontconfig knows about."""
    if not shutil.which("fc-list"):
        sys.exit("fc-list not found -- install fontconfig:\n  sudo apt install -y fontconfig")
    out = subprocess.run(
        ["fc-list", "--format", "%{file}|%{family}|%{style}\n"],
        capture_output=True, text=True, check=True).stdout
    seen, fonts = set(), []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 3 or not parts[0]:
            continue
        path, family, style = parts[0], parts[1].split(",")[0], parts[2].split(",")[0]
        # Only what drawtext can actually load, and one entry per file.
        if path in seen or os.path.splitext(path)[1].lower() not in (".ttf", ".otf"):
            continue
        seen.add(path)
        fonts.append((path, family, style))
    return sorted(fonts, key=lambda f: (f[1].lower(), f[2].lower()))


def render_sheet(fonts, sample, out_path):
    """One ffmpeg pass: a dark canvas with each font drawn on its own row."""
    height = ROW_HEIGHT * len(fonts) + 30
    chains = []
    stage = "[0:v]"
    for i, (path, family, style) in enumerate(fonts):
        top = 20 + i * ROW_HEIGHT
        # fc-list reports each NAMED INSTANCE of a variable font separately
        # (Ubuntu Condensed Extra Bold, Ubuntu Sans Thin Italic ...) but they
        # all share one file, and drawtext renders that file at its DEFAULT
        # instance regardless. Labelling a row with the instance name would
        # promise a weight the sample does not show.
        variable = "[" in os.path.basename(path)
        label = "%d. %s%s" % (i + 1, family,
                              " (variable font, shown at its default weight)"
                              if variable else " " + style)
        nxt = "[l%d]" % i
        # The label is drawn in a known-good face so a broken font cannot make
        # its own row unidentifiable.
        chains.append(
            "%sdrawtext=fontfile=%s:text='%s':fontcolor=0x7fd1ff:fontsize=17:"
            "x=24:y=%d%s" % (stage, esc_path(fonts[0][0]), esc(label), top, nxt))
        stage = nxt
        nxt = "[s%d]" % i
        chains.append(
            "%sdrawtext=fontfile=%s:text='%s':fontcolor=white:fontsize=30:"
            "x=24:y=%d%s" % (stage, esc_path(path), esc(sample), top + 22, nxt))
        stage = nxt
    chains.append("%snull[vout]" % stage)

    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi",
         "-i", "color=c=0x14161a:s=%dx%d:d=1" % (SHEET_WIDTH, height),
         "-filter_complex", ";".join(chains), "-map", "[vout]",
         "-frames:v", "1", out_path],
        check=True, capture_output=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="Print paths only, render nothing")
    parser.add_argument("--grep", default=None, help="Only fonts whose family or path contains this (case-insensitive)")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE, help="Text to draw in each font (default: %(default)s)")
    parser.add_argument("--out", default="./font_sheet.png", help="Where to write the sample sheet (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=40, help="Most fonts to draw on one sheet; use --grep to narrow instead of raising this. (default: %(default)s)")
    args = parser.parse_args()

    fonts = system_fonts()
    if args.grep:
        needle = args.grep.lower()
        fonts = [f for f in fonts if needle in f[1].lower() or needle in f[0].lower()]
    if not fonts:
        sys.exit("No fonts matched. Try without --grep, or install some:\n"
                 "  sudo apt install -y fonts-dejavu-core fonts-liberation")

    if args.list:
        print("%d font file(s):\n" % len(fonts))
        for path, family, style in fonts:
            print("  %-34s %-14s %s" % (family[:34], style[:14], path))
        return

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is not on PATH -- see README step 1.")

    shown = fonts[:args.limit]
    if len(fonts) > len(shown):
        print("%d fonts found, drawing the first %d. Narrow with --grep."
              % (len(fonts), len(shown)))
    render_sheet(shown, args.sample, args.out)
    print("Wrote %s -- %d fonts, drawn with the same drawtext the videos use."
          % (args.out, len(shown)))
    print("\nPick one and set it:")
    print('  studio.json:  "font": "<path>",  "title_font": "<path>"')
    print("  or try it first: python restyle.py ./studio_out/<folder> --set-font <path>")


if __name__ == "__main__":
    main()
