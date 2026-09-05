#!/usr/bin/env python3
"""
Re-render a staged video after editing its overlays.json.

Every run of studio.py writes an overlays.json next to the videos holding
everything drawn on them -- each caption with its timestamp and text, the
title, the watermark, and the font, size, colour and position rules behind
them. Edit that file and re-run this; nothing is replayed and nothing is
recomputed, so the loop is a few seconds rather than a few minutes.

    python restyle.py ./studio_out/20260905-101500_super-mario-bros-3

WHAT YOU CAN CHANGE. Anything in the file. Rewrite a caption, move it in time,
give one line its own colour or size, drop a line by deleting it, add one by
adding an object with "at" and "text". Change the font for everything, or for a
single line. Nothing here is generated -- the wording came from a table in
studio.py, and it is meant to be overwritten.

    --list          print the captions with their timestamps and stop
    --only FILE     render just one output, e.g. only the vertical
    --set-font PATH set the font for everything and render
    --style KEY=V   override one style value, e.g. --style caption.size_div=24

Fields are nested under "style": "caption": {...}, "title": {...},
"watermark": {...}. size_div is a DIVISOR of the frame width, so a smaller
number means larger text. Colours take ffmpeg syntax, including alpha:
"white@0.9", "yellow", "0xFF66CC".
"""
import argparse
import os
import sys

from overlays import DEFAULT_STYLE, load_spec, merge_style, render_spec, save_spec

SPEC_NAME = "overlays.json"


def resolve(path):
    """Accept either the staged folder or the overlays.json inside it."""
    if os.path.isdir(path):
        return os.path.join(path, SPEC_NAME)
    return path


def apply_style_override(spec, assignment):
    """--style caption.size_div=24 -> spec['style']['caption']['size_div'] = 24"""
    if "=" not in assignment:
        sys.exit("--style wants KEY=VALUE, e.g. --style caption.size_div=24")
    key, _, raw = assignment.partition("=")
    try:
        value = int(raw)
    except ValueError:
        try:
            value = float(raw)
        except ValueError:
            value = raw
    style = spec.setdefault("style", {})
    if "." in key:
        group, _, field = key.partition(".")
        if group not in DEFAULT_STYLE or not isinstance(DEFAULT_STYLE[group], dict):
            sys.exit("Unknown style group %r. Known: %s"
                     % (group, ", ".join(k for k, v in DEFAULT_STYLE.items()
                                         if isinstance(v, dict))))
        style.setdefault(group, {})[field] = value
    else:
        style[key] = value
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="A folder staged by studio.py, or the overlays.json in it")
    parser.add_argument("--list", action="store_true", help="Print the captions and exit without rendering")
    parser.add_argument("--only", default=None, help="Render just this output file, e.g. run_9x16.mp4")
    parser.add_argument("--set-font", default=None, help="Font path for captions and watermark")
    parser.add_argument("--set-title-font", default=None, help="Font path for the title alone")
    parser.add_argument("--style", action="append", default=[], metavar="KEY=VALUE",
                        help="Override one style value, repeatable. e.g. caption.size_div=24, caption.color=yellow, title.y_frac=0.03")
    args = parser.parse_args()

    spec_path = resolve(args.folder)
    if not os.path.exists(spec_path):
        sys.exit("No %s at %s\n  Pass the folder studio.py staged, or the json inside it."
                 % (SPEC_NAME, spec_path))
    spec = load_spec(spec_path)

    if args.list:
        print("title:     %s" % spec.get("title"))
        print("watermark: %s" % spec.get("watermark"))
        style = merge_style(spec.get("style"))
        print("font:      %s" % style["font"])
        print("captions:")
        for cap in spec.get("captions", []):
            extra = "".join(" %s=%s" % (k, v) for k, v in sorted(cap.items())
                            if k not in ("at", "text"))
            print("  %7.2fs  %s%s" % (cap["at"], cap["text"], extra))
        return

    for assignment in args.style:
        spec = apply_style_override(spec, assignment)
    if args.set_font:
        spec.setdefault("style", {})["font"] = args.set_font
    if args.set_title_font:
        spec.setdefault("style", {})["title_font"] = args.set_title_font

    style = merge_style(spec.get("style"))
    for path, what in ((style["font"], "font"), (style["title_font"], "title_font")):
        if not os.path.exists(path):
            sys.exit("%s not found: %s" % (what, path))

    folder = os.path.dirname(spec_path) or "."
    written = render_spec(spec, out_dir=folder, only=args.only)
    if not written:
        sys.exit("Nothing rendered -- --only %r matched no output in the spec." % args.only)

    # Persist the overrides so the file keeps matching the videos beside it.
    if args.style or args.set_font or args.set_title_font:
        save_spec(spec, spec_path)
        print("Updated %s with the overrides." % SPEC_NAME)
    for path in written:
        print("  %s" % path)


if __name__ == "__main__":
    main()
