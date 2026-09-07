#!/usr/bin/env python3
"""
Decode the NES sprite table so objects can be fed to the policy directly.

The agent currently gets only an 84x84 grayscale image and has to discover from
reward alone that some smudge of pixels is a goomba worth avoiding. Enemies are
the worst case for that: small, fast, and lethal on contact. Their positions are
already in RAM.

The NES keeps 64 sprites in OAM as 4 bytes each -- Y, tile index, attributes, X
-- shadowed in CPU RAM, conventionally at $0200. Earlier RAM scans in this
project turned up busy addresses at 0x022B, 0x022F, 0x0233, 0x0237, 0x023B,
0x023F: a stride of exactly 4, which is that table's signature, with those hits
landing on the X byte of consecutive sprites. This tool decodes the table so you
can confirm it and learn which tile IDs are which object.

WHAT THIS COVERS: sprites -- enemies, and items that move (a mushroom sliding
out of a block). NOT background: blocks, pipes, ground and walls are nametable
tiles, not sprites, and do not appear here. That is a reasonable split, since
terrain is large, static and high-contrast, which is exactly what a CNN reads
well from pixels, while small fast enemies are what it struggles with.

Usage:
    # Decode sprites from a recording, with screenshots to match IDs against
    python inspect_sprites.py --demo ./human_demos_v3/<file>.bk2 --shots ./sprite_shots

    # Check a different base address if 0x0200 turns out to be wrong
    python inspect_sprites.py --demo <file>.bk2 --oam-base 0x0200

    # Which tile IDs show up most, and where they tend to appear
    python inspect_sprites.py --demo <file>.bk2 --summary
"""
import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import stable_retro as retro

try:
    import cv2
except ImportError:
    cv2 = None

FPS = 60.0988
OFF_SCREEN_Y = 240   # the NES parks unused sprites below the visible area


def stamp(frame):
    secs = frame / FPS
    return f"{int(secs // 60):02d}:{secs % 60:06.3f}"


def decode_oam(ram, base=0x0200, count=64):
    """Return the active sprites as (index, x, y, tile, attr).

    A sprite is 4 bytes: Y, tile, attributes, X. Unused entries are parked at
    Y >= 240, below the visible screen, so they are filtered out."""
    out = []
    for i in range(count):
        b = base + i * 4
        if b + 3 >= len(ram):
            break
        y, tile, attr, x = int(ram[b]), int(ram[b + 1]), int(ram[b + 2]), int(ram[b + 3])
        if y < OFF_SCREEN_Y:
            out.append((i, x, y, tile, attr))
    return out


def replay(bk2_path):
    movie = retro.Movie(bk2_path)
    movie.step()
    env = retro.make(game=movie.get_game(), state=None,
                     use_restricted_actions=retro.Actions.ALL,
                     players=movie.players, render_mode="rgb_array")
    env.initial_state = movie.get_state()
    obs, _info = env.reset()
    while movie.step():
        keys = [movie.get_key(i, 0) for i in range(env.num_buttons)]
        obs, _r, term, trunc, info = env.step(keys)
        yield env.get_ram(), obs, info
        if term or trunc:
            break
    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", required=True, help="A .bk2 recording to decode")
    p.add_argument("--oam-base", type=lambda v: int(v, 0), default=0x0200, help="Base of the sprite table (default: %(default)s)")
    p.add_argument("--every", type=int, default=120, help="Print sprites every N frames (default: %(default)s)")
    p.add_argument("--shots", default=None, help="Directory for screenshots taken at each printed frame, so tile IDs can be matched to what is on screen")
    p.add_argument("--summary", action="store_true", help="Skip the per-frame dump; report which tile IDs appear and where")
    p.add_argument("--max-frames", type=int, default=20000)
    args = p.parse_args()

    if args.shots:
        os.makedirs(args.shots, exist_ok=True)

    print(f"Decoding sprites from {args.demo}")
    print(f"OAM base 0x{args.oam_base:04X}, 64 sprites x 4 bytes (Y, tile, attr, X)\n")

    tile_counts = Counter()
    tile_positions = defaultdict(list)
    active_counts = []
    frame = 0

    for ram, obs, info in replay(args.demo):
        if frame >= args.max_frames:
            break
        sprites = decode_oam(ram, args.oam_base)
        active_counts.append(len(sprites))
        for _i, x, y, tile, _a in sprites:
            tile_counts[tile] += 1
            tile_positions[tile].append((x, y))

        if not args.summary and frame % args.every == 0:
            hero = info.get("hpos")
            print(f"frame {frame:6d} [{stamp(frame)}]  hpos={hero}  {len(sprites)} active sprites")
            for _i, x, y, tile, attr in sprites[:14]:
                rel = f"{x - hero:+4d}" if hero is not None else "   ?"
                print(f"    tile 0x{tile:02X}  at x={x:3d} y={y:3d}  (dx from Mario {rel})  attr 0x{attr:02X}")
            if len(sprites) > 14:
                print(f"    ... and {len(sprites) - 14} more")
            if args.shots and cv2 is not None:
                path = os.path.join(args.shots, f"f{frame:06d}.png")
                cv2.imwrite(path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
                print(f"    [{path}]")
            print()
        frame += 1

    print("=" * 68)
    if not active_counts:
        print("No frames decoded.")
        return
    print(f"{frame} frames. Active sprites per frame: "
          f"min {min(active_counts)}, mean {sum(active_counts)/len(active_counts):.1f}, "
          f"max {max(active_counts)}")
    if max(active_counts) == 0:
        print("\nNOTHING DECODED -- 0x{:04X} is probably not the sprite table.".format(args.oam_base))
        print("Try --oam-base 0x0300 or 0x0700. A correct base shows a handful of")
        print("active sprites that rises when enemies are on screen.")
        return

    print(f"\nTile IDs seen ({len(tile_counts)} distinct), most frequent first:")
    print(f"  {'TILE':>6}  {'FRAMES':>8}  {'typical y':>10}  guess")
    for tile, n in tile_counts.most_common(25):
        ys = [y for _x, y in tile_positions[tile]]
        med = sorted(ys)[len(ys) // 2]
        # Mario's own sprites dominate the count and sit mid-screen; things
        # resting on the ground sit low. Only a hint -- confirm on screenshots.
        guess = "Mario (always present?)" if n > frame * 0.8 else (
                "ground-level object" if med > 150 else "airborne / floating")
        print(f"  0x{tile:02X}  {n:8d}  {med:10d}  {guess}")

    print("\nNEXT: open the screenshots and match a tile ID to what is on screen at")
    print("that frame. Once goomba/koopa/item IDs are known, those positions can be")
    print("fed to the policy as a second observation input, so enemies stop having")
    print("to be inferred from a handful of grey pixels.")


if __name__ == "__main__":
    main()
