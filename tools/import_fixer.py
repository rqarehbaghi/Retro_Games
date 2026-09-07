#!/usr/bin/env python3
"""
ROM Import Fixer & Matcher for Stable-Retro.

Inspects your ROMs/ folder against all stable-retro integration databases
(stable, contrib, and experimental). Detects headered/headerless hash mismatches
and imports all matching games.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import stable_retro as retro


def get_all_known_definitions():
    """Build a mapping of sha1 -> (game_name, integration_dir, ext)."""
    sha_map = {}
    name_map = {}

    # Gather data directories from retro
    data_dirs = []
    base_data = os.path.join(os.path.dirname(retro.__file__), "data")
    for sub in ("stable", "contrib", "experimental"):
        d = os.path.join(base_data, sub)
        if os.path.isdir(d):
            data_dirs.append((sub, d))

    for integ_name, root_dir in data_dirs:
        for game in os.listdir(root_dir):
            game_path = os.path.join(root_dir, game)
            if not os.path.isdir(game_path):
                continue
            sha_file = os.path.join(game_path, "rom.sha")
            metadata_file = os.path.join(game_path, "metadata.json")

            target_sha = None
            ext = None

            if os.path.isfile(sha_file):
                try:
                    with open(sha_file, "r", encoding="utf-8") as f:
                        target_sha = f.read().strip().split()[0].lower()
                except Exception:
                    pass

            if os.path.isfile(metadata_file):
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        ext = meta.get("default_state")  # or check system
                except Exception:
                    pass

            if target_sha:
                sha_map[target_sha] = {
                    "game": game,
                    "dir": game_path,
                    "integration": integ_name,
                }
                name_map[game.lower()] = {
                    "game": game,
                    "dir": game_path,
                    "sha": target_sha,
                    "integration": integ_name,
                }

    return sha_map, name_map


def hash_file(filepath):
    """Return raw SHA1, and header-stripped SHA1s for NES (16 bytes) and SNES (512 bytes)."""
    with open(filepath, "rb") as f:
        data = f.read()

    h_raw = hashlib.sha1(data).hexdigest().lower()
    h_nes = hashlib.sha1(data[16:]).hexdigest().lower() if len(data) > 16 else None
    h_snes = hashlib.sha1(data[512:]).hexdigest().lower() if len(data) > 512 else None

    return h_raw, h_nes, h_snes, data


def main():
    parser = argparse.ArgumentParser(description="Audit and fix ROM imports for stable-retro")
    parser.add_argument("--rom-dir", default="ROMs", help="Directory containing your ROM files (default: %(default)s)")
    parser.add_argument("--link-matches", action="store_true", help="Automatically copy/link header-stripped ROMs into retro data directory")
    args = parser.parse_args()

    if not os.path.isdir(args.rom_dir):
        print(f"Error: Directory '{args.rom_dir}' not found.")
        sys.exit(1)

    print("Scanning stable-retro databases (stable + contrib + experimental)...")
    sha_map, name_map = get_all_known_definitions()
    print(f"Loaded {len(sha_map)} known game SHA-1 definitions.\n")

    rom_files = []
    valid_exts = (".nes", ".sfc", ".smc", ".md", ".bin", ".gen", ".gb", ".gba", ".gg", ".sms", ".a26", ".pce")
    for root, _, files in os.walk(args.rom_dir):
        for f in files:
            if f.lower().endswith(valid_exts) or f.lower().endswith(".zip"):
                rom_files.append(os.path.join(root, f))

    print(f"Found {len(rom_files)} ROM files in '{args.rom_dir}'.")

    already_imported = 0
    direct_matches = []
    header_matches = []
    unmatched = []

    for path in rom_files:
        filename = os.path.basename(path)
        h_raw, h_nes, h_snes, data = hash_file(path)

        if h_raw in sha_map:
            info = sha_map[h_raw]
            # check if already imported
            try:
                cur_rom = retro.data.get_romfile_path(info["game"])
                if cur_rom and os.path.isfile(cur_rom):
                    already_imported += 1
                else:
                    direct_matches.append((path, info, "exact", None))
            except Exception:
                direct_matches.append((path, info, "exact", None))
        elif h_nes in sha_map:
            info = sha_map[h_nes]
            header_matches.append((path, info, "nes-header-stripped", 16))
        elif h_snes in sha_map:
            info = sha_map[h_snes]
            header_matches.append((path, info, "snes-header-stripped", 512))
        else:
            unmatched.append((path, filename, h_raw))

    print(f"\nAudit Results:")
    print(f"  Already active in stable-retro: {already_imported}")
    print(f"  Direct SHA matches ready to import: {len(direct_matches)}")
    print(f"  Header-mismatched ROMs fixable: {len(header_matches)}")
    print(f"  Unmatched ROMs (different region/rev/game): {len(unmatched)}")

    if direct_matches:
        print(f"\nImporting {len(direct_matches)} exact matches into stable-retro...")
        for path, info, _, _ in direct_matches:
            target_dir = info["dir"]
            ext = os.path.splitext(path)[1]
            dest = os.path.join(target_dir, f"rom{ext}")
            try:
                shutil.copyfile(path, dest)
                print(f"  [+] Imported: {info['game']}")
            except Exception as e:
                print(f"  [!] Failed to copy for {info['game']}: {e}")

    if header_matches:
        print(f"\nFixing {len(header_matches)} header-mismatched ROMs...")
        for path, info, fix_type, strip_len in header_matches:
            target_dir = info["dir"]
            ext = os.path.splitext(path)[1]
            dest = os.path.join(target_dir, f"rom{ext}")
            try:
                with open(path, "rb") as f_in:
                    f_in.seek(strip_len)
                    stripped_data = f_in.read()
                with open(dest, "wb") as f_out:
                    f_out.write(stripped_data)
                print(f"  [+] Fixed header & imported: {info['game']} ({fix_type})")
            except Exception as e:
                print(f"  [!] Failed {info['game']}: {e}")

    if unmatched:
        print("\nFirst 10 unmatched ROMs (often region differences like Europe/Japan):")
        for path, name, h in unmatched[:10]:
            print(f"  - {name} (SHA: {h[:8]}...)")

    print("\nRun 'python studio.py --list-games' to see your new list of playable games!")


if __name__ == "__main__":
    main()
