#!/usr/bin/env python3
"""List every game currently importable/imported in your local stable-retro
install (this works for any console stable-retro supports, not just NES --
filter by name if you only want NES titles, e.g. games ending in '-Nes')."""
import stable_retro as retro

games = sorted(retro.data.list_games())
for game in games:
    print(game)

print(f"\n{len(games)} games known to stable-retro.")
print("Note: a game needs its ROM imported (see README) before you can actually play it.")
