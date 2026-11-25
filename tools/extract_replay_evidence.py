"""Export data.json variables from a replay without assuming game-specific meanings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('Output already exists; choose a new path')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import custom_integrations
    custom_integrations.register()
    import stable_retro as retro
    movie = retro.Movie(str(args.replay))
    if not movie.step():
        p.error('Empty replay')
    game = movie.get_game()
    definition = Path(retro.data.get_file_path(game, 'data.json'))
    config = json.loads(definition.read_text())
    names = sorted(config.get('info', {}))
    env = retro.make(game=game, state=None, players=movie.players,
                     use_restricted_actions=retro.Actions.ALL, render_mode='rgb_array')
    try:
        state = movie.get_state()
        if state:
            env.initial_state = state
        _, initial = env.reset()
        fps = float(env.em.get_screen_rate())
        previous = {name: initial[name] for name in names if name in initial}
        baseline = dict(previous)
        changes, frames = [], 0
        while movie.step():
            keys = [movie.get_key(button, player) for player in range(movie.players)
                    for button in range(env.num_buttons)]
            _, _, terminated, truncated, info = env.step(keys)
            frames += 1
            values = {name: info[name] for name in names if name in info}
            delta = {name: {'before': previous.get(name), 'after': value}
                     for name, value in values.items() if previous.get(name) != value}
            if delta or terminated or truncated:
                changes.append({'frame': frames, 'seconds': frames / fps, 'changes': delta,
                                'integration_terminated': bool(terminated),
                                'integration_truncated': bool(truncated)})
            previous = values
        result = {'game': game, 'players': movie.players, 'fps': fps,
                  'frames': frames, 'duration_seconds': frames / fps,
                  'replay_sha256': hashlib.sha256(args.replay.read_bytes()).hexdigest(),
                  'data_json': config, 'initial_values': baseline, 'final_values': previous,
                  'missing_variables': sorted(set(names) - previous.keys()),
                  'warning': 'Raw integration changes, not verified semantic events. Variable suffixes do not establish player identities. Menu resets, wraparound and integration errors can change values.',
                  'events': changes}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, default=lambda x: x.item()), encoding='utf-8')
        print(json.dumps({k: result[k] for k in ('game', 'players', 'frames', 'duration_seconds', 'initial_values', 'final_values', 'missing_variables')}))
        print('Variable-change records:', len(changes))
    finally:
        env.close()


if __name__ == '__main__':
    main()
