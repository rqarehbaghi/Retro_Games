"""Trace configured macro variables and capture excluded piece types in real frames."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    p = argparse.ArgumentParser()
    for name in ('game', 'state', 'config', 'integration-dir', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--probe-type', type=int)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    import custom_integrations
    custom_integrations.INTEGRATIONS_DIR = a.integration_dir
    custom_integrations.register()
    from rl.vars import set_config_path
    set_config_path(a.config)
    from rl.env import make_env, TrainingSpec
    from rl.simulators import get_simulator
    from PIL import Image
    spec = TrainingSpec(a.game, {'state': a.state})
    sim = get_simulator(spec.afterstate_config['simulator'], config=spec.afterstate_config)
    env = make_env(a.game, {'state': a.state})
    macro = env.env
    core = macro.env
    original = core.step_raw_frame
    frame_id = 0
    last = None
    class ProbeReady(Exception):
        pass
    with open(out / 'frames.jsonl', 'x', buffering=1) as log:
        def traced(buttons):
            nonlocal frame_id, last
            result = original(buttons)
            frame_id += 1
            values = {k: macro._read_var(v, None) for k, v in
                      [('type', macro.piece_type_var), ('row', macro.row_var),
                       ('col', macro.col_var), ('rot', macro.rot_var)]}
            excluded = values['type'] not in macro.cfg['valid_piece_types']
            record = dict(frame=frame_id, buttons=buttons, **values)
            if excluded and values != last:
                path = out / ('frame_%06d.png' % frame_id)
                Image.fromarray(result[0]).save(path)
                record['image'] = path.name
            print(json.dumps(record), file=log)
            last = values
            if values['type'] == a.probe_type and values['row'] == 12:
                raise ProbeReady()
            return result
        core.step_raw_frame = traced
        try:
            obs, info = env.reset(seed=17)
            pending = False
            placements = 0
            for _ in range(5000):
                action = None
                if not pending:
                    candidates = sim.get_candidates(obs)
                    if candidates:
                        action = max(candidates, key=lambda c: c['immediate_reward'])['action']
                        pending = True
                        placements += 1
                obs, _, term, trunc, info = env.step(action)
                if info.get('afterstate_ready'):
                    pending = False
                if term or trunc or placements >= 14:
                    break
        except ProbeReady:
            import numpy as np
            from rl.features import grid_from_frame
            measurements = []
            for rotation in range(4):
                frame = core.render()
                grid = grid_from_frame(frame, spec.grid)
                points = np.argwhere(grid[:10] > 0)
                low = points.min(axis=0)
                record = {'type': macro._read_var(macro.piece_type_var),
                          'rot': macro._read_var(macro.rot_var),
                          'cells': (points - low).tolist(),
                          'col_offset': int(macro._read_var(macro.col_var) - low[1]),
                          'row_offset': int(macro._read_var(macro.row_var) - low[0])}
                measurements.append(record)
                Image.fromarray(frame).save(out / ('rotation_%d.png' % record['rot']))
                for buttons in [[macro.rotate_button]] * macro.tap_hold + [[]] * macro.tap_release:
                    original(buttons)
            with open(out / 'rotations.json', 'x') as measured:
                json.dump(measurements, measured, indent=2)
        finally:
            env.close()
    print(json.dumps({'frames': frame_id, 'placements': placements, 'output': str(out)}))

if __name__ == '__main__':
    main()
