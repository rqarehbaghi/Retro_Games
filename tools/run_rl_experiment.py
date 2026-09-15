"""Run an isolated, manifest-driven learned-RL experiment in WSL.

Snapshots configs and source, takes an exclusive worker lock, logs training,
then evaluates each requested policy on an explicit state schedule. No promotion
to the production config is automatic.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def dump(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def evaluate(game, config, states, checkpoint, max_placements):
    import numpy as np
    from rl.vars import set_config_path
    from rl.env import TrainingSpec, make_env
    from rl.simulators import get_simulator
    from rl.afterstate import AfterstateAgent
    set_config_path(str(config))
    rows = []
    for state in states:
        spec = TrainingSpec(game, {'state': state})
        sim = get_simulator(spec.afterstate_config['simulator'], config=spec.afterstate_config)
        agent = AfterstateAgent(sim.feature_dim, device='cpu')
        if checkpoint:
            agent.load(str(checkpoint))
        else:
            for param in agent.val_net.parameters():
                param.data.zero_()
        env = make_env(game, {'state': state})
        try:
            obs, info = env.reset(seed=17)
            initial_lines = info[sim.lines_var]
            placements = waits = mismatches = 0
            pending = None
            terminal = truncated = False
            for _ in range(10000):
                action = None
                if pending is None:
                    candidates = sim.get_candidates(obs)
                    if candidates:
                        action, pending, _ = agent.select_action(candidates, epsilon=0)
                        placements += 1
                obs, _, terminal, truncated, info = env.step(action)
                waits += int(action is None)
                if pending is not None and info.get('afterstate_ready', True) and not terminal:
                    mismatches += int(not np.array_equal(pending, sim.encode_observation(obs)))
                    pending = None
                if info.get('afterstate_discontinuity'):
                    pending = None
                if terminal or truncated or (placements >= max_placements and pending is None):
                    break
            rows.append({'state': state, 'placements': placements,
                         'lines': info[sim.lines_var] - initial_lines,
                         'terminal': terminal, 'truncated': truncated,
                         'capped': not (terminal or truncated), 'waits': waits,
                         'mismatches': mismatches})
        finally:
            env.close()
            del env
            gc.collect()
    return rows


def run(manifest_path):
    import fcntl
    import numpy as np
    manifest = json.loads(manifest_path.read_text())
    output = (ROOT / manifest['output']).resolve()
    output.relative_to(ROOT / 'checkpoints')
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output.parent / '.worker.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Reusing a directory would mix configs, metrics and checkpoint history.
        output.mkdir(exist_ok=False)
        status = {'phase': 'preparing', 'pid': os.getpid(), 'started': time.time(),
                  'manifest': str(manifest_path), 'output': str(output)}
        dump(output / 'status.json', status)
        dump(output / 'manifest.json', manifest)
        source_config = json.loads((ROOT / manifest.get('config', 'games.json')).read_text())
        dump(output / 'baseline_games.json', source_config)
        cfg = source_config['games'][manifest['game']]['training']['afterstate']
        cfg.update(manifest.get('afterstate_overrides', {}))
        dump(output / 'games.json', source_config)
        fingerprints = {}
        for source in ['train.py', 'rl/afterstate.py', 'rl/control.py', 'rl/env.py', 'rl/macro.py',
                       'rl/features.py', 'rl/vars.py', 'rl/simulators/base.py',
                       'rl/simulators/grid_placement.py', 'tools/run_rl_experiment.py']:
            path = ROOT / source
            dest = output / 'source' / source
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            fingerprints[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        dump(output / 'source_hashes.json', fingerprints)
        try:
            import custom_integrations
            custom_integrations.INTEGRATIONS_DIR = manifest['integration_dir']
            custom_integrations.register()
            from rl.afterstate import torch, train_afterstate
            from rl.env import TrainingSpec
            from rl.vars import set_config_path
            torch.set_num_threads(1)
            seed = manifest['seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            set_config_path(str(output / 'games.json'))
            overrides = {'state': ','.join(manifest['train_states']), 'seed': seed}
            spec = TrainingSpec(manifest['game'], overrides)
            options = dict(game=manifest['game'], save_dir=str(output / 'model'),
                           lr=None, gamma=0.99, batch_size=64, device='cuda',
                           ent_coef=0.2, ent_coef_final=0.03, explore_steps=6000,
                           timesteps=10000, resume=None, save_every=2500)
            options.update(manifest.get('train_options', {}))
            dump(output / 'effective_options.json', options)
            status['phase'] = 'training'
            dump(output / 'status.json', status)
            with open(output / 'train.log', 'w', buffering=1) as log, contextlib.redirect_stdout(log):
                train_afterstate(SimpleNamespace(**options), spec, overrides)
            gc.collect()
            policies = [('baseline_value_disabled', output / 'baseline_games.json', None),
                        ('baseline_v36', output / 'baseline_games.json', Path(manifest['baseline'])),
                        ('experiment_value_disabled', output / 'games.json', None)]
            for step in manifest.get('evaluate_steps', []):
                policies.append((f'learned_{step}', output / 'games.json', output / 'model' / f'ckpt_{step}_steps.zip'))
            status['phase'] = 'evaluating'
            dump(output / 'status.json', status)
            results = {}
            with open(output / 'evaluation.log', 'w', buffering=1) as log, contextlib.redirect_stdout(log):
                for name, config, ck in policies:
                    results[name] = evaluate(manifest['game'], config, manifest['eval_states'], ck,
                                             manifest.get('max_eval_placements', 1000))
                    dump(output / 'results.json', results)
            status.update(phase='complete', finished=time.time())
            dump(output / 'status.json', status)
            print(json.dumps(results), flush=True)
        except BaseException:
            status.update(phase='failed', finished=time.time(), error=traceback.format_exc())
            dump(output / 'status.json', status)
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    run(p.parse_args().manifest.resolve())
