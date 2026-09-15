"""Measure counter rewards at ordinary and discontinuous macro boundaries."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    p = argparse.ArgumentParser()
    for key in ('game', 'state', 'config', 'integration-dir', 'output'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    import custom_integrations
    custom_integrations.INTEGRATIONS_DIR = a.integration_dir
    custom_integrations.register()
    from rl.vars import set_config_path
    set_config_path(a.config)
    from rl.env import make_env, TrainingSpec
    from rl.simulators import get_simulator
    s = TrainingSpec(a.game, {'state': a.state})
    sim = get_simulator(s.afterstate_config['simulator'], config=s.afterstate_config)
    env = make_env(a.game, {'state': a.state})
    core = env.env.env
    original = core.finalize_macro_step
    boundary = {}
    def finalize(obs, info, term, trunc):
        boundary.clear()
        boundary.update(pre_skip=core._skipping(core._ram(), core._seen(info)),
                        pre_lines=core.vars.read(sim.lines_var, core._ram(), {}))
        return original(obs, info, term, trunc)
    core.finalize_macro_step = finalize
    try:
        with open(a.output, 'x', buffering=1) as log:
            obs, info = env.reset(seed=17)
            pending = None
            for step in range(1500):
                action = None
                if pending is None:
                    cs = sim.get_candidates(obs)
                    if cs:
                        c = max(cs, key=lambda x: x['immediate_reward'])
                        pending = (obs.copy(), dict(info), c)
                        action = c['action']
                nxt, er, term, trunc, ni = env.step(action)
                if pending and (ni.get('afterstate_ready') or term or ni.get('afterstate_discontinuity')):
                    before, bi, c = pending
                    measured = sim.observed_reward(before, nxt, bi, ni, term)
                    row = dict(step=step, before_lines=bi[sim.lines_var], after_lines=ni[sim.lines_var],
                               predicted_lines=c['lines_cleared'], measured_reward=measured,
                               replay_reward=0. if ni.get('afterstate_discontinuity') else measured,
                               discontinuity=bool(ni.get('afterstate_discontinuity')), **boundary)
                    print(json.dumps(row), file=log)
                    if row['discontinuity']:
                        print(json.dumps(row))
                        break
                    pending = None
                obs, info = nxt, ni
                if term or trunc:
                    break
    finally:
        env.close()

if __name__ == '__main__':
    main()
