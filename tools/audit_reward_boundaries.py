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
    p.add_argument('--continue-after-boundary', action='store_true',
                   help='Trace repeated skip/finalization calls until play resumes or terminates.')
    p.add_argument('--max-steps', type=int, default=1500)
    p.add_argument('--player', type=int)
    p.add_argument('--players', type=int)
    p.add_argument('--frame-dir', type=Path,
                   help='Optionally save a rendered frame for each discontinuity.')
    a = p.parse_args()
    import custom_integrations
    custom_integrations.INTEGRATIONS_DIR = a.integration_dir
    custom_integrations.register()
    from rl.vars import set_config_path
    set_config_path(a.config)
    from rl.env import make_env, TrainingSpec
    from rl.simulators import get_simulator
    overrides = {k: v for k, v in {'state': a.state, 'player': a.player,
                                    'players': a.players}.items() if v is not None}
    s = TrainingSpec(a.game, overrides)
    sim = get_simulator(s.afterstate_config['simulator'], config=s.afterstate_config)
    env = make_env(a.game, overrides)
    core = env.env.env
    original = core.finalize_macro_step
    boundary = {}
    def finalize(obs, info, term, trunc):
        boundary.clear()
        end_cfg = core.spec_.episode_end or {}
        skip_cfg = core.spec_.skip_while or {}
        boundary.update(pre_skip=core._skipping(core._ram(), core._seen(info)),
                        pre_lines=core.vars.read(sim.lines_var, core._ram(), {}),
                        pre_episode_end=core.vars.read(end_cfg.get('var'), core._ram(), {})
                        if end_cfg.get('var') else None,
                        pre_skip_value=core.vars.read(skip_cfg.get('var'), core._ram(), {})
                        if skip_cfg.get('var') else None)
        return original(obs, info, term, trunc)
    core.finalize_macro_step = finalize
    if a.frame_dir:
        a.frame_dir.mkdir(parents=True, exist_ok=False)
    try:
        with open(a.output, 'x', buffering=1) as log:
            obs, info = env.reset(seed=17)
            pending = None
            for step in range(a.max_steps):
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
                               legacy_replay_reward=0. if ni.get('afterstate_discontinuity') else measured,
                               discontinuity=bool(ni.get('afterstate_discontinuity')),
                               terminal=bool(term), truncated=bool(trunc),
                               episode_end_var=(s.episode_end or {}).get('var'),
                               episode_end_value=ni.get((s.episode_end or {}).get('var')),
                               skip_var=(s.skip_while or {}).get('var'),
                               skip_value=ni.get((s.skip_while or {}).get('var')),
                               skip_frames=ni.get('skip_frames'),
                               skip_timeout=bool(ni.get('skip_timeout')),
                               skip_resumed_on_change=bool(ni.get('skip_resumed_on_change')),
                               **boundary)
                    print(json.dumps(row), file=log)
                    if row['discontinuity']:
                        print(json.dumps(row))
                        if a.frame_dir:
                            from PIL import Image
                            Image.fromarray(core.render()).save(
                                a.frame_dir / ('boundary_%04d.png' % step))
                        if not a.continue_after_boundary:
                            break
                    else:
                        pending = None
                obs, info = nxt, ni
                if term or trunc:
                    break
    finally:
        env.close()

if __name__ == '__main__':
    main()
