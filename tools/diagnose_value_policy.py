"""Read-only fixed-state ablations and Monte Carlo value calibration."""
import argparse
import gc
import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game', required=True)
    p.add_argument('--integration-dir', required=True)
    p.add_argument('--state', action='append', required=True)
    p.add_argument('--checkpoint', action='append', default=[])
    p.add_argument('--config', help='Exact games.json snapshot used by this checkpoint')
    p.add_argument('--output', help='New JSONL result file; existing files are never overwritten')
    args = p.parse_args()
    output = open(args.output, 'x', buffering=1) if args.output else None
    if args.config:
        from rl.vars import set_config_path
        set_config_path(args.config)
    import custom_integrations
    custom_integrations.INTEGRATIONS_DIR = args.integration_dir
    custom_integrations.register()
    from rl.env import make_env, TrainingSpec
    from rl.simulators import get_simulator
    from rl.afterstate import AfterstateAgent, torch
    torch.set_num_threads(1)
    for state in args.state:
        spec = TrainingSpec(args.game, {'state': state})
        sim = get_simulator(spec.afterstate_config['simulator'], config=spec.afterstate_config)
        env = make_env(args.game, {'state': state})
        penalty = spec.afterstate_config['terminal_penalty'] * sim.reward_scale
        for ck in [None] + args.checkpoint:
            agent = AfterstateAgent(sim.feature_dim)
            if ck:
                agent.load(ck)
            else:
                for param in agent.val_net.parameters():
                    param.data.zero_()
            obs, info = env.reset(seed=17)
            initial_lines = info[sim.lines_var]
            pending = None
            records = []
            mismatch_details = []
            mismatch = overrides = n = waits = 0
            rewards = []
            term = trunc = False
            for _ in range(5000):
                cs = sim.get_candidates(obs)
                action = None
                if pending is None and cs:
                    action, predicted, imm = agent.select_action(cs)
                    overrides += int(action != max(cs, key=lambda c:c['immediate_reward'])['action'])
                    selected = next(c for c in cs if c['action'] == action)
                    pending = (obs.copy(), dict(info), predicted, imm, selected.get('lines_cleared'), action)
                    n += 1
                old_info = info
                obs, _, term, trunc, info = env.step(action)
                if action is None:
                    waits += 1
                if term:
                    r = penalty
                    if pending is not None:
                        r += sim.observed_reward(pending[0], obs, pending[1], info, True)
                    rewards.append(r)
                    break
                if info.get('afterstate_discontinuity'):
                    raise RuntimeError('Discontinuity needs a separate return segment')
                if pending is not None and info.get('afterstate_ready', True):
                    before, before_info, pred, imm, predicted_lines, chosen_action = pending
                    feat = sim.encode_observation(obs)
                    r = sim.observed_reward(before, obs, before_info, info)
                    mismatch += int(not np.array_equal(feat, pred))
                    if not np.array_equal(feat, pred):
                        mismatch_details.append({'placement':n, 'predicted_lines':predicted_lines,
                            'action':int(chosen_action),
                            'before_observation':before.tolist(),
                            'after_observation':obs.tolist(),
                            'predicted_features':pred.tolist(),
                            'observed_lines':info[sim.lines_var]-before_info[sim.lines_var],
                            'cells_differing':int(np.count_nonzero(feat[:sim.rows*sim.cols]!=pred[:sim.rows*sim.cols])),
                            'predicted_reward':round(imm,3),'observed_reward':round(r,3)})
                    with torch.no_grad():
                        value = float(agent.val_net(torch.tensor(feat)))
                    records.append((len(rewards), value))
                    rewards.append(r)
                    pending = None
                if trunc or n >= 200:
                    break
            returns = np.zeros(len(rewards) + 1)
            for i in range(len(rewards)-1, -1, -1):
                returns[i] = rewards[i] + agent.gamma * returns[i+1]
            errors = [v - returns[i+1] for i,v in records]
            result = json.dumps({'state':state, 'checkpoint':ck or 'immediate_only',
                'placements':n,'lines_gained':info[sim.lines_var]-initial_lines,
                'terminal':term,'truncated':trunc,'waits':waits,'mismatches':mismatch,
                'value_overrides':overrides,'discounted_return':round(float(returns[0]),3),
                'mismatch_details':mismatch_details,
                'mean_predicted_value':round(float(np.mean([v for _,v in records])),3) if records else None,
                'mean_value_overestimate':round(float(np.mean(errors)),3) if errors and term else None,
                'first_predictions_and_actual_returns':[(round(v,2),round(float(returns[i+1]),2)) for i,v in records[:4]]})
            print(result, flush=True)
            if output:
                print(result, file=output, flush=True)
        env.close()
        del env
        gc.collect()
    if output:
        output.close()

if __name__ == '__main__':
    main()
