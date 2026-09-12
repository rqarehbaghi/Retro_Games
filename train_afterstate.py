#!/usr/bin/env python3
"""
Generic Deep Afterstate Value Network Trainer (Lookahead RL).

Completely game-agnostic:
  - Resolves the game-specific afterstate simulator dynamically from games.json:
      spec.afterstate_config["simulator"]
  - Never hardcodes game rules, grid dimensions, piece sets, or scoring mechanics.
  - Can be used for any game with a lookahead simulator registered under rl.simulators.
"""
import argparse
import os
import sys
import numpy as np

# Ensure root directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import custom_integrations
    custom_integrations.register()
except Exception:                                                # noqa: BLE001
    pass

from rl.afterstate import AfterstateAgent, AfterstateReplayBuffer
from rl.env import make_env, TrainingSpec
from rl.simulators import get_simulator
from rl.vars import GameVars


def parse_args():
    p = argparse.ArgumentParser(description="Train Generic Deep Afterstate Value Network")
    p.add_argument("--game", required=True, help="Retro game identifier (e.g. TetrisTime-Nes-v0)")
    p.add_argument("--state", default=None, help="Comma-separated start state(s)")
    p.add_argument("--episodes", type=int, default=3000, help="Number of training episodes")
    p.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--batch-size", type=int, default=64, help="Minibatch size for TD updates")
    p.add_argument("--target-sync-every", type=int, default=500, help="Steps between target network syncs")
    p.add_argument("--epsilon-start", type=float, default=0.20, help="Initial exploration rate")
    p.add_argument("--epsilon-final", type=float, default=0.01, help="Final exploration rate")
    p.add_argument("--epsilon-decay", type=int, default=1500, help="Episodes over which to decay epsilon")
    p.add_argument("--device", default="cuda", help="Device ('cuda' or 'cpu')")
    p.add_argument("--save-dir", default=None, help="Directory to save model checkpoints")
    p.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N episodes")
    p.add_argument("--resume", default=None, help="Path to checkpoint (.pt) to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.state:
        overrides["state"] = args.state

    spec = TrainingSpec(args.game, overrides)
    sim_name = spec.afterstate_config.get("simulator") or spec.features_name or spec.game
    simulator = get_simulator(sim_name)

    if simulator is None:
        sys.exit(
            f"Error: Game '{args.game}' does not have an afterstate simulator registered.\n"
            f"Please register a BaseAfterstateSimulator in rl/simulators/ and declare it in games.json under 'afterstate'."
        )

    save_dir = args.save_dir or os.path.join("checkpoints", f"{args.game}_afterstate")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Initializing Afterstate RL for: {args.game}")
    print(f"  Simulator: {sim_name} (Feature Dim: {simulator.feature_dim})")
    print(f"  Start State(s): {spec.state}")
    print(f"  Target Episodes: {args.episodes}")
    print(f"  Device: {args.device}")

    env = make_env(args.game, overrides=overrides)
    vars = GameVars(spec.game, entry=spec.entry)

    agent = AfterstateAgent(
        input_dim=simulator.feature_dim,
        device=args.device,
        lr=args.lr,
        gamma=args.gamma
    )
    replay = AfterstateReplayBuffer(capacity=50000)

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming weights from {args.resume}")
        agent.load(args.resume)

    total_steps = 0
    recent_rewards = []
    recent_steps = []

    print("\nBeginning training...")
    header = f"{'Episode':>8} | {'Epsilon':>7} | {'Steps':>6} | {'Reward':>9} | {'AvgRew (50)':>11} | {'Loss':>7}"
    print(header)
    print("-" * len(header))

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        prev_afterstate_feat = None
        ep_loss = 0.0
        loss_updates = 0

        eps_progress = min(1.0, ep / float(max(1, args.epsilon_decay)))
        epsilon = args.epsilon_start + (args.epsilon_final - args.epsilon_start) * eps_progress

        while not done:
            ram = getattr(env.unwrapped, "ram", None) if hasattr(env, "unwrapped") else None
            candidates = simulator.get_candidates(obs=obs, ram=ram, info=info, vars=vars, spec=spec)

            action, afterstate_feat = agent.select_action(candidates, epsilon=epsilon)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += float(reward)
            ep_steps += 1
            total_steps += 1

            if prev_afterstate_feat is not None:
                replay.push(prev_afterstate_feat, reward, afterstate_feat, done)

            prev_afterstate_feat = afterstate_feat

            if len(replay) >= args.batch_size:
                loss = agent.update(replay, batch_size=args.batch_size)
                ep_loss += loss
                loss_updates += 1

            if total_steps % args.target_sync_every == 0:
                agent.sync_target_network()

            obs = next_obs

        # Terminal transition penalty/terminal state
        if prev_afterstate_feat is not None:
            replay.push(prev_afterstate_feat, -20.0, None, True)

        recent_rewards.append(ep_reward)
        recent_steps.append(ep_steps)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
            recent_steps.pop(0)

        avg_rew = np.mean(recent_rewards)
        avg_loss = (ep_loss / max(1, loss_updates)) if loss_updates > 0 else 0.0

        if ep % 5 == 0 or ep == 1:
            print(f"{ep:8d} | {epsilon:7.3f} | {ep_steps:6d} | {ep_reward:9.1f} | {avg_rew:11.1f} | {avg_loss:7.4f}")

        if ep % args.save_every == 0:
            ckpt_path = os.path.join(save_dir, f"afterstate_ep_{ep}.pt")
            agent.save(ckpt_path)
            print(f"  --> Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(save_dir, "afterstate_final.pt")
    agent.save(final_path)
    print(f"\nTraining complete. Final weights saved to {final_path}")
    env.close()


if __name__ == "__main__":
    main()
