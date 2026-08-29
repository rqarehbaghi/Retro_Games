#!/usr/bin/env python3
"""
"Human coach" pretraining: record yourself playing a level or two with
play_and_record.py --human, then use this script to train a PPO network
to imitate what you did, before train.py fine-tunes it further with RL.

This is the actual mechanism behind "record 1-2 levels, then have
training continue from there" -- it doesn't rely on the agent randomly
stumbling onto good play; it directly supervises the network toward what
you demonstrated.

DURATION AWARENESS: a common mistake here is sampling the human's button
presses at a fixed interval and losing exactly how long each button was
actually held -- which throws away the short-hop-vs-full-jump
information you're trying to teach in the first place. This script
instead walks the recording frame-by-frame, finds each contiguous run
where the same combo was held, and matches that run's actual length
against train.py's ACTION_TABLE to pick the closest (combo, hold_frames)
entry -- so a quick tap of A genuinely becomes a different training
label than a long hold of A, matching how you actually played it.

Usage:
    # 1. Record yourself playing (see play_and_record.py --human)
    python play_and_record.py --game SuperMarioBros3-Nes-v0 \\
        --human --record-dir ./human_demos

    # 2. Turn that into a warm-start checkpoint:
    python pretrain_imitation.py --game SuperMarioBros3-Nes-v0 \\
        --demo-dir ./human_demos --epochs 25 \\
        --output ./checkpoints/SuperMarioBros3-Nes-v0/pretrained_human_bc.zip

    # 3. Fine-tune it with RL (note the gentler --lr/--ent-coef):
    python train.py --game SuperMarioBros3-Nes-v0 \\
        --resume-from ./checkpoints/SuperMarioBros3-Nes-v0/pretrained_human_bc.zip \\
        --lr 3e-5 --ent-coef 0.001 --iterations 300
"""
import argparse
import glob
import os
from collections import deque

import cv2
import numpy as np
import stable_retro as retro
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from train import ACTION_TABLE, VariableHoldDiscretizer, WarpFrame, jump_action_indices


def warp(frame, size=84):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def closest_action_for_run(combo_names, run_length, action_table):
    """Among ACTION_TABLE entries whose button combo matches what was
    actually held, pick the one whose hold_frames is closest to how long
    the human actually held it. Returns None if no entry's combo matches
    at all (the human pressed something not represented in the table --
    skipped rather than mapped to something misleading)."""
    combo_set = frozenset(combo_names)
    candidates = [
        (i, hold) for i, (combo, hold) in enumerate(action_table)
        if frozenset(combo) == combo_set
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(c[1] - run_length))[0]


def extract_demo_pairs(bk2_path, action_table):
    """Walks a .bk2 frame-by-frame (the official stable-retro Movie API --
    see stable-retro's Python docs for this exact pattern), detects
    contiguous button-hold runs, and yields (stacked_4_frame_obs,
    action_index) pairs -- one per detected run, matching how the policy
    will actually be queried once per decision at inference time."""
    movie = retro.Movie(bk2_path)
    movie.step()  # primes movie.get_state() -- see stable-retro's documented Movie usage

    env = retro.make(
        game=movie.get_game(),
        state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
        render_mode="rgb_array",
    )
    env.initial_state = movie.get_state()
    obs, info = env.reset()
    buttons = env.unwrapped.buttons

    # One stack entry per DECISION (button-run), not per emulator frame.
    # This must match what the deployed policy sees: at inference, WarpFrame
    # sits above VariableHoldDiscretizer, so VecFrameStack's 4 slots each hold
    # the observation at a decision boundary -- frames separated by each
    # action's 4-20 frame hold, NOT 4 consecutive emulator frames. A human
    # button-run is exactly one decision, so the stack advances once per run
    # (with the run's final frame), and the observation labeled with a run's
    # action is the stack as it stood when that run began -- the last 4
    # decision-boundary frames the human had seen when they made the choice.
    # (A per-emulator-frame stack here would train on ~4-frame-span stacks
    # while the policy plays on ~16-80-frame-span stacks -- a distribution
    # mismatch that quietly degrades the warm start.)
    frame_stack = deque(maxlen=4)
    frame_stack.extend([warp(obs)] * 4)

    pairs = []
    skipped = 0

    current_combo = None
    current_run_len = 0
    run_start_stack = None

    def close_run(end_frame):
        """Close the run that just finished: emit its (obs, action) pair and
        advance the per-decision frame stack with the run's final frame --
        even for runs whose combo has no ACTION_TABLE match (time still
        passed; the stack must advance regardless)."""
        nonlocal skipped
        if current_combo is None or current_run_len == 0:
            return
        action_idx = closest_action_for_run(current_combo, current_run_len, action_table)
        if action_idx is None:
            skipped += 1
        else:
            pairs.append((np.stack(run_start_stack, axis=0), action_idx))
        frame_stack.append(end_frame)

    last_obs = obs
    while movie.step():
        keys = [movie.get_key(i, 0) for i in range(env.num_buttons)]
        obs, reward, terminated, truncated, info = env.step(keys)
        combo = frozenset(b for b, pressed in zip(buttons, keys) if pressed)

        if combo != current_combo:
            # last_obs is the final frame of the run that just ended -- the
            # screen the human was looking at when they chose the new combo.
            close_run(warp(last_obs))
            current_combo = combo
            current_run_len = 0
            run_start_stack = list(frame_stack)

        current_run_len += 1
        last_obs = obs

        if terminated or truncated:
            break

    close_run(warp(last_obs))
    env.close()

    return pairs, skipped


def build_dataset(demo_dir, action_table):
    bk2_files = sorted(glob.glob(os.path.join(demo_dir, "*.bk2")))
    if not bk2_files:
        raise SystemExit(f"No .bk2 files found in {demo_dir} -- record with play_and_record.py --human first.")

    all_pairs = []
    total_skipped = 0
    for path in bk2_files:
        print(f"Extracting {path} ...")
        pairs, skipped = extract_demo_pairs(path, action_table)
        print(f"  {len(pairs)} decisions extracted, {skipped} runs skipped (no matching action)")
        all_pairs.extend(pairs)
        total_skipped += skipped

    if not all_pairs:
        raise SystemExit("No usable (observation, action) pairs extracted from any recording.")

    print(f"\nTotal: {len(all_pairs)} decisions across {len(bk2_files)} recording(s), {total_skipped} skipped")
    jump_count = sum(1 for _, a in all_pairs if a in jump_action_indices(action_table))
    print(f"Jump-action decisions: {jump_count} ({100 * jump_count / len(all_pairs):.1f}% of total)")

    return all_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True, help="stable-retro game id -- used to build the policy's env shape, must match what you recorded")
    parser.add_argument("--demo-dir", required=True, help="Folder of .bk2 recordings from play_and_record.py --human")
    parser.add_argument("--output", required=True, help="Where to save the resulting checkpoint (.zip), for use as --resume-from in train.py")
    parser.add_argument("--epochs", type=int, default=25, help="Passes over the demonstration data (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=64, help="(default: %(default)s)")
    parser.add_argument("--jump-weight", type=float, default=8.0, help="Loss multiplier for jump-action examples. Running dominates typical demo data (often 85%%+ of frames) while jumping is rare, so without upweighting, imitation learning mostly just learns to run and undervalues jump timing. (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Supervised learning rate for this pretraining pass. (default: %(default)s)")
    args = parser.parse_args()

    pairs = build_dataset(args.demo_dir, ACTION_TABLE)
    jump_idx = jump_action_indices(ACTION_TABLE)

    observations = np.stack([p[0] for p in pairs]).astype(np.float32)
    actions = np.array([p[1] for p in pairs], dtype=np.int64)
    weights = np.array([args.jump_weight if a in jump_idx else 1.0 for a in actions], dtype=np.float32)

    # Build a real PPO model with the exact same architecture train.py
    # will use, so the saved checkpoint is a drop-in --resume-from for it.
    def make_dummy_env():
        # Must reproduce train.py's action space EXACTLY. VariableHoldDiscretizer
        # collapses the raw MultiBinary buttons into Discrete(len(ACTION_TABLE)).
        # Without it, PPO builds a MultiBinary/Bernoulli policy whose log_prob
        # expects a per-button vector, so evaluate_actions() on our discrete
        # action indices fails with a shape mismatch ([B] vs [B, n_buttons]).
        # It also guarantees the saved checkpoint is a drop-in --resume-from for
        # train.py. Reward-shaping wrappers are intentionally omitted here: they
        # change only the reward, not the observation/action space.
        env = retro.make(game=args.game, state=retro.State.DEFAULT, render_mode="rgb_array")
        env = VariableHoldDiscretizer(env, ACTION_TABLE)
        env = WarpFrame(env)
        return env

    dummy_env = VecFrameStack(DummyVecEnv([make_dummy_env]), n_stack=4)
    # verbose=1 so the saved checkpoint carries it: PPO.load() restores this
    # attribute, and a verbose=0 checkpoint silences SB3's rollout table for
    # the entire fine-tuning run that resumes from it.
    model = PPO("CnnPolicy", dummy_env, verbose=1)
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    device = policy.device
    print(f"Device: {device}")
    print(f"Training on {len(pairs)} examples for {args.epochs} epochs...")

    n = len(pairs)
    for epoch in range(args.epochs):
        perm = np.random.permutation(n)
        total_loss = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            obs_batch = torch.as_tensor(observations[idx], device=device)  # already (B, 4, 84, 84)
            act_batch = torch.as_tensor(actions[idx], device=device)
            weight_batch = torch.as_tensor(weights[idx], device=device)

            _, log_prob, _ = policy.evaluate_actions(obs_batch, act_batch)
            loss = -(log_prob * weight_batch).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        print(f"Epoch {epoch + 1}/{args.epochs}: loss={total_loss / n:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"\nSaved imitation-pretrained checkpoint: {args.output}")
    print(f"Fine-tune it with: python train.py --game {args.game} --resume-from {args.output} --lr 3e-5 --ent-coef 0.001")


if __name__ == "__main__":
    main()
