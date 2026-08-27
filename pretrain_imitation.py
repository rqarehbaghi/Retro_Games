#!/usr/bin/env python3
"""
Behavioral Cloning & Imitation Learning (Teacher Mode):
Pre-train a PPO policy network on human gameplay demonstrations!

Instead of waiting 50,000 iterations for random flailing to discover how to jump,
you record 5-10 minutes of human gameplay. This script fits the PPO CnnPolicy
to your demonstrations using Supervised Cross-Entropy Loss, then saves a
warm-started .zip checkpoint ready for PPO fine-tuning.

Usage:
  # Step 1: Record yourself playing for a few levels:
  python play_and_record.py --game SuperMarioBros3-Nes-v0 --human --record-dir ./human_demos

  # Step 2: Pre-train the neural network on your demonstration files (.bk2):
  python pretrain_imitation.py --game SuperMarioBros3-Nes-v0 \\
      --demo-dir ./human_demos --epochs 20 --output ./checkpoints/pretrained_human_bc.zip

  # Step 3: Run PPO to take it from human basics to superhuman perfection:
  python train.py --game SuperMarioBros3-Nes-v0 \\
      --resume-from ./checkpoints/pretrained_human_bc.zip --iterations 500
"""
import argparse
import glob
import os
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from train import DEFAULT_COMBOS, Discretizer, FrameSkip, WarpFrame

class RetroDemoDataset(Dataset):
    """
    Parses retro .bk2 replay recordings and builds (4-frame stack -> action_idx) pairs.
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stacked_obs, action_idx = self.samples[idx]
        # shape: (4, 84, 84) -> float32 [0.0, 1.0] or [0, 255]
        obs_tensor = torch.tensor(stacked_obs, dtype=torch.float32)
        act_tensor = torch.tensor(action_idx, dtype=torch.long)
        return obs_tensor, act_tensor

def map_binary_to_combo(button_array, env_buttons, combos=DEFAULT_COMBOS):
    """
    Matches a raw human button press array to the closest discrete combo index.
    """
    active_buttons = {env_buttons[i] for i, pressed in enumerate(button_array) if pressed}
    best_idx = 0
    best_match_score = -1

    for idx, combo in enumerate(combos):
        combo_set = set(combo)
        if combo_set == active_buttons:
            return idx  # Exact match
        # Partial match scoring
        intersection = len(combo_set & active_buttons)
        if intersection > best_match_score:
            best_match_score = intersection
            best_idx = idx

    return best_idx

def extract_demos_from_bk2(game, bk2_paths, combos=DEFAULT_COMBOS, frame_skip=4):
    print(f"📦 Extracting demonstration frames from {len(bk2_paths)} .bk2 replays (with FrameSkip={frame_skip} alignment)...")
    samples = []
    
    for bk2_file in bk2_paths:
        try:
            movie = retro.Movie(bk2_file)
            movie.step()
            env = retro.make(game=game, state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)
            env.initial_state = movie.get_state()
            env.reset()
            buttons = env.unwrapped.buttons

            frame_queue = deque(maxlen=4)
            step_count = 0
            frame_counter = 0

            while movie.step():
                frame_counter += 1
                keys = []
                for p in range(movie.players):
                    for i in range(env.num_buttons):
                        keys.append(movie.get_key(i, p))

                obs, _, term, trunc, _ = env.step(keys)
                
                # Align with training FrameSkip=4 so observation dynamics match exactly
                if frame_counter % frame_skip != 0:
                    continue

                # Preprocess frame (grayscale 84x84)
                gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

                if len(frame_queue) < 4:
                    while len(frame_queue) < 4:
                        frame_queue.append(resized)
                else:
                    frame_queue.append(resized)

                # Match human action to discrete action space
                combo_idx = map_binary_to_combo(keys, buttons, combos)
                
                # Stack 4 frames (shape: 4, 84, 84)
                stacked_frames = np.array(frame_queue, dtype=np.uint8)
                samples.append((stacked_frames, combo_idx))
                step_count += 1

            env.close()
            print(f"  ✓ Processed {bk2_file}: extracted {step_count} demonstration steps")
        except Exception as e:
            print(f"  ⚠️ Error parsing {bk2_file}: {e}")

    return samples

def main():
    parser = argparse.ArgumentParser(description="Imitation Learning / Behavioral Cloning for Retro AI")
    parser.add_argument("--game", type=str, default="SuperMarioBros3-Nes-v0", help="Retro game identifier")
    parser.add_argument("--demo-dir", type=str, default="./recordings", help="Directory containing human .bk2 replays")
    parser.add_argument("--epochs", type=int, default=25, help="Supervised training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for behavioral cloning")
    parser.add_argument("--jump-weight", type=float, default=8.0, help="Loss multiplier for jump/trick actions (fixes 85% run vs 5% jump class imbalance)")
    parser.add_argument("--output", type=str, default="./checkpoints/pretrained_human_bc.zip", help="Output .zip model")
    args = parser.parse_args()

    bk2_files = glob.glob(os.path.join(args.demo_dir, "*.bk2"))
    if not bk2_files:
        print(f"❌ No .bk2 replay files found in '{args.demo_dir}'!")
        print("Tip: Run `python play_and_record.py --game <game> --human --record-dir ./recordings` first to record human gameplay!")
        return

    samples = extract_demos_from_bk2(args.game, bk2_files, DEFAULT_COMBOS, frame_skip=4)
    if len(samples) < 50:
        print(f"⚠️ Warning: Only {len(samples)} demonstration steps extracted. Record at least 2-3 full minutes of gameplay for best results.")

    # Calculate action frequency distribution
    action_counts = {}
    for _, a in samples:
        action_counts[a] = action_counts.get(a, 0) + 1

    print("\n📊 Human Action Distribution in Demonstrations:")
    jump_indices = set()
    for idx, combo in enumerate(DEFAULT_COMBOS):
        count = action_counts.get(idx, 0)
        pct = (count / max(1, len(samples))) * 100
        is_jump = "A" in combo
        if is_jump:
            jump_indices.add(idx)
        tag = " 🦘 [JUMP/TRICK ACTION]" if is_jump else ""
        combo_name = "+".join(combo) if combo else "NO-OP"
        print(f"  Combo #{idx:02d} [{combo_name:<16}]: {count:>5} frames ({pct:>5.1f}%){tag}")

    # Build Class Weight Tensor to boost rare jump actions
    num_classes = len(DEFAULT_COMBOS)
    weights = np.ones(num_classes, dtype=np.float32)
    for idx in range(num_classes):
        if idx in jump_indices:
            weights[idx] = args.jump_weight
        else:
            # Scale down common running actions if overrepresented
            if action_counts.get(idx, 0) > (0.3 * len(samples)):
                weights[idx] = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"\n⚖️ Class-Loss Balancing Applied (Jump Actions boosted by {args.jump_weight}x)\n")

    dataset = RetroDemoDataset(samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Initialize a dummy environment to setup the PPO policy architecture
    def make_dummy():
        e = retro.make(game=args.game)
        e = Discretizer(e, DEFAULT_COMBOS)
        e = WarpFrame(e)
        e = FrameSkip(e, skip=4)
        return e

    venv = DummyVecEnv([make_dummy])
    venv = VecFrameStack(venv, n_stack=4)

    print("🧠 Initializing Stable-Baselines3 PPO Actor-Critic Network...")
    ppo_model = PPO("CnnPolicy", venv, verbose=0)
    policy_net = ppo_model.policy

    # Behavioral Cloning: Supervised optimization on policy action logits
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    policy_net.to(device)
    policy_net.train()

    print(f"🚀 Training Behavioral Cloning on {len(samples):,} frames for {args.epochs} epochs on {device}...")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        correct = 0
        jump_correct = 0
        jump_total = 0
        total = 0

        for obs_batch, act_batch in dataloader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            optimizer.zero_grad()
            
            # Forward pass through policy distribution
            distribution = policy_net.get_distribution(obs_batch)
            logits = distribution.distribution.logits

            loss = criterion(logits, act_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * obs_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == act_batch).sum().item()
            total += obs_batch.size(0)

            # Measure Jump action accuracy specifically
            for p, a in zip(preds, act_batch):
                if a.item() in jump_indices:
                    jump_total += 1
                    if p.item() == a.item():
                        jump_correct += 1

        epoch_loss = total_loss / max(1, total)
        epoch_acc = (correct / max(1, total)) * 100.0
        jump_acc = (jump_correct / max(1, jump_total)) * 100.0 if jump_total > 0 else 0.0
        print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {epoch_loss:.4f} | Overall Acc: {epoch_acc:.1f}% | Jump/Trick Acc: {jump_acc:.1f}%")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ppo_model.save(args.output)
    print(f"\n🎉 Success! Warm-started policy saved to: {args.output}")
    print(f"👉 Recommended next step to fine-tune without catastrophic forgetting:")
    print(f"   python train.py --game {args.game} --resume-from {args.output} --lr 3e-5 --ent-coef 0.001 --death-penalty 50.0 --iterations 300")
    venv.close()

if __name__ == "__main__":
    main()
