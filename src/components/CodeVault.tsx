import React, { useState } from 'react';
import { Terminal, Copy, Check, FileCode2, Download, ExternalLink, Sparkles, Layers, GraduationCap } from 'lucide-react';

export const CodeVault: React.FC = () => {
  const [activeFile, setActiveFile] = useState<'pretrain_imitation' | 'play_human_vs_ai' | 'train' | 'play_and_record' | 'make_progress_reel' | 'custom_wrappers'>('pretrain_imitation');
  const [copied, setCopied] = useState<boolean>(false);

  const scripts = {
    pretrain_imitation: {
      name: 'pretrain_imitation.py (Teacher Mode)',
      description: 'Supervised Behavioral Cloning: trains the PPO CnnPolicy on your human gameplay .bk2 recordings with FrameSkip-4 alignment & Jump Class Weighting (fixes 85% run vs 5% jump imbalance).',
      code: `#!/usr/bin/env python3
"""
Behavioral Cloning & Imitation Learning (Teacher Mode):
Pre-train a PPO policy network on human gameplay demonstrations!

Includes Class-Loss Balancing (8x weight on Jump Actions) so the model
actively learns jump triggers and doesn't get tricked into just running right.
"""
import argparse, glob, os
from collections import deque
import cv2, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from train import DEFAULT_COMBOS, Discretizer, FrameSkip, WarpFrame

class RetroDemoDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stacked_obs, action_idx = self.samples[idx]
        return torch.tensor(stacked_obs, dtype=torch.float32), torch.tensor(action_idx, dtype=torch.long)

def map_binary_to_combo(button_array, env_buttons, combos=DEFAULT_COMBOS):
    active_buttons = {env_buttons[i] for i, pressed in enumerate(button_array) if pressed}
    best_idx, best_match_score = 0, -1
    for idx, combo in enumerate(combos):
        combo_set = set(combo)
        if combo_set == active_buttons:
            return idx
        intersection = len(combo_set & active_buttons)
        if intersection > best_match_score:
            best_match_score = intersection
            best_idx = idx
    return best_idx

def extract_demos_from_bk2(game, bk2_paths, combos=DEFAULT_COMBOS, frame_skip=4):
    print(f"📦 Extracting frames from {len(bk2_paths)} .bk2 replays (FrameSkip={frame_skip} aligned)...")
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
                
                # Align with training FrameSkip
                if frame_counter % frame_skip != 0:
                    continue

                gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

                if len(frame_queue) < 4:
                    while len(frame_queue) < 4:
                        frame_queue.append(resized)
                else:
                    frame_queue.append(resized)

                combo_idx = map_binary_to_combo(keys, buttons, combos)
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
    parser.add_argument("--game", type=str, default="SuperMarioBros3-Nes-v0")
    parser.add_argument("--demo-dir", type=str, default="./recordings")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--jump-weight", type=float, default=8.0, help="Class loss multiplier for Jump/Trick actions")
    parser.add_argument("--output", type=str, default="./checkpoints/pretrained_human_bc.zip")
    args = parser.parse_args()

    bk2_files = glob.glob(os.path.join(args.demo_dir, "*.bk2"))
    if not bk2_files:
        print(f"❌ No .bk2 replay files found in '{args.demo_dir}'!")
        return

    samples = extract_demos_from_bk2(args.game, bk2_files, DEFAULT_COMBOS, frame_skip=4)
    
    # Class frequency balancing
    jump_indices = {i for i, c in enumerate(DEFAULT_COMBOS) if "A" in c}
    num_classes = len(DEFAULT_COMBOS)
    weights = np.ones(num_classes, dtype=np.float32)
    for idx in range(num_classes):
        if idx in jump_indices:
            weights[idx] = args.jump_weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    dataset = RetroDemoDataset(samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    def make_dummy():
        e = retro.make(game=args.game)
        e = Discretizer(e, DEFAULT_COMBOS)
        e = WarpFrame(e)
        e = FrameSkip(e, skip=4)
        return e

    venv = DummyVecEnv([make_dummy])
    venv = VecFrameStack(venv, n_stack=4)

    print("🧠 Initializing PPO Actor-Critic Network...")
    ppo_model = PPO("CnnPolicy", venv, verbose=0)
    policy_net = ppo_model.policy

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    policy_net.to(device)
    policy_net.train()

    print(f"🚀 Training Behavioral Cloning on {len(samples):,} frames for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        total_loss, correct, jump_correct, jump_total, total = 0.0, 0, 0, 0, 0
        for obs_batch, act_batch in dataloader:
            obs_batch, act_batch = obs_batch.to(device), act_batch.to(device)
            optimizer.zero_grad()
            distribution = policy_net.get_distribution(obs_batch)
            logits = distribution.distribution.logits
            loss = criterion(logits, act_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * obs_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == act_batch).sum().item()
            total += obs_batch.size(0)

            for p, a in zip(preds, act_batch):
                if a.item() in jump_indices:
                    jump_total += 1
                    if p.item() == a.item():
                        jump_correct += 1

        jump_acc = (jump_correct / max(1, jump_total)) * 100.0 if jump_total > 0 else 0.0
        print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {total_loss/total:.4f} | Overall: {(correct/total)*100:.1f}% | Jump Acc: {jump_acc:.1f}%")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ppo_model.save(args.output)
    print(f"🎉 Warm-started policy saved to: {args.output}")
    print(f"👉 Fine-tune with gentle LR: python train.py --game {args.game} --resume-from {args.output} --lr 3e-5 --ent-coef 0.001")
    venv.close()

if __name__ == "__main__":
    main()
`
    },
    train: {
      name: 'train.py (with Cumulative Resume & Jump Shaper)',
      description: 'PPO Training Engine with parallel subproc environments, JumpIncentiveWrapper, Death Penalty, cumulative resume numbering, and rolling autosaves.',
      code: `#!/usr/bin/env python3
"""
Train a PPO agent from scratch or fine-tune from a checkpoint with cumulative iteration numbering.
"""
import argparse, os, re, cv2, numpy as np
import stable_retro as retro
from gymnasium import ActionWrapper, ObservationWrapper, Wrapper
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

DEFAULT_COMBOS = [
    [], ["RIGHT"], ["LEFT"], ["RIGHT", "B"], ["RIGHT", "A", "B"],
    ["LEFT", "A", "B"], ["A"], ["B"], ["DOWN"], ["UP"],
]

class SaveCheckpointCallback(BaseCallback):
    """
    Saves rolling checkpoints at each iteration with cumulative numbering.
    """
    def __init__(self, check_freq, save_path, start_iter=0, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.start_iter = start_iter
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            current_run_iter = self.n_calls // self.check_freq
            total_iter = self.start_iter + current_run_iter
            save_file = os.path.join(self.save_path, f"latest_iter_{total_iter}.zip")
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"[Checkpoint] Iteration {total_iter} saved to {save_file}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("--game", type=str, default="SuperMarioBros3-Nes-v0")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--death-penalty", type=float, default=50.0)
    parser.add_argument("--jump-bonus", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    # Automatically extract previous iteration count if resuming from latest_iter_X.zip
    start_iter = 0
    if args.resume_from and os.path.exists(args.resume_from):
        match = re.search(r"latest_iter_(\d+)", os.path.basename(args.resume_from))
        if match:
            start_iter = int(match.group(1))
            print(f"📈 Resuming from Iteration {start_iter}. Next save will be latest_iter_{start_iter + 1}.zip")

    # Pass reset_num_timesteps=False to keep cumulative step counters
    callback = SaveCheckpointCallback(check_freq=args.n_steps, save_path=args.output_dir, start_iter=start_iter)
    # model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)
`
    },
    play_human_vs_ai: {
      name: 'play_human_vs_ai.py',
      description: '2-Player Human vs. AI live fighting & co-op gameplay script with Pygame keyboard/gamepad capture, frame wrappers, and automatic HD MP4 video rendering.',
      code: `#!/usr/bin/env python3
"""
Play 2-Player Retro Games: Human vs. AI (or Human + AI Co-Op) with
automatic high-definition MP4 recording, live score/health overlay,
and telemetry.
"""
import argparse, glob, os, subprocess, sys, time
from collections import deque
import cv2, numpy as np, pygame
import stable_retro as retro
from stable_baselines3 import PPO

KEY_MAPPING = {
    pygame.K_UP: "UP", pygame.K_w: "UP",
    pygame.K_DOWN: "DOWN", pygame.K_s: "DOWN",
    pygame.K_LEFT: "LEFT", pygame.K_a: "LEFT",
    pygame.K_RIGHT: "RIGHT", pygame.K_d: "RIGHT",
    pygame.K_z: "B", pygame.K_j: "B",
    pygame.K_x: "A", pygame.K_k: "A",
    pygame.K_c: "C", pygame.K_l: "C",
    pygame.K_q: "X", pygame.K_u: "X",
    pygame.K_e: "Y", pygame.K_i: "Y",
    pygame.K_r: "Z", pygame.K_o: "Z",
    pygame.K_RETURN: "START", pygame.K_RSHIFT: "SELECT",
}

FIGHTER_COMBOS = [
    [], ["RIGHT"], ["LEFT"], ["UP"], ["DOWN"],
    ["A"], ["B"], ["C"], ["X"], ["Y"], ["Z"],
    ["DOWN", "A"], ["DOWN", "B"], ["RIGHT", "A"], ["RIGHT", "B"],
]

def make_p1_action(env_buttons, pressed_keys):
    action = np.array([False] * len(env_buttons), dtype=bool)
    for key, button_name in KEY_MAPPING.items():
        if pressed_keys[key] and button_name in env_buttons:
            action[env_buttons.index(button_name)] = True
    return action
`
    },
    play_and_record: {
      name: 'play_and_record.py',
      description: 'Single-player replay and video renderer with fixed (4,84,84) observation wrappers for Stable-Baselines3 checkpoints and human play capture.',
      code: `#!/usr/bin/env python3
"""
Play an NES game (AI or Human) and automatically export a finished MP4.
Fixed: Observation shape wrappers automatically applied when --model is loaded.
"""
import argparse, glob, os, subprocess, sys, time
import stable_retro as retro
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from train import DEFAULT_COMBOS, Discretizer, FrameSkip, WarpFrame

def play_agent_episode(game, state, model_path, max_steps, record_dir, render):
    render_mode = "human" if render else "rgb_array"
    
    if model_path is None:
        env = retro.make(game=game, state=state or retro.State.DEFAULT, record=record_dir, render_mode=render_mode)
        obs, info = env.reset()
        steps, total_reward = 0, 0.0
        while True:
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            total_reward += reward
            steps += 1
            if term or trunc or steps >= max_steps:
                break
        env.close()
    else:
        from stable_baselines3 import PPO
        def make_env():
            e = retro.make(game=game, state=state or retro.State.DEFAULT, record=record_dir, render_mode=render_mode)
            e = Discretizer(e, DEFAULT_COMBOS)
            e = WarpFrame(e)
            e = FrameSkip(e, skip=4)
            return e

        venv = DummyVecEnv([make_env])
        venv = VecFrameStack(venv, n_stack=4)
        model = PPO.load(model_path)
        obs = venv.reset()
        steps, total_reward = 0, 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            total_reward += reward[0]
            steps += 1
            if dones[0] or steps >= (max_steps // 4):
                break
        venv.close()
`
    },
    make_progress_reel: {
      name: 'make_progress_reel.py',
      description: 'Automated video stitcher exporting both 1920x1080 Landscape and 1080x1920 Portrait video reels stamped with iteration labels.',
      code: `#!/usr/bin/env python3
"""
Turn training checkpoints into YouTube & TikTok progress reels.
Produces progress_youtube.mp4 and progress_shorts.mp4 automatically.
"""
import argparse, glob, os, shutil, subprocess, sys, tempfile

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def build_segment(src_mp4, label_text, width, height, clip_seconds, out_path):
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"drawtext=fontfile={FONT}:text='{label_text}':fontcolor=white:"
        f"fontsize={max(24, width // 20)}:x=(w-text_w)/2:y=40:"
        f"box=1:boxcolor=black@0.5:boxborderw=10"
    )
    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-i", src_mp4,
        "-vf", vf, "-r", "60", "-t", str(clip_seconds),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", out_path
    ], check=True)
`
    },
    custom_wrappers: {
      name: 'custom_wrappers.py (Jump Incentive & Death Penalty)',
      description: 'JumpIncentiveWrapper (forces jump exploration in <100 iterations), DeathPenaltyWrapper, and Obstacle Progress Shaper for stable-retro.',
      code: `#!/usr/bin/env python3
"""
Custom Gym Wrappers for Stable-Retro Reinforcement Learning:
- JumpIncentiveWrapper (Forces jump exploration within the first 100 iterations)
- DeathPenaltyWrapper (Stops suicidal rushing into obstacles)
- Distance & Velocity Progress Shaper
"""
from gymnasium import Wrapper
import numpy as np

class JumpIncentiveWrapper(Wrapper):
    """
    Forces the agent to attempt jumps within 100 iterations:
    1. Small exploration bonus when forward jump is initiated ('RIGHT' + 'A').
    2. Penalizes being stuck against an obstacle without jumping.
    3. Rewards gaining vertical airtime while progressing horizontally.
    """
    def __init__(self, env, jump_bonus=0.2, stuck_penalty=0.05, jump_button="A"):
        super().__init__(env)
        self.jump_bonus = jump_bonus
        self.stuck_penalty = stuck_penalty
        self.jump_button = jump_button
        self.prev_x = 0
        self.stalled_frames = 0
        self.prev_y = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = info.get("x", info.get("x_pos", 0))
        self.prev_y = info.get("y", info.get("y_pos", 0))
        self.stalled_frames = 0
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        current_x = info.get("x", info.get("x_pos", None))
        current_y = info.get("y", info.get("y_pos", None))

        buttons = getattr(self.env.unwrapped, "buttons", [])
        jump_idx = buttons.index(self.jump_button) if self.jump_button in buttons else None

        # 1. Action exploration incentive for jumping
        if jump_idx is not None and isinstance(action, (list, np.ndarray)) and len(action) > jump_idx:
            if action[jump_idx]:
                reward += self.jump_bonus

        # 2. Discourage bumping into walls without jumping
        if current_x is not None and self.prev_x is not None:
            if abs(current_x - self.prev_x) < 0.5:
                self.stalled_frames += 1
                if self.stalled_frames > 8:
                    reward -= self.stuck_penalty
            else:
                self.stalled_frames = 0
            self.prev_x = current_x

        # 3. Vertical airtime bonus
        if current_y is not None and self.prev_y is not None:
            if current_y > self.prev_y:
                reward += self.jump_bonus * 0.5
            self.prev_y = current_y

        return obs, reward, term, trunc, info

class DeathPenaltyWrapper(Wrapper):
    def __init__(self, env, penalty=50.0, survival_tick=0.01):
        super().__init__(env)
        self.penalty = penalty
        self.survival_tick = survival_tick
        self.prev_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_lives = info.get("lives", None)
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        reward += self.survival_tick

        current_lives = info.get("lives", None)
        if self.prev_lives is not None and current_lives is not None:
            if current_lives < self.prev_lives:
                reward -= self.penalty
            self.prev_lives = current_lives
        elif term and not info.get("is_stage_clear", False):
            reward -= self.penalty

        return obs, reward, term, trunc, info
`
    }
  };

  const current = scripts[activeFile];

  const copyCode = () => {
    navigator.clipboard.writeText(current.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-sm">
        <div className="flex items-center space-x-2">
          <Terminal className="w-5 h-5 text-emerald-400" />
          <h2 className="text-lg font-bold text-white">Production Python Scripts & Pipeline Vault</h2>
        </div>
        <p className="text-xs text-slate-400 mt-1">
          Complete, verified, copyable source scripts ready to execute on your local Ubuntu / WSL2 / Cloud GPU machine.
        </p>
      </div>

      {/* Script File Tabs */}
      <div className="flex flex-wrap gap-2">
        {(Object.keys(scripts) as (keyof typeof scripts)[]).map(key => {
          const s = scripts[key];
          const isActive = activeFile === key;
          return (
            <button
              key={key}
              onClick={() => setActiveFile(key)}
              className={`flex items-center space-x-2 px-4 py-2.5 rounded-xl text-xs font-semibold transition-all border ${
                isActive
                  ? 'bg-amber-500/20 text-white border-amber-500 shadow-md shadow-amber-500/10'
                  : 'bg-slate-900 text-slate-400 border-slate-800 hover:border-slate-700 hover:text-slate-200'
              }`}
            >
              <FileCode2 className={`w-4 h-4 ${isActive ? 'text-amber-400' : 'text-slate-500'}`} />
              <span>{s.name}</span>
            </button>
          );
        })}
      </div>

      {/* Code Viewer Panel */}
      <div className="bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
        <div className="bg-slate-900/90 px-4 py-3 border-b border-slate-800 flex items-center justify-between">
          <div>
            <div className="text-xs font-mono font-bold text-amber-400">{current.name}</div>
            <div className="text-[11px] text-slate-400">{current.description}</div>
          </div>

          <button
            onClick={copyCode}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition-all"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
            <span>{copied ? 'Copied to Clipboard!' : 'Copy Script'}</span>
          </button>
        </div>

        <pre className="p-4 text-xs font-mono text-slate-300 overflow-x-auto leading-relaxed bg-slate-950 max-h-[500px]">
          <code>{current.code}</code>
        </pre>
      </div>

      {/* Quick Setup Cheatsheet */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-1.5">
          <Sparkles className="w-4 h-4 text-amber-400" />
          Teacher Mode Quick-Start Workflow
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono">
            <span className="text-slate-500 block text-[10px]"># 1. Record 5 mins of human play</span>
            <span className="text-amber-400 text-[11px]">python play_and_record.py --game SuperMarioBros3-Nes-v0 --human --record-dir ./human_demos</span>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono">
            <span className="text-slate-500 block text-[10px]"># 2. Supervised Pre-Training (2 mins)</span>
            <span className="text-indigo-400 text-[11px]">python pretrain_imitation.py --game SuperMarioBros3-Nes-v0 --demo-dir ./human_demos --epochs 20</span>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono">
            <span className="text-slate-500 block text-[10px]"># 3. Fine-tune with PPO + Death Penalty</span>
            <span className="text-emerald-400 text-[11px]">python train.py --game SuperMarioBros3-Nes-v0 --resume-from ./checkpoints/pretrained_human_bc.zip --death-penalty 50.0</span>
          </div>
        </div>
      </div>
    </div>
  );
};
