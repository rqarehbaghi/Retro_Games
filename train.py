#!/usr/bin/env python3
"""
Train a PPO agent from scratch or fine-tune from a human imitation checkpoint.
Features rolling latest_iter_N.zip autosaves, custom reward shaping (death penalties),
and save-state curriculum learning.
"""
import argparse
import os
import re
import cv2
import numpy as np
import stable_retro as retro
from gymnasium import ActionWrapper, ObservationWrapper, Wrapper
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

DEFAULT_COMBOS = [
    [],
    ["RIGHT"],
    ["LEFT"],
    ["RIGHT", "B"],
    ["RIGHT", "A", "B"],
    ["LEFT", "A", "B"],
    ["A"],
    ["B"],
    ["DOWN"],
    ["UP"],
]

class Discretizer(ActionWrapper):
    """
    Compresses multi-binary button spaces down to discrete action combos.
    """
    def __init__(self, env, combos):
        super().__init__(env)
        buttons = env.unwrapped.buttons
        self._actions = []
        for combo in combos:
            arr = np.array([False] * len(buttons))
            for b in combo:
                if b in buttons:
                    arr[buttons.index(b)] = True
            self._actions.append(arr)
        self.action_space = Discrete(len(self._actions))

    def action(self, act):
        return self._actions[act].copy()

class WarpFrame(ObservationWrapper):
    """
    Downsamples frame to 84x84 grayscale.
    """
    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = Box(low=0, high=255, shape=(size, size, 1), dtype=np.uint8)

    def observation(self, obs):
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class FrameSkip(Wrapper):
    """
    Repeats action for `skip` frames and aggregates reward.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, r, term, trunc, info = self.env.step(action)
            total_reward += r
            if term or trunc:
                break
        return obs, total_reward, term, trunc, info

class JumpIncentiveWrapper(Wrapper):
    """
    Solves 'no jump attempts in 100 iterations':
    1. Small reward whenever a running-jump is attempted.
    2. Penalizes being stuck / blocked by an obstacle without jumping.
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

        # Jump attempt incentive
        if jump_idx is not None and isinstance(action, (list, np.ndarray)) and len(action) > jump_idx:
            if action[jump_idx]:
                reward += self.jump_bonus

        # Penalize standing still against a wall / pipe
        if current_x is not None and self.prev_x is not None:
            if abs(current_x - self.prev_x) < 0.5:
                self.stalled_frames += 1
                if self.stalled_frames > 8:
                    reward -= self.stuck_penalty
            else:
                self.stalled_frames = 0
            self.prev_x = current_x

        # Vertical height bonus
        if current_y is not None and self.prev_y is not None:
            if current_y > self.prev_y:
                reward += self.jump_bonus * 0.5
            self.prev_y = current_y

        return obs, reward, term, trunc, info

class CustomRewardShaper(Wrapper):
    """
    Fixes the 'Running into obstacles and dying' problem:
    1. Penalizes death (-25 to -100) so blind suicide is strongly discouraged.
    2. Adds reward for staying alive and moving right.
    3. Rewards clearing obstacles.
    """
    def __init__(self, env, death_penalty=50.0, survival_reward=0.01):
        super().__init__(env)
        self.death_penalty = death_penalty
        self.survival_reward = survival_reward
        self.prev_x = 0
        self.prev_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = info.get("x", info.get("x_pos", 0))
        self.prev_lives = info.get("lives", info.get("health", None))
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        # 1. Survival tick
        reward += self.survival_reward

        # 2. Horizontal progress bonus
        current_x = info.get("x", info.get("x_pos", None))
        if current_x is not None:
            x_delta = current_x - self.prev_x
            if x_delta > 0:
                reward += float(x_delta) * 0.1
            self.prev_x = current_x

        # 3. Death / Damage Penalty
        current_lives = info.get("lives", info.get("health", None))
        if self.prev_lives is not None and current_lives is not None:
            if current_lives < self.prev_lives:
                reward -= self.death_penalty
            self.prev_lives = current_lives
        elif term and not info.get("is_stage_clear", False):
            # Terminal state that wasn't a level win
            reward -= self.death_penalty

        return obs, reward, term, trunc, info

class SaveCheckpointCallback(BaseCallback):
    """
    Saves rolling checkpoints at each iteration with correct cumulative iteration numbering.
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
                print(f"[Checkpoint] Iteration {total_iter} (Session iter {current_run_iter}) saved to {save_file}")
        return True

def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)

def make_env_fn(game, state, combos, death_penalty, jump_bonus, render=False):
    def _init():
        render_mode = "human" if render else "rgb_array"
        e = retro.make(game=game, state=state or retro.State.DEFAULT, render_mode=render_mode)
        e = Discretizer(e, combos)
        e = CustomRewardShaper(e, death_penalty=death_penalty)
        if jump_bonus > 0:
            e = JumpIncentiveWrapper(e, jump_bonus=jump_bonus)
        e = WarpFrame(e)
        e = FrameSkip(e, skip=4)
        return e
    return _init

def main():
    parser = argparse.ArgumentParser(description="Train or fine-tune PPO agent on retro games.")
    parser.add_argument("--game", type=str, default="SuperMarioBros3-Nes-v0", help="Retro game identifier")
    parser.add_argument("--state", type=str, default=None, help="Save state (e.g. Level1-1)")
    parser.add_argument("--iterations", type=int, default=500, help="Number of PPO training iterations")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel CPU environments")
    parser.add_argument("--n-steps", type=int, default=256, help="Rollout steps per env")
    parser.add_argument("--render", action="store_true", help="Open visual emulator windows for envs (slows down training)")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to existing .zip checkpoint to continue from")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default 2.5e-4 for scratch, 3e-5 for fine-tuning imitation checkpoints)")
    parser.add_argument("--death-penalty", type=float, default=50.0, help="Penalty applied when agent dies or loses health")
    parser.add_argument("--jump-bonus", type=float, default=0.2, help="Bonus for exploring jump actions (forces jumps in <100 iterations)")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient (default 0.05 for scratch, 0.001 for fine-tuning)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory for weights")
    args = parser.parse_args()

    # Determine optimal LR and entropy coefficient based on mode
    is_resuming = bool(args.resume_from and os.path.exists(args.resume_from))
    effective_lr = args.lr if args.lr is not None else (3e-5 if is_resuming else 2.5e-4)
    effective_ent = args.ent_coef if args.ent_coef is not None else (0.001 if is_resuming else 0.05)

    save_dir = os.path.join(args.output_dir, safe_name(args.game))
    os.makedirs(save_dir, exist_ok=True)

    print(f"🎮 Initializing {args.num_envs} parallel environments for [{args.game}]...")
    env_fns = [make_env_fn(args.game, args.state, DEFAULT_COMBOS, args.death_penalty, args.jump_bonus, args.render) for _ in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    total_timesteps_per_iter = args.num_envs * args.n_steps
    total_timesteps = args.iterations * total_timesteps_per_iter

    start_iter = 0
    if is_resuming:
        print(f"🚀 Loading weights from checkpoint: {args.resume_from}")
        print(f"🎯 Fine-Tuning Mode: Learning Rate = {effective_lr}, Entropy = {effective_ent} (Prevents catastrophic forgetting)")
        # Extract previous iteration number if filename matches latest_iter_X.zip
        match = re.search(r"latest_iter_(\d+)", os.path.basename(args.resume_from))
        if match:
            start_iter = int(match.group(1))
            print(f"📈 Resuming from previous Iteration {start_iter} (Checkpoints will continue at {start_iter + 1}, {start_iter + 2}...)")
        
        model = PPO.load(
            args.resume_from,
            env=vec_env,
            learning_rate=effective_lr,
            ent_coef=effective_ent,
            n_steps=args.n_steps,
        )
    else:
        print(f"🧠 Building fresh PPO Agent with CnnPolicy (LR={effective_lr}, Ent={effective_ent})...")
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            learning_rate=effective_lr,
            n_steps=args.n_steps,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=effective_ent,
        )

    callback = SaveCheckpointCallback(check_freq=args.n_steps, save_path=save_dir, start_iter=start_iter)

    print(f"🔥 Starting PPO Training for {args.iterations} additional iterations (Iterations {start_iter + 1} -> {start_iter + args.iterations})...")
    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)

    final_path = os.path.join(save_dir, "final_model.zip")
    model.save(final_path)
    print(f"✅ Training completed! Model saved to {final_path}")

    vec_env.close()

if __name__ == "__main__":
    main()
