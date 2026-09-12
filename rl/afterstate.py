"""
Generic Deep Afterstate Value Network (Lookahead RL).

Completely game-agnostic:
  - Sutton & Barto Afterstate Reinforcement Learning (Chapter 6).
  - Evaluates afterstates produced by any game simulator adhering to BaseAfterstateSimulator.
  - Zero game-specific assumptions, shapes, or heuristics.
"""
import os
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class AfterstateValueNet(nn.Module):
        """
        Deep Neural Network evaluating generic state/afterstate representations.
        Input: input_dim-dimensional feature vector.
        Output: Scalar state-value estimate V(S').
        """

        def __init__(self, input_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.input_dim = input_dim
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)


class AfterstateReplayBuffer:
    """Experience replay buffer for off-policy afterstate TD learning."""

    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, float, Optional[np.ndarray], bool]] = []
        self.pos = 0

    def push(self, state: np.ndarray, reward: float, next_state: Optional[np.ndarray], done: bool):
        item = (state, float(reward), next_state, bool(done))
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, rewards, next_states, dones = zip(*batch)

        s_arr = np.array(states, dtype=np.float32)
        r_arr = np.array(rewards, dtype=np.float32)
        d_arr = np.array(dones, dtype=np.bool_)

        ns_clean = [ns if ns is not None else np.zeros_like(states[0]) for ns in next_states]
        ns_arr = np.array(ns_clean, dtype=np.float32)

        return s_arr, r_arr, ns_arr, d_arr

    def __len__(self):
        return len(self.buffer)


class AfterstateAgent:
    """
    Generic Lookahead Agent powered by Deep Afterstate Value Network.
    Evaluates candidate action transitions produced by any game simulator.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        device: str = "cpu",
        lr: float = 2.5e-4,
        gamma: float = 0.99
    ):
        self.input_dim = input_dim
        self.device = torch.device(device if (HAS_TORCH and torch.cuda.is_available() and device == "cuda") else "cpu")
        self.gamma = gamma

        if HAS_TORCH:
            self.val_net = AfterstateValueNet(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
            self.target_net = AfterstateValueNet(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
            self.target_net.load_state_dict(self.val_net.state_dict())
            self.optimizer = torch.optim.Adam(self.val_net.parameters(), lr=lr)
        else:
            self.val_net = None
            self.target_net = None
            self.optimizer = None

    def select_action(
        self,
        candidates: List[Dict[str, Any]],
        epsilon: float = 0.0
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Generic decision selection over candidate transitions.

        Args:
            candidates: List of dicts, each with:
                "action": int
                "afterstate": 1D np.ndarray feature vector
                "immediate_reward": float (optional, default 0.0)
            epsilon: Exploration probability for epsilon-greedy selection.

        Returns:
            best_action: int
            best_afterstate: np.ndarray (or None if no candidates)
        """
        if not candidates:
            return 0, None

        if epsilon > 0.0 and random.random() < epsilon:
            chosen = random.choice(candidates)
            return chosen["action"], chosen.get("afterstate")

        afterstates = [c["afterstate"] for c in candidates]

        if HAS_TORCH and self.val_net is not None:
            with torch.no_grad():
                tensor_states = torch.tensor(np.array(afterstates), dtype=torch.float32, device=self.device)
                future_values = self.val_net(tensor_states).cpu().numpy()
        else:
            future_values = np.zeros(len(candidates), dtype=np.float32)

        scores = []
        for i, c in enumerate(candidates):
            imm_r = float(c.get("immediate_reward", 0.0))
            scores.append(imm_r + self.gamma * future_values[i])

        best_idx = int(np.argmax(scores))
        best_cand = candidates[best_idx]
        return best_cand["action"], best_cand.get("afterstate")

    def update(self, replay_buffer: AfterstateReplayBuffer, batch_size: int = 64) -> float:
        """Perform one step of TD value-network optimization."""
        if not HAS_TORCH or len(replay_buffer) < batch_size:
            return 0.0

        states, rewards, next_states, dones = replay_buffer.sample(batch_size)

        s_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        ns_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        d_t = torch.tensor(dones, dtype=torch.bool, device=self.device)

        pred_vals = self.val_net(s_t)

        with torch.no_grad():
            next_vals = self.target_net(ns_t)
            next_vals[d_t] = 0.0
            targets = r_t + self.gamma * next_vals

        loss = F.smooth_l1_loss(pred_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.val_net.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())

    def sync_target_network(self):
        """Update target network weights."""
        if HAS_TORCH and self.val_net is not None and self.target_net is not None:
            self.target_net.load_state_dict(self.val_net.state_dict())

    def save(self, filepath: str):
        """Save model checkpoint as a .zip archive (matching standard studio/train checkpoint format)."""
        if not filepath.endswith(".zip"):
            filepath = filepath + ".zip"
        if HAS_TORCH and self.val_net is not None:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            import io
            import json
            import zipfile

            buf = io.BytesIO()
            torch.save({
                "input_dim": self.input_dim,
                "model_state_dict": self.val_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            }, buf)
            meta = json.dumps({"model_type": "afterstate", "input_dim": self.input_dim}, indent=2)

            with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("value_net.pth", buf.getvalue())
                zf.writestr("meta.json", meta)

    def load(self, filepath: str):
        """Load model checkpoint from .zip (or raw .pt)."""
        resolved = filepath
        if not os.path.exists(resolved):
            if os.path.exists(filepath + ".zip"):
                resolved = filepath + ".zip"
            elif os.path.exists(filepath + ".pt"):
                resolved = filepath + ".pt"

        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

        if HAS_TORCH and self.val_net is not None:
            import io
            import zipfile

            if zipfile.is_zipfile(resolved):
                with zipfile.ZipFile(resolved, "r") as zf:
                    if "value_net.pth" in zf.namelist():
                        raw_bytes = zf.read("value_net.pth")
                    elif "value_net.pt" in zf.namelist():
                        raw_bytes = zf.read("value_net.pt")
                    else:
                        raise ValueError(f"{resolved} is not a valid afterstate checkpoint")
                    data = torch.load(io.BytesIO(raw_bytes), map_location=self.device)
            else:
                data = torch.load(resolved, map_location=self.device)

            self.val_net.load_state_dict(data["model_state_dict"])
            self.target_net.load_state_dict(self.val_net.state_dict())
            if self.optimizer and "optimizer_state_dict" in data and data["optimizer_state_dict"]:
                self.optimizer.load_state_dict(data["optimizer_state_dict"])


def train_afterstate(args, spec, overrides):
    """
    Standard training runner for Afterstate Lookahead RL.
    Called directly by train.py when games.json specifies 'algorithm': 'afterstate'.
    """
    import sys
    from rl.simulators import get_simulator
    from rl.env import make_env
    from rl.vars import GameVars

    sim_name = spec.afterstate_config.get("simulator") or spec.features_name or spec.game
    simulator = get_simulator(sim_name)
    if simulator is None:
        sys.exit(
            f"Error: Game '{args.game}' does not have an afterstate simulator registered.\n"
            f"Expected simulator '{sim_name}' in rl/simulators/."
        )

    save_dir = args.save_dir or os.path.join("checkpoints", args.game)
    os.makedirs(save_dir, exist_ok=True)

    lr = args.lr if args.lr is not None else 2.5e-4
    gamma = args.gamma if args.gamma is not None else 0.99
    batch_size = args.batch_size if args.batch_size is not None else 64
    device = args.device or ("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
    eps_start = args.ent_coef if args.ent_coef is not None else 0.20
    eps_final = args.ent_coef_final if args.ent_coef_final is not None else 0.01

    print(f"algorithm   : Afterstate Lookahead Value Network")
    print(f"simulator   : {sim_name} (feature dim: {simulator.feature_dim})")
    print(f"timesteps   : {args.timesteps}")
    print(f"device      : {device}")
    print(f"batch / lr  : {batch_size} / {lr} (gamma {gamma})")
    print(f"exploration : epsilon {eps_start} -> {eps_final}")

    env = make_env(args.game, overrides=overrides)
    vars = GameVars(spec.game, entry=spec.entry)

    agent = AfterstateAgent(
        input_dim=simulator.feature_dim,
        device=device,
        lr=lr,
        gamma=gamma
    )
    replay = AfterstateReplayBuffer(capacity=50000)

    if args.resume:
        print(f"resuming weights from {args.resume}")
        agent.load(args.resume)

    total_steps = 0
    ep = 0
    recent_rewards = []
    recent_steps = []
    next_save_step = args.save_every

    print("\nBeginning training...")
    header = f"{'Steps':>8} | {'Ep':>5} | {'Epsilon':>7} | {'Reward':>8} | {'AvgRew':>8} | {'Loss':>7} | Game Stats"
    print(header)
    print("-" * 75)

    while total_steps < args.timesteps:
        ep += 1
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        prev_afterstate_feat = None
        ep_loss = 0.0
        loss_updates = 0
        last_info = info

        eps_progress = min(1.0, total_steps / float(max(1, args.timesteps)))
        epsilon = eps_start + (eps_final - eps_start) * eps_progress

        while not done and total_steps < args.timesteps:
            ram = getattr(env.unwrapped, "ram", None) if hasattr(env, "unwrapped") else None
            candidates = simulator.get_candidates(obs=obs, ram=ram, info=info, vars=vars, spec=spec)

            action, afterstate_feat = agent.select_action(candidates, epsilon=epsilon)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_info = info

            ep_reward += float(reward)
            ep_steps += 1
            total_steps += 1

            if prev_afterstate_feat is not None:
                replay.push(prev_afterstate_feat, reward, afterstate_feat, done)

            prev_afterstate_feat = afterstate_feat

            if len(replay) >= batch_size:
                loss = agent.update(replay, batch_size=batch_size)
                ep_loss += loss
                loss_updates += 1

            if total_steps % 500 == 0:
                agent.sync_target_network()

            if total_steps >= next_save_step:
                ckpt_path = os.path.join(save_dir, f"ckpt_{next_save_step}_steps.zip")
                agent.save(ckpt_path)
                print(f"  --> Saved checkpoint: {ckpt_path} (step {total_steps})")
                next_save_step += args.save_every

            obs = next_obs

        # Terminal transition penalty
        if prev_afterstate_feat is not None:
            replay.push(prev_afterstate_feat, -20.0, None, True)

        recent_rewards.append(ep_reward)
        recent_steps.append(ep_steps)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
            recent_steps.pop(0)

        avg_rew = np.mean(recent_rewards)
        avg_loss = (ep_loss / max(1, loss_updates)) if loss_updates > 0 else 0.0

        stat_parts = []
        for k in [f"lines_p{spec.player}", "lines", f"score_p{spec.player}", "score", "holes", "height"]:
            if k in last_info:
                stat_parts.append(f"{k}={last_info[k]}")
        stats_str = " ".join(stat_parts) if stat_parts else ""

        print(f"{total_steps:8d} | {ep:5d} | {epsilon:7.3f} | {ep_reward:8.1f} | {avg_rew:8.1f} | {avg_loss:7.4f} | {stats_str}")

    final_path = os.path.join(save_dir, "final.zip")
    agent.save(final_path)
    print(f"\nTraining complete. Final weights saved to {final_path}")
    env.close()

