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

        def __init__(self, input_dim: int, hidden_dim: int = 256,
                     model_type: str = "mlp"):
            super().__init__()
            self.input_dim = input_dim
            self.model_type = model_type
            if model_type == "linear":
                self.net = nn.Linear(input_dim, 1)
            elif model_type == "mlp":
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
            else:
                raise ValueError("Unknown afterstate model_type: " + str(model_type))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)


class AfterstateReplayBuffer:
    """Experience replay for sampled afterstate TD policy evaluation."""

    def __init__(self, capacity: int = 50000):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, float, Optional[np.ndarray], bool]] = []
        self.pos = 0

    def push(self, state: np.ndarray, reward: float, next_state: Optional[np.ndarray], done: bool):
        if next_state is None and not done:
            raise ValueError("A nonterminal transition requires an observed successor")
        item = (np.array(state, copy=True), float(reward),
                None if next_state is None else np.array(next_state, copy=True), bool(done))
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


class AfterstateTrajectory:
    """V(afterstate) excludes the reward already earned creating that state.

    Only real settled states enter replay. A death belongs to the last observed
    afterstate; neither collector exhaustion nor truncation is a death.
    """
    def __init__(self, replay):
        self.replay = replay
        self.previous = None

    def record(self, reward, observed=None, terminated=False):
        recorded = self.previous is not None and (terminated or observed is not None)
        if terminated:
            if self.previous is not None:
                self.replay.push(self.previous, reward, None, True)
            self.previous = None
        elif observed is not None:
            if self.previous is not None:
                self.replay.push(self.previous, reward, observed, False)
            self.previous = np.array(observed, copy=True)
        return recorded


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
        gamma: float = 0.99,
        model_type: str = "mlp",
        value_reward_scale: float = 1.0,
        target_tau: float = 1.0,
        zero_init_value: bool = False
    ):
        if not HAS_TORCH:
            raise RuntimeError("Afterstate learning requires PyTorch")
        self.input_dim = input_dim
        self.device = torch.device(device if (HAS_TORCH and torch.cuda.is_available() and device == "cuda") else "cpu")
        self.gamma = gamma
        self.model_type = model_type
        self.value_reward_scale = float(value_reward_scale)
        self.target_tau = float(target_tau)
        self.zero_init_value = bool(zero_init_value)
        if self.value_reward_scale <= 0:
            raise ValueError("value_reward_scale must be positive")
        if not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must be in (0, 1]")

        if HAS_TORCH:
            self.val_net = AfterstateValueNet(input_dim=input_dim, hidden_dim=hidden_dim,
                                               model_type=model_type).to(self.device)
            self.target_net = AfterstateValueNet(input_dim=input_dim, hidden_dim=hidden_dim,
                                                  model_type=model_type).to(self.device)
            if self.zero_init_value:
                # With reward-rate value units, a default random head can be
                # larger than the immediate action signal before a single
                # transition has been learned.  Starting V at zero makes the
                # initial policy exactly the configured immediate-reward policy.
                output = self.val_net.net if model_type == "linear" else self.val_net.net[-1]
                nn.init.zeros_(output.weight)
                if output.bias is not None:
                    nn.init.zeros_(output.bias)
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
    ) -> Tuple[int, Optional[np.ndarray], float]:
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
            immediate_reward: float -- predicted reward for action ranking.
                Training separately scores the observed transition.
        """
        if not candidates:
            return 0, None, 0.0

        if epsilon > 0.0 and random.random() < epsilon:
            chosen = random.choice(candidates)
            return chosen["action"], chosen.get("afterstate"), float(chosen.get("immediate_reward", 0.0))

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
            scores.append(self.value_reward_scale * imm_r + self.gamma * future_values[i])

        best_idx = int(np.argmax(scores))
        best_cand = candidates[best_idx]
        return best_cand["action"], best_cand.get("afterstate"), float(best_cand.get("immediate_reward", 0.0))

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
            targets = self.value_reward_scale * r_t + self.gamma * next_vals

        loss = F.smooth_l1_loss(pred_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.val_net.parameters(), 1.0)
        self.optimizer.step()
        self.update_target_network()

        return float(loss.item())

    def sync_target_network(self):
        """Update target network weights."""
        if HAS_TORCH and self.val_net is not None and self.target_net is not None:
            self.target_net.load_state_dict(self.val_net.state_dict())

    def update_target_network(self):
        """Polyak target update; tau=1 retains the historical hard update."""
        if HAS_TORCH and self.val_net is not None and self.target_net is not None:
            with torch.no_grad():
                for online, target in zip(self.val_net.parameters(), self.target_net.parameters()):
                    target.lerp_(online, self.target_tau)

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
                "model_type": self.model_type,
                "value_reward_scale": self.value_reward_scale,
                "target_tau": self.target_tau,
                "zero_init_value": self.zero_init_value,
                "model_state_dict": self.val_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            }, buf)
            meta = json.dumps({"model_type": "afterstate", "input_dim": self.input_dim,
                               "value_model_type": self.model_type,
                               "value_reward_scale": self.value_reward_scale,
                               "target_tau": self.target_tau,
                               "zero_init_value": self.zero_init_value}, indent=2)

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

            saved_type = data.get("model_type", "mlp")
            if saved_type != self.model_type:
                raise ValueError(
                    f"Checkpoint value model is {saved_type}, but configuration requests "
                    f"{self.model_type}; start a fresh run or use matching configuration")
            if int(data.get("input_dim", self.input_dim)) != self.input_dim:
                raise ValueError("Checkpoint input dimension does not match the resolved game schema")
            # Preserve the units in which the checkpoint learned V.  Historical
            # checkpoints predate this field and therefore use raw reward units.
            saved_scale = float(data.get("value_reward_scale", 1.0))
            if saved_scale <= 0:
                raise ValueError("Checkpoint value_reward_scale must be positive")
            self.value_reward_scale = saved_scale
            self.val_net.load_state_dict(data["model_state_dict"])
            self.target_net.load_state_dict(self.val_net.state_dict())
            if self.optimizer and "optimizer_state_dict" in data and data["optimizer_state_dict"]:
                self.optimizer.load_state_dict(data["optimizer_state_dict"])


def _verify_simulator(env, simulator, vars, spec, samples: int = 6):
    """Check the simulator's predicted afterstate against what the emulator
    actually does, before training on it. This catches wrong piece shapes on the
    FIRST run instead of after a wasted one -- exactly the bug this pipeline had,
    where standard tetrominoes were assumed and four of six were wrong.

    A few cells of pixel noise are tolerated; a wrong piece shifts many.
    """
    import sys
    n = getattr(simulator, "rows", 20) * getattr(simulator, "cols", 10)
    obs, info = env.reset()
    checked = mism = 0
    for _ in range(300):
        cands = simulator.get_candidates(obs=obs, ram=None, info=info, vars=vars, spec=spec)
        if not cands:
            obs, _r, term, trunc, info = env.step(None)
            if term or trunc:
                obs, info = env.reset()
            continue
        c = cands[len(cands) // 2]
        predicted = (np.asarray(c["afterstate"])[:n] > 0.5).astype(np.uint8)
        obs, _r, term, trunc, info = env.step(c["action"])
        actual = (np.asarray(obs)[:n] > 0.5).astype(np.uint8)
        diff = int(np.abs(predicted.astype(int) - actual.astype(int)).sum())
        checked += 1
        if diff > 4:
            mism += 1
        if term or trunc:
            obs, info = env.reset()
        if checked >= samples:
            break
    if checked and mism > checked // 2:
        print("")
        print("FATAL: the simulator's predicted board disagrees with the emulator on "
              "%d of %d sampled placements." % (mism, checked))
        print("The declared piece shapes almost certainly do not match this game, so "
              "training would learn on afterstates that never happen.")
        print("Re-measure the pieces and fix the 'afterstate.shapes' block in games.json.")
        sys.exit("Simulator shapes do not match the game.")
    print("simulator self-check : %d/%d sampled placements match the emulator"
          % (checked - mism, checked))


def _find_macro_wrapper(env):
    """Walk the wrapper chain to the MacroPlacementWrapper (it holds lock_frame)."""
    e = env
    for _ in range(8):
        if hasattr(e, "capture_lock") and hasattr(e, "lock_frame"):
            return e
        e = getattr(e, "env", None)
        if e is None:
            break
    return None


def unittest_afterstate(args, spec, overrides):
    n_samples = max(1, int(getattr(args, "unittest_samples", None) or 5))
    max_steps = max(100, n_samples * 16)    # enough placements to spread samples over
    """A short, VISUAL check that placements are decided and executed correctly.

    Runs at most `max_steps` afterstate placements (so it is quick), and for a
    handful of them saves one composite image per placement showing four moments:
      1. the new piece as it appears,
      2. what the model DECIDED to do with it (the predicted afterstate),
      3. the frame at the moment the piece hits the stack,
      4. the settled result, with the board the code reads overlaid and a verdict
         on whether the emulator produced the afterstate the model predicted.

    With --resume it loads a checkpoint first, so you can eyeball a trained model
    from the middle of a real run. Nothing is trained here; it only observes.
    """
    import sys
    import numpy as _np
    from rl.simulators import get_simulator
    from rl.env import make_env
    from rl.vars import GameVars
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        sys.exit("The visual unit test needs matplotlib. Install it (pip install matplotlib) and retry.")

    sim = get_simulator(spec.afterstate_config.get("simulator"), config=spec.afterstate_config)
    if sim is None or not getattr(sim, "shapes", None):
        sys.exit("This game has no afterstate simulator/shapes; nothing to unit-test.")

    out_dir = args.save_dir or os.path.join("checkpoints", args.game)
    out_dir = os.path.join(out_dir, "unittest_samples")
    os.makedirs(out_dir, exist_ok=True)

    env = make_env(args.game, overrides=overrides)
    vars = GameVars(spec.game, entry=spec.entry)
    macro = _find_macro_wrapper(env)
    if macro is not None:
        macro.capture_lock = True

    device = args.device or ("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
    agent = AfterstateAgent(input_dim=sim.feature_dim, device=device)
    if args.resume:
        print("loading weights from %s" % args.resume)
        agent.load(args.resume)
        tag = "trained"
    else:
        print("no --resume given: sampling an UNTRAINED model (decisions will be poor, "
              "but execution can still be verified)")
        tag = "untrained"

    grid = spec.grid or {}
    rows = int(grid.get("rows", 20)); cols = int(grid.get("cols", 10)); n = rows * cols
    gx = int(grid.get("x", 153)); gy = int(grid.get("y", 56)); cell = int(grid.get("cell", 8))
    names = {int(k): v for k, v in spec.afterstate_config.get("piece_names", {}).items()}

    def board_of(obs):
        return (_np.asarray(obs)[:n].reshape(rows, cols) > 0.5).astype(_np.uint8)

    def overlay(ax, frame, board, title):
        ax.imshow(frame)
        for r in range(rows):
            for c in range(cols):
                if board[r, c]:
                    ax.add_patch(Rectangle((gx + c * cell, gy + r * cell), cell, cell,
                                 fill=False, edgecolor="lime", lw=0.9))
        ax.set_xlim(0, frame.shape[1]); ax.set_ylim(frame.shape[0], 0)
        ax.set_title(title, fontsize=9); ax.axis("off")

    player = getattr(spec, "player", 2)

    def metrics_line(info):
        """The same game stats the training log prints, from info after the step."""
        parts = []
        for label, key in (("score", "score_p%d" % player), ("lines", "lines_p%d" % player),
                           ("holes", "holes"), ("filled", "filled"),
                           ("max_col_height", "max_height"), ("bumpiness", "bumpiness")):
            if key in info:
                v = info[key]
                parts.append("%s=%s" % (label, int(v) if float(v).is_integer() else round(float(v), 1)))
        return "   ".join(parts)

    def save_sample(idx, frame_spawn, piece_type, board_before, chosen,
                    frame_lock, frame_result, board_after, info):
        pred = (_np.asarray(chosen["afterstate"])[:n].reshape(rows, cols) > 0.5).astype(_np.uint8)
        diff = int(_np.abs(pred.astype(int) - board_after.astype(int)).sum())
        verdict = "MATCH" if diff <= 4 else "MISMATCH (%d cells)" % diff
        fig, ax = plt.subplots(1, 4, figsize=(19, 5.4))
        pname = names.get(piece_type, "?")
        overlay(ax[0], frame_spawn, board_before, "1. New piece appears: type %d (%s)" % (piece_type, pname))
        # panel 2: the decision, drawn as a PROPER cell grid (imshow centres
        # cells on integers, which made blocks look like they start at x.5).
        # Pre-existing stack is grey; the cells this placement adds are orange.
        added = (pred == 1) & (board_before == 0)
        for r in range(rows):
            for c in range(cols):
                if pred[r, c]:
                    face = "#f08a24" if added[r, c] else "#4a4a4a"
                else:
                    face = "#ffffff"
                ax[1].add_patch(Rectangle((c, r), 1, 1, facecolor=face,
                                edgecolor="#c8c8c8", linewidth=0.6))
        ax[1].set_xlim(0, cols); ax[1].set_ylim(rows, 0)   # row 0 at the top
        ax[1].set_aspect("equal")
        ax[1].set_xticks([c + 0.5 for c in range(cols)]); ax[1].set_xticklabels(range(cols), fontsize=7)
        ax[1].set_yticks([r + 0.5 for r in range(0, rows, 2)]); ax[1].set_yticklabels(range(0, rows, 2), fontsize=7)
        ax[1].tick_params(length=0)
        ax[1].set_title("2. Model decides: rot=%d col=%d\npredicts %d line(s) cleared "
                        "(orange = piece placed)"
                        % (chosen["rot"], chosen["col"], chosen["lines_cleared"]), fontsize=9)
        if frame_lock is not None:
            overlay(ax[2], frame_lock, board_before, "3. Piece hits the stack")
        else:
            ax[2].axis("off"); ax[2].set_title("3. (lock frame unavailable)", fontsize=9)
        overlay(ax[3], frame_result, board_after,
                "4. Result: execution %s\nboard now: %d filled cells" % (verdict, int(board_after.sum())))
        fig.suptitle("placement sample #%d  (%s model)" % (idx, tag), fontsize=11)
        # Game metrics after this placement -- the same stats the training log
        # shows (score / lines / holes / height / bumpiness). The log's other
        # columns (loss, reward) do not exist here because --unittest does not train.
        mline = metrics_line(info)
        if mline:
            fig.text(0.5, 0.015, "metrics after this placement:   " + mline,
                     ha="center", va="bottom", fontsize=10, family="monospace")
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        p = os.path.join(out_dir, "sample_%02d.png" % idx)
        fig.savefig(p, dpi=85); plt.close(fig)
        return p, verdict

    print("\nVisual unit test: up to %d placements, saving %d samples to %s\n"
          % (max_steps, n_samples, out_dir))
    obs, info = env.reset()
    placements = 0
    saved = []
    # spread samples across the run, but not the very first placements (empty board)
    sample_at = set(int(x) for x in _np.linspace(6, max_steps - 2, n_samples))
    while placements < max_steps:
        ram = getattr(env.unwrapped, "ram", None) if hasattr(env, "unwrapped") else None
        cands = sim.get_candidates(obs=obs, ram=ram, info=info, vars=vars, spec=spec)
        if not cands:
            obs, _r, term, trunc, info = env.step(None)
            if term or trunc:
                obs, info = env.reset()
            continue
        piece_type = int(_np.argmax(_np.asarray(obs)[n:n + sim.piece_types]))
        action, feat, _imm = agent.select_action(cands, epsilon=0.0)
        chosen = next((c for c in cands if c["action"] == action), cands[0])
        take_sample = placements in sample_at
        frame_spawn = env.render() if take_sample else None
        board_before = board_of(obs)
        if macro is not None:
            macro.lock_frame = None
        obs, _r, term, trunc, info = env.step(action)
        placements += 1
        if take_sample:
            board_after = board_of(obs)
            frame_lock = macro.lock_frame if macro is not None else None
            frame_result = env.render()
            p, verdict = save_sample(placements, frame_spawn, piece_type, board_before,
                                     chosen, frame_lock, frame_result, board_after, info)
            saved.append((p, verdict))
            print("  sample #%d saved: rot=%d col=%d predict_lines=%d  execution=%s  -> %s"
                  % (placements, chosen["rot"], chosen["col"], chosen["lines_cleared"], verdict, p))
        if term or trunc:
            obs, info = env.reset()
    env.close()
    matches = sum(1 for _p, v in saved if v == "MATCH")
    print("\nunit test done: %d samples, execution correct on %d/%d."
          % (len(saved), matches, len(saved)))
    print("Open the PNGs in %s to confirm the decisions and their execution by eye." % out_dir)
    return saved


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
    simulator = get_simulator(sim_name, config=spec.afterstate_config)
    if simulator is None:
        sys.exit(
            f"Error: Game '{args.game}' does not have an afterstate simulator registered.\n"
            f"Expected simulator '{sim_name}' in rl/simulators/."
        )
    if hasattr(simulator, "validate_training"):
        simulator.validate_training(spec)
    save_dir = args.save_dir or os.path.join("checkpoints", args.game)
    os.makedirs(save_dir, exist_ok=True)

    lr = args.lr if args.lr is not None else 2.5e-4
    gamma = args.gamma if args.gamma is not None else 0.99
    if hasattr(simulator, "discount"):
        simulator.discount = gamma
    batch_size = args.batch_size if args.batch_size is not None else 64
    device = args.device or ("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
    eps_start = args.ent_coef if args.ent_coef is not None else 0.20
    eps_final = args.ent_coef_final if args.ent_coef_final is not None else 0.01
    # Epsilon anneals over its OWN horizon, decoupled from the training length:
    # tying it to a 500k run left epsilon ~0.19 at 20k (1 in 5 placements random),
    # so measured play looked bad long before the schedule had moved. Defaults to
    # the full run when unset, preserving the old behaviour.
    explore_steps = getattr(args, "explore_steps", None) or args.timesteps

    # The terminal death penalty and the global reward scale are game values, so
    # they live in games.json (afterstate block), not hardcoded here.
    a_cfg = spec.afterstate_config or {}
    model_type = str(a_cfg.get("model_type", "mlp"))
    if model_type not in ("mlp", "linear"):
        raise ValueError("afterstate.model_type must be 'mlp' or 'linear'")
    value_units = str(a_cfg.get("value_units", "reward"))
    if value_units not in ("reward", "reward_rate"):
        raise ValueError("afterstate.value_units must be 'reward' or 'reward_rate'")
    value_reward_scale = (1.0 - gamma) if value_units == "reward_rate" else 1.0
    target_tau = float(a_cfg.get("target_tau", 1.0))
    zero_init_value = bool(a_cfg.get("zero_init_value", False))
    reward_scale = float(a_cfg.get("reward_scale", 1.0))
    terminal_penalty = float(a_cfg.get("terminal_penalty", 0.0)) * reward_scale
    if "terminal_penalty" not in a_cfg:
        raise ValueError("Declare afterstate.terminal_penalty explicitly (0 disables it)")
    max_wait_steps = int(a_cfg.get("max_wait_steps", 10000))
    max_wait_frames = int(a_cfg.get("max_wait_frames", 10000))
    if max_wait_steps <= 0 or max_wait_frames <= 0:
        raise ValueError("Afterstate wait limits must be positive")

    print(f"algorithm   : Afterstate Lookahead Value Network ({model_type})")
    print(f"simulator   : {sim_name} (feature dim: {simulator.feature_dim})")
    print(f"timesteps   : {args.timesteps}")
    print(f"device      : {device}")
    print(f"batch / lr  : {batch_size} / {lr} (gamma {gamma})")
    print(f"exploration : epsilon {eps_start} -> {eps_final} over {explore_steps} steps")
    print(f"reward scale: {reward_scale}   terminal penalty: {terminal_penalty:.2f}")
    print(f"value units : {value_units} (reward multiplier {value_reward_scale:g}); target tau {target_tau:g}")
    print(f"value init  : {'zero output' if zero_init_value else 'framework default'}")

    env = make_env(args.game, overrides=overrides)
    vars = GameVars(spec.game, entry=spec.entry)

    if sim_name == "grid_placement":
        _verify_simulator(env, simulator, vars, spec)

    agent = AfterstateAgent(
        input_dim=simulator.feature_dim,
        device=device,
        lr=lr,
        gamma=gamma,
        model_type=model_type,
        value_reward_scale=value_reward_scale,
        target_tau=target_tau,
        zero_init_value=zero_init_value
    )
    replay_capacity = int(a_cfg.get("replay_capacity", 50000))
    if replay_capacity < batch_size:
        raise ValueError("afterstate.replay_capacity must be at least batch_size")
    replay = AfterstateReplayBuffer(capacity=replay_capacity)
    backup = a_cfg.get("backup", "sampled")
    if backup not in ("sampled", "greedy_candidates"):
        raise ValueError("Unknown afterstate.backup: " + str(backup))
    control_replay = None
    if backup == "greedy_candidates":
        from rl.control import CandidateReplay, update_candidates
        # V learns task return, but the greedy backup must choose actions with
        # the same fixed candidate guidance used by the behaviour policy.  If
        # it maximizes task_reward alone, selection_delta disappears from the
        # backup and zero-reward ties collapse to an arbitrary first placement.
        control_replay = CandidateReplay(
            replay_capacity,
            reward_key="task_reward",
            selection_reward_key="immediate_reward",
        )
    print(f"value backup: {backup}; replay capacity {replay_capacity}")

    if args.resume:
        print(f"resuming weights from {args.resume}")
        agent.load(args.resume)

    total_steps = 0
    ep = 0
    recent_rewards = []
    recent_steps = []
    next_save_step = args.save_every

    print("\nBeginning training...")
    header = f"{'Steps':>8} | {'Ep':>5} | {'Len':>5} | {'AvgLen':>6} | {'Epsilon':>7} | {'Reward':>8} | {'AvgRew':>8} | {'Loss':>7} | Game Stats"
    print(header)
    print("-" * 90)

    while total_steps < args.timesteps:
        ep += 1
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        trajectory = AfterstateTrajectory(replay)
        pending = None
        mismatches = 0
        pending_boundary = None
        control_pending = None
        wait_steps = 0
        consecutive_waits = 0
        waiting_frames = 0
        ep_loss = 0.0
        loss_updates = 0
        last_info = info
        control_previous = None

        eps_progress = min(1.0, total_steps / float(max(1, explore_steps)))
        epsilon = eps_start + (eps_final - eps_start) * eps_progress

        while not done and (total_steps < args.timesteps or pending is not None):
            ram = getattr(env.unwrapped, "ram", None) if hasattr(env, "unwrapped") else None
            candidates = simulator.get_candidates(obs=obs, ram=ram, info=info, vars=vars, spec=spec)

            action = None
            control_added = False
            if pending is None and candidates:
                control_item = None
                if control_replay is not None and control_previous is not None:
                    # Sample the real next piece, optimize its action choice.
                    # Do not turn missing candidates/transient frames into death.
                    control_item = control_replay.push(control_previous, candidates=candidates)
                    control_previous = None
                action, predicted, _imm_reward = agent.select_action(candidates, epsilon=epsilon)
                if control_item is not None:
                    control_pending = (control_item, next(i for i, c in enumerate(candidates) if c['action'] == action))
                pending = (np.array(obs, copy=True), dict(info), predicted)
            next_obs, reward, terminated, truncated, info = env.step(action)
            if action is None:
                consecutive_waits += 1
                waiting_frames += max(0, int(info.get("frames", 0)) - int(last_info.get("frames", 0)))
            else:
                consecutive_waits = waiting_frames = 0
            if (not (terminated or truncated) and
                    (consecutive_waits >= max_wait_steps or waiting_frames >= max_wait_frames)):
                # Abort a broken environment contract, without inventing a
                # terminal transition or repeatedly resetting into the stall.
                agent.save(os.path.join(save_dir, "wait_timeout.zip"))
                env.close()
                raise RuntimeError(
                    f"Afterstate wait limit reached for {args.game}: "
                    f"{consecutive_waits} neutral steps / {waiting_frames} frames; "
                    "check spawn detection and afterstate_ready. Saved wait_timeout.zip.")
            done = terminated or truncated
            last_info = info

            step_reward = 0.0
            new_transition = False
            if terminated:
                if pending is not None:
                    before, before_info, _predicted = pending
                    # A clear that also crossed a level boundary must be credited
                    # from the counters captured BEFORE the redraw, even when the
                    # same placement ends the game. This branch runs before the
                    # discontinuity branch, so without this those lines were lost.
                    terminal_info = info
                    if pending_boundary is not None:
                        terminal_info = pending_boundary
                    elif info.get("afterstate_discontinuity", False):
                        terminal_info = dict(info.get("afterstate_boundary_info", info))
                    step_reward = simulator.observed_reward(
                        before, next_obs, before_info, terminal_info, terminated=True)
                step_reward += terminal_penalty
                pending_boundary = None
                if control_pending is not None:
                    control_replay.correct(*control_pending, step_reward, terminal=True)
                    control_added = True
                    control_pending = None
                if control_replay is not None and control_previous is not None:
                    control_replay.push(control_previous, terminal_reward=step_reward)
                    control_added = True
                    control_previous = None
                new_transition = trajectory.record(step_reward, terminated=True)
                pending = None
            elif info.get("afterstate_discontinuity", False):
                # Keep the successful action and last real afterstate. Capture
                # its counters before the redraw, then wait for a settled board.
                # Keep the FIRST boundary: it is the one that bounds the clear the
                # still-pending placement made. A later boundary arriving before
                # the board settles must not overwrite and discard it.
                if pending_boundary is None:
                    pending_boundary = dict(info.get("afterstate_boundary_info", info))
            elif pending is not None and info.get("afterstate_ready", True):
                before, before_info, predicted = pending
                actual = simulator.encode_observation(next_obs, info)
                control_previous = np.array(actual, copy=True)
                if pending_boundary is not None:
                    step_reward = simulator.discontinuity_reward(before, next_obs, before_info, pending_boundary)
                    pending_boundary = None
                else:
                    step_reward = simulator.observed_reward(before, next_obs, before_info, info)
                if control_pending is not None:
                    control_replay.correct(*control_pending, step_reward, observed=actual)
                    control_added = True
                    control_pending = None
                new_transition = trajectory.record(step_reward, actual)
                mismatches += int(not np.array_equal(predicted, actual))
                pending = None
            ep_reward += step_reward
            if action is not None:
                ep_steps += 1
                total_steps += 1
            else:
                wait_steps += 1
                # An entirely unplayable episode must still consume budget.
                if done and ep_steps == 0:
                    total_steps += 1

            if control_replay is not None and control_added and len(control_replay) >= batch_size:
                loss = update_candidates(agent, control_replay, batch_size)
                ep_loss += loss
                loss_updates += 1
            elif control_replay is None and new_transition and len(replay) >= batch_size:
                loss = agent.update(replay, batch_size=batch_size)
                ep_loss += loss
                loss_updates += 1

            if target_tau >= 1.0 and action is not None and total_steps % 500 == 0:
                agent.sync_target_network()

            if total_steps >= next_save_step:
                ckpt_path = os.path.join(save_dir, f"ckpt_{next_save_step}_steps.zip")
                agent.save(ckpt_path)
                print(f"  --> Saved checkpoint: {ckpt_path} (step {total_steps})")
                next_save_step += args.save_every

            obs = next_obs

        recent_rewards.append(ep_reward)
        recent_steps.append(ep_steps)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
            recent_steps.pop(0)

        avg_rew = np.mean(recent_rewards)
        avg_len = np.mean(recent_steps)
        avg_loss = (ep_loss / max(1, loss_updates)) if loss_updates > 0 else 0.0

        # Display domain-specific game metrics declared in games.json
        keys_to_show = getattr(spec, "report_stats", None)
        if not keys_to_show:
            # Fallback: variables from reward terms that are not event triggers
            keys_to_show = []
            for tm in getattr(spec, "terms", []):
                if tm.get("kind") in ("delta", "delta_tiered", "feature_delta", "feature_level"):
                    v = tm.get("var") or tm.get("name")
                    if v and v not in keys_to_show:
                        keys_to_show.append(v)

        stat_parts = []
        for k in keys_to_show:
            if k in last_info:
                val = last_info[k]
                if isinstance(val, (int, np.integer)):
                    stat_parts.append(f"{k}={val}")
                elif isinstance(val, (float, np.floating)):
                    stat_parts.append(f"{k}={val:.1f}" if not val.is_integer() else f"{k}={int(val)}")

        stats_str = " ".join(stat_parts) + f" prediction_mismatches={mismatches} waits={wait_steps} start={last_info.get('start_state', '')}"

        print(f"{total_steps:8d} | {ep:5d} | {ep_steps:5d} | {avg_len:6.1f} | {epsilon:7.3f} | {ep_reward:8.1f} | {avg_rew:8.1f} | {avg_loss:7.4f} | {stats_str}")

    final_path = os.path.join(save_dir, "final.zip")
    agent.save(final_path)
    print(f"\nTraining complete. Final weights saved to {final_path}")
    env.close()
