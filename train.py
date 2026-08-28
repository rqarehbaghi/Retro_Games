#!/usr/bin/env python3
"""
Train a PPO agent from scratch (Stable-Baselines3) to play an NES game via
stable-retro, running entirely on your own machine. "From scratch" means:
a freshly-initialized CNN policy network with random weights -- unless
you pass --resume-from a checkpoint (including one produced by
pretrain_imitation.py from your own recorded gameplay), in which case
training continues from those weights instead.

Checkpoints go in a subfolder named after --game, e.g.
./checkpoints/SuperMarioBros3-Nes-v0/, so different games' progress never
mixes. Every checkpoint filename embeds the cumulative iteration count
across ALL runs (including resumed ones), e.g. iter_372.zip -- so you
never have to track that number yourself.

"Iteration" here means the same thing PPO's own console logging reports:
one rollout-collection + policy-update cycle, i.e. (n_steps * num_envs)
environment steps.

ACTIONS AND PRESS DURATION: unlike a simple discretizer that just picks a
button combo and holds it for a fixed number of frames, the action table
below gives the agent separate, explicitly different actions for a short
tap vs. a long hold of the jump button (e.g. NES Mario's jump height
depends on how long A is held) -- these are genuinely different, directly
selectable actions, not something that only emerges by coincidence of
picking the same action on consecutive decisions.

REWARD SHAPING: a bare "did the game's built-in score go up" signal is
often too sparse for PPO to learn much from before it converges on
degenerate behavior (e.g. walking right until something kills it, since
that's the easiest way to rack up a little x-distance before dying). This
script adds a death penalty, a small per-frame survival reward, a
progress-toward-goal reward, and a jump/stuck incentive on top of
whatever reward the game's own stable-retro integration provides.

Usage:
    # Train for 100 iterations (the default) from a fresh random network:
    python train.py --game SuperMarioBros3-Nes-v0

    # Stop early any time with Ctrl+C -- it autosaves before exiting.

    # Continue training an existing checkpoint for 200 more iterations:
    python train.py --game SuperMarioBros3-Nes-v0 \\
        --resume-from ./checkpoints/SuperMarioBros3-Nes-v0/latest_iter_100.zip \\
        --iterations 200

    # Continue from a human-demonstration warm start (see pretrain_imitation.py),
    # with a gentler learning rate so it doesn't unlearn the demonstration fast:
    python train.py --game SuperMarioBros3-Nes-v0 \\
        --resume-from ./checkpoints/SuperMarioBros3-Nes-v0/pretrained_human_bc.zip \\
        --lr 3e-5 --ent-coef 0.001 --iterations 300
"""
import argparse
import os
import re

import cv2
import numpy as np
import stable_retro as retro
from gymnasium import ObservationWrapper, Wrapper
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor

# Each entry is (button combo, hold_frames): how many raw emulator frames
# the buttons stay pressed once this action is chosen. This is what makes
# "short tap" and "long hold" of the same button genuinely different,
# separately-selectable actions instead of something that only happens by
# coincidence. Button NAMES are looked up dynamically against this game's
# actual button list, not a hardcoded order, so this stays correct across
# different games/cores.
#
# Tuned for platformer-style games with an NES-style variable jump height
# (A = jump). Edit this table for a different genre or console.
ACTION_TABLE = [
    ([], 4),                    # no-op / release
    (["LEFT"], 4),
    (["RIGHT"], 4),
    (["RIGHT", "B"], 4),        # run right
    (["LEFT", "B"], 4),         # run left
    (["A"], 6),                 # short hop, in place
    (["A"], 20),                # full-height jump, in place
    (["RIGHT", "A"], 6),        # short forward hop
    (["RIGHT", "A"], 20),       # full forward jump
    (["LEFT", "A"], 6),
    (["LEFT", "A"], 20),
    (["DOWN"], 4),
    (["UP"], 4),
]

# Indices into ACTION_TABLE whose combo includes the jump button -- used
# by JumpIncentiveWrapper below to know "was this decision a jump
# attempt" directly, rather than inspecting the raw button array (which
# doesn't work: at the point reward-shaping wrappers run, the action is
# still the discrete index chosen by the policy, not yet expanded into a
# button array -- trying to index into it as if it were one silently
# never matches).
JUMP_BUTTON = "A"


def jump_action_indices(action_table=ACTION_TABLE):
    """Which ACTION_TABLE indices include the jump button. A pure
    function of the table itself (no live game/env needed), so
    pretrain_imitation.py can use the exact same jump-detection logic
    VariableHoldDiscretizer uses, e.g. for weighting jump examples more
    heavily in the imitation-learning loss."""
    return {i for i, (combo, _hold) in enumerate(action_table) if JUMP_BUTTON in combo}


# Different stable-retro integrations expose horizontal position under
# different info keys. SuperMarioBros3-Nes-v0, for instance, calls it 'hpos',
# while others use 'x' or 'x_pos'. The progress-reward and stuck-detection
# shaping is worthless if it reads the wrong name (it silently sees None and
# does nothing), so look the value up across the known aliases.
X_POS_KEYS = ("x", "x_pos", "hpos", "xpos", "x_position")


def read_x(info):
    """Horizontal position from `info`, across the key names different game
    integrations use. Returns None if the integration exposes none of them --
    in which case progress/stuck shaping simply can't apply (check your game's
    info keys, see the README)."""
    for key in X_POS_KEYS:
        if key in info:
            return info[key]
    return None


class VariableHoldDiscretizer(Wrapper):
    """Collapses stable-retro's full MultiBinary button space down to a
    small set of meaningful (combo, hold_frames) actions, and internally
    repeats the raw env.step() for however many frames THIS action calls
    for. Learning a handful of sane combos is far faster than learning
    which of 256+ raw button states matter, and giving jump its own
    short/long variants is what lets the agent express press duration at
    all -- a fixed-length repeat can never do that regardless of how long
    training runs."""

    def __init__(self, env, action_table):
        super().__init__(env)
        buttons = env.unwrapped.buttons
        self._raw_actions = []
        self._hold_frames = []
        for combo, hold in action_table:
            arr = np.array([False] * len(buttons))
            for button in combo:
                arr[buttons.index(button)] = True
            self._raw_actions.append(arr)
            self._hold_frames.append(hold)
        self.jump_action_indices = jump_action_indices(action_table)
        self.action_space = Discrete(len(action_table))

    def step(self, action):
        raw = self._raw_actions[action]
        hold = self._hold_frames[action]
        total_reward = 0.0
        frames_used = 0
        obs = terminated = truncated = info = None
        for _ in range(hold):
            obs, reward, terminated, truncated, info = self.env.step(raw)
            total_reward += reward
            frames_used += 1
            if terminated or truncated:
                break
        # So callers counting real emulator frames (e.g. --max-steps in
        # play_and_record.py) stay accurate even though hold length now
        # varies per action, and can end mid-hold on death/win.
        info["frames_this_step"] = frames_used
        return obs, total_reward, terminated, truncated, info


class WarpFrame(ObservationWrapper):
    """Grayscale + resize to 84x84 -- standard RL preprocessing. This is
    most of what makes training practical on a single machine: far less
    data per frame for the network to chew through."""

    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = Box(low=0, high=255, shape=(size, size, 1), dtype=np.uint8)

    def observation(self, obs):
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class RewardShaper(Wrapper):
    """Adds a death penalty, a tiny per-decision survival reward, and a
    reward for horizontal progress, on top of whatever the game's own
    stable-retro integration already provides. Without this, PPO on a
    sparse/naive reward commonly converges on exactly the failure mode of
    'walk right until something kills you' -- that's the cheapest way to
    rack up a little reward before dying, and nothing here was pushing
    back against it.

    Placement matters: this wraps the discretizer (see make_env), so one
    step() here == one agent DECISION, not one emulator frame. That keeps
    the flat survival_tick worth the same regardless of an action's hold
    length -- wrapping the raw env instead would multiply the tick by the
    hold length (4-20 frames), silently biasing the policy toward
    longer-hold actions just for existing. The horizontal-progress reward
    telescopes to the true per-decision x-delta either way.

    Reads x/lives/health from `info` if the game's integration exposes
    them (varies by game -- check yours if this seems to have no effect,
    see the README). Silently does nothing extra if it doesn't."""

    def __init__(self, env, death_penalty=50.0, survival_tick=0.0, progress_scale=0.1,
                 score_scale=0.01, time_penalty=0.0, life_bonus=25.0,
                 power_bonus=0.0, powerup_address=None, end_on_life_loss=True):
        super().__init__(env)
        self.death_penalty = death_penalty
        self.survival_tick = survival_tick
        self.progress_scale = progress_scale
        # Reward for the game's own score going up: coins, power-ups, stomped
        # enemies, AND the leftover-time bonus paid out when a level is cleared
        # all raise `score`. So this single term rewards "collect points" and,
        # because finishing faster leaves more time to convert to score, also
        # rewards finishing quickly. `time_penalty` is a small per-decision cost
        # that discourages dawdling before the agent can reliably clear levels.
        self.score_scale = score_scale
        self.time_penalty = time_penalty
        self.life_bonus = life_bonus
        self.power_bonus = power_bonus
        # Optional RAM index to read the power-up tier from directly.
        # Integrations often don't publish power state in `info` (this one
        # doesn't), and adding it to the integration's data.json means
        # editing site-packages -- lost on any venv rebuild and outside
        # version control. Reading the byte here keeps it in the repo.
        # Locate the address for your game with find_ram_variable.py.
        self.powerup_address = powerup_address
        self.prev_score = None
        self.prev_power = None
        # End the episode the moment a life is lost, so the vec-env auto-reset
        # returns to the clean in-level default state instead of letting the
        # emulator run on into the post-death world map / continue screen --
        # an out-of-distribution screen with no shaped reward where the policy
        # has no signal and just stalls. Requires the integration to expose
        # `lives` in info; if it doesn't, this silently has no effect (see the
        # info-keys check in the README) and the episode ends only when the
        # game's own scenario says so.
        self.end_on_life_loss = end_on_life_loss
        self.prev_x = None
        self.prev_lives = None
        self.prev_health = None

    def _read_power(self, info):
        """Power-up tier: from the configured RAM address if one was given,
        else from `info` if the integration happens to publish it. Returns
        None if neither is available, which disables power shaping."""
        if self.powerup_address is not None:
            try:
                return int(self.env.unwrapped.get_ram()[self.powerup_address])
            except Exception:
                return None
        return info.get("powerup", info.get("power", info.get("status")))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = read_x(info)
        self.prev_lives = info.get("lives")
        self.prev_health = info.get("health")
        self.prev_score = info.get("score")
        self.prev_power = self._read_power(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward += self.survival_tick
        reward -= self.time_penalty

        current_x = read_x(info)
        if current_x is not None and self.prev_x is not None:
            reward += max(0, current_x - self.prev_x) * self.progress_scale
        if current_x is not None:
            self.prev_x = current_x

        # Points: coins, power-ups, stomped enemies, and the level-clear time
        # bonus all raise score. Clamp to >=0 so a score reset on death/new
        # level isn't read as a negative reward.
        current_score = info.get("score")
        if current_score is not None and self.prev_score is not None:
            reward += max(0, current_score - self.prev_score) * self.score_scale
        if current_score is not None:
            self.prev_score = current_score

        # Power-up tier. Verified empirically for SuperMarioBros3-Nes-v0 at
        # RAM 0x00ED: 0=small, 1=big (mushroom), 2=fire, 3=raccoon -- i.e.
        # monotonically ordered by power, which is what makes the linear
        # `delta * power_bonus` below correct in both directions (each tier
        # gained pays the same, and a hit dropping raccoon->small costs 3x a
        # big->small hit). If you port this to a game whose numbering is NOT
        # ordered by desirability, replace the linear term with an explicit
        # per-tier value table. Only available
        # if the integration publishes it; see find_ram_variable.py for
        # locating the RAM address and the README for adding it to the
        # integration's data.json as `powerup`. Rewarded on INCREASE and
        # penalized on decrease (taking a hit and shrinking is a real loss),
        # scaled by how many tiers changed.
        current_power = self._read_power(info)
        if current_power is not None and self.prev_power is not None and self.power_bonus:
            delta = int(current_power) - int(self.prev_power)
            if delta > 0:
                reward += delta * self.power_bonus
            elif delta < 0:
                reward += delta * self.power_bonus  # delta negative -> penalty
        if current_power is not None:
            self.prev_power = current_power

        current_lives = info.get("lives")
        lost_life = (
            current_lives is not None and self.prev_lives is not None
            and current_lives < self.prev_lives
        )
        # Lives going UP means a 1-Up: the green mushroom, the 100-coin
        # threshold, or a score milestone. Worth an explicit reward -- an extra
        # life is strategically far more valuable than the handful of points a
        # 1-Up adds to `score`, so score_scale alone badly undervalues it.
        # (Regular power-up mushrooms are NOT detectable here: this
        # integration exposes no health/power-state variable, only hpos/lives/
        # score/time -- so those are rewarded solely through their score.)
        if (current_lives is not None and self.prev_lives is not None
                and current_lives > self.prev_lives):
            reward += (current_lives - self.prev_lives) * self.life_bonus
        if current_lives is not None:
            self.prev_lives = current_lives

        current_health = info.get("health")
        if current_health is not None and self.prev_health is not None and current_health < self.prev_health:
            reward -= (self.prev_health - current_health) * 2.0
        if current_health is not None:
            self.prev_health = current_health

        if lost_life and self.end_on_life_loss:
            terminated = True

        # One death penalty per non-clear episode end. This now also covers a
        # life-loss death (it just terminated the episode above), so we don't
        # add a second, separate life-loss penalty and double-count.
        if (terminated or truncated) and not info.get("is_stage_clear", False):
            reward -= self.death_penalty
        elif lost_life:
            # Life lost but end_on_life_loss is off and the game didn't end the
            # episode: still penalize the death, play just continues.
            reward -= self.death_penalty

        return obs, reward, terminated, truncated, info


class JumpIncentiveWrapper(Wrapper):
    """Must wrap a VariableHoldDiscretizer (reads its .jump_action_indices
    and receives its DISCRETE action, not a raw button array -- this is
    what the previous version got wrong: it tried to check the action as
    if it were already a button array at a point in the wrapper stack
    where it was still just the discrete index PPO picked, so the check
    silently never matched).

    Rewards choosing a jump action, and penalizes being stalled against
    an obstacle without jumping -- targets the specific 'runs into a pipe
    and just keeps walking into it' failure mode."""

    def __init__(self, discretizer_env, jump_bonus=0.2, stuck_penalty=0.05, stuck_frames=8):
        super().__init__(discretizer_env)
        self.jump_action_indices = discretizer_env.jump_action_indices
        self.jump_bonus = jump_bonus
        self.stuck_penalty = stuck_penalty
        self.stuck_frames = stuck_frames
        self.prev_x = None
        self.stalled_frames = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = read_x(info)
        self.stalled_frames = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if action in self.jump_action_indices:
            reward += self.jump_bonus

        current_x = read_x(info)
        if current_x is not None and self.prev_x is not None:
            if abs(current_x - self.prev_x) < 0.5:
                self.stalled_frames += 1
                if self.stalled_frames > self.stuck_frames:
                    reward -= self.stuck_penalty
            else:
                self.stalled_frames = 0
        if current_x is not None:
            self.prev_x = current_x

        return obs, reward, terminated, truncated, info


def make_env(game, state, death_penalty, jump_bonus, render=False, end_on_life_loss=True,
             score_bonus=0.01, time_penalty=0.0, life_bonus=25.0, power_bonus=0.0, powerup_address=None):
    def _init():
        render_mode = "human" if render else "rgb_array"
        env = retro.make(game=game, state=state or retro.State.DEFAULT, render_mode=render_mode)
        # Order matters. The discretizer must be innermost so everything
        # above it operates at the level of one agent decision. JumpIncentive
        # must wrap the discretizer directly (it reads .jump_action_indices
        # and receives the discrete action). RewardShaper then sits on top so
        # its survival tick / progress / death shaping is applied once per
        # decision, not once per emulator frame (see RewardShaper docstring).
        env = VariableHoldDiscretizer(env, ACTION_TABLE)
        env = JumpIncentiveWrapper(env, jump_bonus=jump_bonus)
        env = RewardShaper(env, death_penalty=death_penalty, end_on_life_loss=end_on_life_loss,
                           score_scale=score_bonus, time_penalty=time_penalty,
                           life_bonus=life_bonus, power_bonus=power_bonus,
                           powerup_address=powerup_address)
        env = WarpFrame(env)
        return env
    return _init


def safe_name(game):
    """Turns a game id into something guaranteed safe as a folder name.
    Game ids are normally already filesystem-safe, but this guards
    against the unexpected rather than assuming."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", game)


def parse_iteration(path):
    """Pulls the cumulative iteration count back out of a checkpoint
    filename like iter_372.zip or latest_iter_372.zip. Returns None if the
    filename doesn't follow that pattern (e.g. it was renamed by hand)."""
    match = re.search(r"iter_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


class IterationCheckpointCallback(BaseCallback):
    """Saves the model at specific PPO training iterations -- the same
    'iterations' counter PPO's own logging reports -- rather than on a
    fixed timestep schedule. Fires at the START of the target iteration,
    so iteration 1's checkpoint is the untouched, randomly-initialized
    network (or the resumed-from checkpoint, unchanged), before any
    update in this run has happened.

    Also keeps a rolling 'latest_iter_N.zip' every autosave_every
    iterations (previous one deleted each time) as a crash/resume safety
    net independent of the milestone list."""

    def __init__(self, milestones, out_dir, autosave_every=25, start_iteration=0, verbose=0):
        super().__init__(verbose)
        self.milestones = set(milestones)
        self.out_dir = out_dir
        self.autosave_every = autosave_every
        self.iteration = start_iteration
        self._last_autosave_path = None
        os.makedirs(out_dir, exist_ok=True)

    def save_now(self, tag="iter"):
        path = os.path.join(self.out_dir, f"{tag}_{self.iteration}.zip")
        self.model.save(path)
        return path

    def _on_rollout_start(self):
        self.iteration += 1

        if self.iteration in self.milestones:
            path = self.save_now(tag="iter")
            if self.verbose:
                print(f"Saved milestone checkpoint: {path}")

        if self.autosave_every and self.iteration % self.autosave_every == 0:
            if self._last_autosave_path and os.path.exists(self._last_autosave_path):
                os.remove(self._last_autosave_path)
            self._last_autosave_path = self.save_now(tag="latest_iter")
            if self.verbose:
                print(f"Autosaved: {self._last_autosave_path} -- safe to resume from here if training stops")

    def _on_step(self):
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True, help="stable-retro game id, e.g. SuperMarioBros3-Nes-v0")
    parser.add_argument("--state", default=None, help="Save state to start from (default state if omitted)")
    parser.add_argument("--iterations", type=int, default=100, help="How many MORE iterations to run this invocation -- the main training-length knob. (default: %(default)s)")
    parser.add_argument("--num-envs", type=int, default=8, help="Parallel emulator instances -- match roughly to your CPU core count. (default: %(default)s)")
    parser.add_argument("--n-steps", type=int, default=128, help="Env steps collected per env before each PPO update. (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="PPO learning rate. Lower this (e.g. 3e-5) when resuming from an imitation-pretrained checkpoint, so RL fine-tuning doesn't wash out what it already learned. (default: %(default)s)")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="PPO entropy coefficient -- higher encourages more exploration. Lower this (e.g. 0.001) when fine-tuning a pretrained checkpoint. (default: %(default)s)")
    parser.add_argument("--death-penalty", type=float, default=50.0, help="Reward subtracted on death/episode-end without clearing the stage. (default: %(default)s)")
    parser.add_argument("--jump-bonus", type=float, default=0.2, help="Reward added for choosing a jump action. Set to 0 to disable jump-incentive shaping entirely. (default: %(default)s)")
    parser.add_argument("--score-bonus", type=float, default=0.01, help="Reward per point the game's own score goes up -- coins, power-ups, stomped enemies, and the level-clear leftover-time bonus. This is what makes the agent value points AND finishing fast (faster clear = more time converted to score). Raise it to prioritize collecting/points, lower toward 0 for a pure speedrun-right agent. Tune to your game's score magnitudes. (default: %(default)s)")
    parser.add_argument("--life-bonus", type=float, default=25.0, help="Reward for each extra life GAINED -- 1-Up mushrooms, the 100-coin threshold, score milestones. An extra life is worth far more strategically than the few points a 1-Up adds to score, so --score-bonus alone undervalues it. Set to 0 to disable. Note regular power-up mushrooms can't be rewarded directly (this integration exposes no power-state variable), only via their score. (default: %(default)s)")
    parser.add_argument("--power-bonus", type=float, default=0.0, help="Reward per power-up TIER gained (small->big->fire->raccoon...), and the same penalty per tier lost when you take a hit. Requires the integration to publish the power state in info as `powerup` -- it is NOT exposed by default. Use find_ram_variable.py to locate the RAM address in your own recording, add it to the integration data.json, then set this (try 10-20). Left at 0 it does nothing. (default: %(default)s)")
    parser.add_argument("--powerup-address", type=lambda v: int(v, 0), default=None, help="RAM index holding the power-up tier, read directly so the integration does not need to publish it (SuperMarioBros3-Nes-v0: 0x00ED, found with find_ram_variable.py -- verify for your own game/version). Accepts hex (0x00ED) or decimal. Needed for --power-bonus to do anything here. (default: %(default)s)")
    parser.add_argument("--time-penalty", type=float, default=0.0, help="Small reward subtracted every decision, to discourage dawdling and push toward finishing the level sooner. Start around 0.01-0.05 if the agent loiters; too high and it rushes into danger. (default: %(default)s)")
    parser.add_argument("--no-end-on-death", dest="end_on_life_loss", action="store_false", help="By default the episode ends the instant a life is lost, so training always restarts from the clean in-level state instead of wandering onto the post-death world map / continue screen. Pass this to disable that and let the game's own scenario decide when an episode ends. (Requires the integration to expose 'lives' in info either way.)")
    parser.set_defaults(end_on_life_loss=True)
    parser.add_argument("--checkpoint-iterations", type=int, nargs="+", default=None, help="Which cumulative iterations to snapshot as named milestones. Defaults to [1, <the iteration this run ends on>] if omitted.")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Parent folder -- a subfolder named after --game is created inside it. (default: %(default)s)")
    parser.add_argument("--autosave-every", type=int, default=25, help="Also save a rolling latest_iter_N.zip every N iterations, regardless of --checkpoint-iterations -- your crash/resume safety net. Set to 0 to disable. (default: %(default)s)")
    parser.add_argument("--render", action="store_true", help="Open a live emulator window for ONE of the parallel envs so you can watch training happen. Needs a display (see README) and slows training down. To watch cleanly WITHOUT the slowdown, train headless and periodically play a checkpoint with: play_and_record.py --model <ckpt> --render")
    parser.add_argument("--resume-from", default=None, help="Path to a checkpoint to continue training from -- a previous PPO run, OR an imitation-pretrained checkpoint from pretrain_imitation.py. The cumulative iteration count is read from its filename.")
    parser.add_argument("--start-iteration", type=int, default=None, help="Override the cumulative iteration count when resuming, if the checkpoint filename doesn't encode it (e.g. you renamed it).")
    args = parser.parse_args()

    # A live window means one env in THIS process (DummyVecEnv). SubprocVecEnv
    # requires every env share a render mode, and running many "human" windows
    # would be unwatchable and crawl -- so --render forces a single visible env.
    # Drop --render to go back to full headless parallelism.
    if args.render and args.num_envs > 1:
        print(
            f"--render is on: using a single live env instead of {args.num_envs} "
            f"parallel ones (you can only sanely watch one window, and mixing "
            f"render modes isn't allowed). Remove --render to train with "
            f"{args.num_envs} envs at full speed."
        )
        args.num_envs = 1

    checkpoint_dir = os.path.join(args.checkpoint_dir, safe_name(args.game))

    start_iteration = 0
    if args.resume_from:
        parsed = parse_iteration(args.resume_from)
        if args.start_iteration is not None:
            start_iteration = args.start_iteration
            print(f"Resuming from {args.resume_from} at iteration {start_iteration} (from --start-iteration)")
        elif parsed is not None:
            start_iteration = parsed
            print(f"Resuming from {args.resume_from} at iteration {start_iteration} (read from filename)")
        else:
            print(
                f"Resuming from {args.resume_from} -- couldn't read an iteration count from "
                f"that filename. Starting the counter at 0 for checkpoint-naming purposes "
                f"(the model's own learned weights are still fully preserved either way). "
                f"Pass --start-iteration N to set this correctly."
            )

    end_iteration = start_iteration + args.iterations
    if args.checkpoint_iterations is None:
        args.checkpoint_iterations = sorted({1, end_iteration})

    steps_per_iteration = args.n_steps * args.num_envs
    total_timesteps = args.iterations * steps_per_iteration

    print(f"Checkpoints folder: {checkpoint_dir}")
    print(f"Steps per iteration: {steps_per_iteration:,} ({args.n_steps} n_steps x {args.num_envs} envs)")
    print(f"This run: iterations {start_iteration + 1}-{end_iteration} ({total_timesteps:,} env steps)")
    print(f"Reward shaping: death_penalty={args.death_penalty}, jump_bonus={args.jump_bonus}, score_bonus={args.score_bonus}, life_bonus={args.life_bonus}, power_bonus={args.power_bonus}, powerup_address={args.powerup_address}, time_penalty={args.time_penalty}, end_on_life_loss={args.end_on_life_loss}")
    print(
        "Rough guide from earlier: simple games often reach solid play in "
        "hours, medium-complexity platformers in about a day, on a modern "
        "multi-core machine with several parallel envs. This is a guide, "
        "not a promise -- actual time depends heavily on the game."
    )

    # --render runs the single env in-process (DummyVecEnv) so its window lives
    # in the main process and stays responsive. Headless training keeps using
    # SubprocVecEnv for true parallelism across --num-envs processes.
    if args.render:
        env = DummyVecEnv([
            make_env(args.game, args.state, args.death_penalty, args.jump_bonus,
                     render=True, end_on_life_loss=args.end_on_life_loss,
                     score_bonus=args.score_bonus, time_penalty=args.time_penalty,
                     life_bonus=args.life_bonus, power_bonus=args.power_bonus,
                     powerup_address=args.powerup_address)
        ])
    else:
        env = SubprocVecEnv([
            make_env(args.game, args.state, args.death_penalty, args.jump_bonus,
                     end_on_life_loss=args.end_on_life_loss,
                     score_bonus=args.score_bonus, time_penalty=args.time_penalty,
                     life_bonus=args.life_bonus, power_bonus=args.power_bonus,
                     powerup_address=args.powerup_address)
            for _ in range(args.num_envs)
        ])
    # Without a Monitor layer SB3 has no episode statistics at all --
    # rollout/ep_rew_mean and ep_len_mean simply never appear in the console
    # table, leaving no way to tell whether training is working. VecMonitor
    # records per-episode shaped reward and length at the vec-env level.
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4)

    if args.resume_from:
        # PPO.load restores the checkpoint's OWN saved hyperparameters,
        # including n_steps. The imitation checkpoint from pretrain_imitation.py
        # is built with PPO's default n_steps=2048, so a plain load would make a
        # resumed run collect 2048 steps/env per iteration instead of --n-steps
        # -- ~16x the printed steps-per-iteration, so iteration 1 appears to
        # hang for a very long time. custom_objects is applied BEFORE the
        # rollout buffer and LR schedule are rebuilt, so overriding n_steps here
        # actually resizes the buffer (and reschedules the LR) correctly, unlike
        # assigning the attributes after load.
        model = PPO.load(
            args.resume_from,
            env=env,
            custom_objects={
                "n_steps": args.n_steps,
                "learning_rate": args.lr,
                "ent_coef": args.ent_coef,
            },
        )
    else:
        model = PPO("CnnPolicy", env, n_steps=args.n_steps, learning_rate=args.lr, ent_coef=args.ent_coef, verbose=1)

    print(f"Device: {model.policy.device}")
    if str(model.policy.device) == "cpu":
        print(
            "Running on CPU, not GPU. Check with: python3 -c \"import torch; "
            "print(torch.cuda.is_available())\" -- if that prints False, PyTorch "
            "was installed without CUDA support and needs reinstalling with a "
            "CUDA-enabled build to use your GPU."
        )

    callback = IterationCheckpointCallback(
        args.checkpoint_iterations, checkpoint_dir,
        autosave_every=args.autosave_every, start_iteration=start_iteration, verbose=1,
    )

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\nInterrupted -- saving progress before exiting...")
        path = callback.save_now(tag="latest_iter")
        print(f"Saved: {path}")
        print(f"Resume later with: --resume-from {path}")
        return

    final_path = callback.save_now(tag="final_iter")
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Checkpoints saved in: {checkpoint_dir}")


if __name__ == "__main__":
    main()
