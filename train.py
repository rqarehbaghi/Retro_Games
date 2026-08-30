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
import sys

import cv2
import numpy as np
import stable_retro as retro
from gymnasium import ObservationWrapper, Wrapper
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecNormalize,
)

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


def make_progress_reader(env, addr_lo=None, addr_hi=None, use_info_x=False):
    """A function(info) -> level position.

        position = low + (high << 8)

    `low` is either a RAM byte (addr_lo) or the integration's own on-screen x
    (use_info_x); `high` is an optional page counter (addr_hi).

    For SuperMarioBros3-Nes-v0 the answer turned out to be the integration's
    `hpos` PLUS a page byte at 0x0075. hpos alone looks broken -- it runs up to
    241 and then jumps to 3 -- but that is a wrap at 256, and 0x0075 ticked up
    on 13 of 13 forward wraps and down on 2 of 2 backward ones while changing
    only 16 times in 4971 frames. Two earlier readings of hpos were wrong: that
    it "caps at 144" (that was Mario stuck against the first obstacle, not a
    camera limit) and that the wraps were garbage (screen x cannot wrap; only a
    position can).

    With no source configured this falls back to raw info x, which is only the
    low byte and therefore resets every 256 pixels of travel -- workable for a
    game whose levels fit in one page, misleading for anything longer.
    """
    if addr_lo is None and not use_info_x:
        return read_x

    def reader(info):
        try:
            if use_info_x:
                low = read_x(info)
                if low is None:
                    return None
                low = int(low)
            else:
                low = int(env.unwrapped.get_ram()[addr_lo])
            if addr_hi is not None:
                low += int(env.unwrapped.get_ram()[addr_hi]) << 8
            return low
        except Exception:
            return None
    return reader


class AutoAdvanceLevel(Wrapper):
    """Carries training through a level clear into the next level.

    Without this, every episode starts at the integration's one default state,
    so the agent's entire experience is a single level and it flounders the
    moment it reaches another one. The alternative -- capturing a save state per
    level by hand -- works but makes the human do setup the run could do itself.

    This wraps the RAW env, below the discretizer. When a level ends it takes
    over and walks the world map internally, as extra emulator frames inside one
    step(), the same way the discretizer repeats frames for a held button. PPO
    therefore never sees the map as decisions: no map frames enter the rollout,
    no scripted actions are attributed to the policy, and the on-policy
    assumption is preserved. The agent simply finds itself in the next level.

    Map routes differ per level (1-1 exits right-then-up, 1-2 right-then-down),
    so a fixed sequence cannot work; it cycles through direction/enter
    variations until the level timer starts ticking again, which only happens
    inside a level. Gives up after max_nav_frames and lets the episode end
    normally rather than hanging.
    """

    def __init__(self, env, read_progress, read_timer, max_nav_frames=1800):
        super().__init__(env)
        self.read_progress = read_progress
        self.read_timer = read_timer
        self.max_nav_frames = max_nav_frames
        self.prev_progress = None
        self.prev_lives = None
        self.levels_advanced = 0
        buttons = env.unwrapped.buttons
        self._idx = {b: i for i, b in enumerate(buttons) if b}
        self._n = len(buttons)

    def _press(self, names, frames):
        arr = np.zeros(self._n, dtype=bool)
        for nm in names:
            if nm in self._idx:
                arr[self._idx[nm]] = True
        obs = info = None
        for _ in range(frames):
            obs, _r, term, trunc, info = self.env.step(arr)
            if term or trunc:
                return obs, info, True
        return obs, info, False

    def _navigate(self):
        """Walk the map until the timer ticks, i.e. we are in a level again."""
        # tap a direction, then press A to enter; routes vary so try several.
        routes = [["RIGHT"], ["RIGHT", "RIGHT"], ["UP"], ["DOWN"],
                  ["RIGHT", "UP"], ["RIGHT", "DOWN"], ["LEFT"]]
        obs, info, _ = self._press([], 60)      # let the clear animation finish
        spent = 60
        for route in routes:
            for direction in route:
                obs, info, done = self._press([direction], 10)
                obs, info, done = self._press([], 12)
                spent += 22
            for _ in range(3):
                obs, info, done = self._press(["A"], 10)
                obs, info, done = self._press([], 14)
                spent += 24
            before = self.read_timer(info)
            obs, info, done = self._press([], 90)
            spent += 90
            after = self.read_timer(info)
            if before is not None and after is not None and after < before:
                self.levels_advanced += 1
                return obs, info, True   # timer ticking -> in a level
            if spent >= self.max_nav_frames:
                break
        return obs, info, False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        progress = self.read_progress(info)
        lives = info.get("lives")
        if (progress is not None and self.prev_progress is not None
                and lives is not None and self.prev_lives is not None
                and progress < self.prev_progress - 200 and lives >= self.prev_lives):
            # Position collapsed with no life lost: the level ended.
            nav_obs, nav_info, ok = self._navigate()
            if ok and nav_obs is not None:
                obs, info = nav_obs, nav_info
                info["advanced_level"] = True
                progress = self.read_progress(info)
            else:
                truncated = True   # couldn't reach a level; end cleanly
        self.prev_progress = progress if progress is not None else self.prev_progress
        self.prev_lives = lives if lives is not None else self.prev_lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_progress = self.read_progress(info)
        self.prev_lives = info.get("lives")
        return obs, info


class VariableHoldDiscretizer(Wrapper):
    """Collapses stable-retro's full MultiBinary button space down to a
    small set of meaningful (combo, hold_frames) actions, and internally
    repeats the raw env.step() for however many frames THIS action calls
    for. Learning a handful of sane combos is far faster than learning
    which of 256+ raw button states matter, and giving jump its own
    short/long variants is what lets the agent express press duration at
    all -- a fixed-length repeat can never do that regardless of how long
    training runs."""

    def __init__(self, env, action_table, use_game_reward=False):
        super().__init__(env)
        # The integration's own scenario reward is OPAQUE and, for
        # SuperMarioBros3-Nes-v0, actively harmful: measured with
        # debug_rewards.py, it pays an hpos-shaped progress term that caps out
        # ~1.2s into the level (hpos pins at the 144 scroll threshold) and then
        # lands a hidden ~-124 at death -- swamping every penalty we tuned and
        # flattening the reward landscape until standing, running, and jumping
        # all score about the same. Default is to ZERO it and take full
        # ownership of the reward; pass use_game_reward=True to keep it.
        self.use_game_reward = use_game_reward
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
            if self.use_game_reward:
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
                 power_bonus=0.0, powerup_address=None, x_jump_limit=64,
                 backtrack_scale=0.5, progress_address=None, progress_address_high=None,
                 progress_use_info_x=False, playstate_address=None, playstate_value=None,
                 coin_address=None, coin_bonus=0.0,
                 speed_address=None, speed_full=127, speed_bonus=0.0,
                 clear_bonus=0.0, end_on_clear=True,
                 end_on_life_loss=True):
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
        # Coin counter and P-meter, both located by find_game_vars.py and
        # confirmed against video (see VERIFIED_ADDRESSES).
        self.coin_address = coin_address
        self.coin_bonus = coin_bonus
        self.speed_address = speed_address
        self.speed_full = speed_full
        self.speed_bonus = speed_bonus
        self.prev_score = None
        self.prev_power = None
        self.clear_bonus = clear_bonus
        self.end_on_clear = end_on_clear
        self._cleared = False
        self.prev_coins = None
        self.prev_speed = None
        # End the episode the moment a life is lost, so the vec-env auto-reset
        # returns to the clean in-level default state instead of letting the
        # emulator run on into the post-death world map / continue screen --
        # an out-of-distribution screen with no shaped reward where the policy
        # has no signal and just stalls. Requires the integration to expose
        # `lives` in info; if it doesn't, this silently has no effect (see the
        # info-keys check in the README) and the episode ends only when the
        # game's own scenario says so.
        self.end_on_life_loss = end_on_life_loss
        # Furthest point reached this episode, in accumulated x-units, so
        # progress can only be earned once per stretch of ground.
        self.travelled = 0.0
        self.max_travelled = 0.0
        self.x_jump_limit = x_jump_limit
        self.backtrack_scale = backtrack_scale
        # Game-state byte gate: most NES engines keep a mode byte that holds
        # one value during normal play and different values during the death
        # sequence / transitions (find it with probe_after_death.py). When
        # configured, any step whose state differs from the in-play value is
        # treated as 'not playing': the x-tracker is suspended (no progress,
        # no backtrack, no phantom deltas from transition garbage) and the
        # episode terminates immediately -- ending it at the moment of the
        # hit instead of ~28 decisions later when `lives` finally decrements.
        # Note: a level CLEAR also leaves the play state, so it too ends the
        # episode here (with the death penalty; the clear's score bonus
        # more than offsets it, and reaching the clear paid the whole level
        # in progress).
        self.playstate_address = playstate_address
        self.playstate_value = playstate_value
        self._playstate_armed = False
        self._read_x = make_progress_reader(env, progress_address, progress_address_high,
                                            use_info_x=progress_use_info_x)
        self.prev_x = None
        self.prev_lives = None
        self.prev_health = None

    def _read_byte(self, address):
        """One RAM byte, or None if unreadable/unset."""
        if address is None:
            return None
        try:
            return int(self.env.unwrapped.get_ram()[address])
        except Exception:
            return None

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
        self.prev_x = self._read_x(info)
        self._playstate_armed = False
        self.travelled = 0.0
        self.max_travelled = 0.0
        self.prev_lives = info.get("lives")
        self.prev_health = info.get("health")
        self.prev_score = info.get("score")
        self.prev_power = self._read_power(info)
        self._cleared = False
        self.prev_coins = self._read_byte(self.coin_address)
        self.prev_speed = self._read_byte(self.speed_address)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Per-term ledger, published in info["shaping"] each step. Costs almost
        # nothing and lets debug_rewards.py attribute every point of an
        # episode's total to a specific term -- the difference between knowing
        # WHICH term is broken and guessing (an idle policy was once observed
        # earning +0.97/decision and the total alone couldn't say from where).
        comp = {"tick": self.survival_tick - self.time_penalty, "progress": 0.0,
                "backtrack": 0.0, "score": 0.0, "power": 0.0, "life": 0.0,
                "coins": 0.0, "speed": 0.0, "clear": 0.0, "death": 0.0}

        reward += self.survival_tick
        reward -= self.time_penalty

        in_play = True
        if self.playstate_address is not None and self.playstate_value is not None:
            try:
                state = int(self.env.unwrapped.get_ram()[self.playstate_address])
                if state == self.playstate_value:
                    # Seen normal play this episode -- from here on, leaving
                    # the value means the episode is over.
                    self._playstate_armed = True
                elif self._playstate_armed:
                    in_play = False
                # Not yet armed and not the play value: level still loading at
                # episode start -- a mode enum holds other values there, and
                # terminating on it would kill every episode at birth. Wait.
            except Exception:
                pass
        if not in_play:
            # Left normal play (death sequence / transition): suspend the
            # x-tracker so garbage can't be paid, drop the baseline so
            # re-entry can't produce a giant delta, and end the episode NOW.
            self.prev_x = None
            terminated = True
            comp["death"] = -self.death_penalty
            reward -= self.death_penalty
            info["shaping"] = comp
            return obs, reward, terminated, truncated, info

        # Progress is paid ONLY for ground never reached before in this episode.
        #
        # The previous version paid max(0, x - prev_x) every step, which is
        # farmable: moving right paid, moving left cost nothing, so oscillating
        # in place -- step right, step back, step right -- is an unbounded income
        # stream that requires no actual progress. PPO reliably finds this, and
        # it looks exactly like "the agent keeps backing up and gets killed".
        # Rewarding only NEW furthest-progress removes the exploit at the source:
        # re-covering old ground pays nothing, so the only way to earn is to get
        # somewhere new.
        # Lives is needed inside the progress block to tell a level clear from a
        # death: both reset position, only one costs a life.
        lives_now = info.get("lives")
        died_now = (lives_now is not None and self.prev_lives is not None
                    and lives_now < self.prev_lives)

        current_x = self._read_x(info)
        if current_x is not None and self.prev_x is not None:
            delta = current_x - self.prev_x
            # Ignore teleport-sized jumps (screen wraps, room transitions).
            # NOTE a per-frame physics cap was tried here and reverted: the
            # death-transition garbage arrives in SMALL steps (~+8/decision as
            # the scroll byte starts running during the death sequence) that
            # pass any cap, while the camera legitimately catches up after
            # jumps in +26..36 bursts on short decisions that a physics cap
            # wrongly filters (measured: it cut +113 of real progress from a
            # RUN+JUMP probe while leaving STAND's phantom +120 untouched).
            # Transition garbage is instead excluded by the playstate gate
            # below, which knows WHEN the game is in normal play at all.
            if abs(delta) < self.x_jump_limit:
                self.travelled += delta
                if self.travelled > self.max_travelled:
                    comp["progress"] = (self.travelled - self.max_travelled) * self.progress_scale
                    reward += comp["progress"]
                    self.max_travelled = self.travelled
                elif delta < 0:
                    # Small cost for actively retreating. Not strictly needed to
                    # kill the oscillation exploit (new-ground-only already does
                    # that), but it discourages dawdling backwards and nudges the
                    # policy toward facing forward.
                    comp["backtrack"] = delta * self.progress_scale * self.backtrack_scale
                    reward += comp["backtrack"]
            elif delta < 0 and not died_now and not self._cleared and self.clear_bonus:
                # Position collapsed without losing a life: the level or area
                # ended. Because the page byte absorbs hpos's 256-wraps, normal
                # travel never produces a large backward jump -- so this is a
                # transition, not movement.
                #
                # This exists because progress alone leaves a dead spot at the
                # END of a level: there is no more ground to gain, so pushing
                # right pays nothing and hitting the goal block pays nothing
                # either. Observed exactly that -- the agent reaching the end
                # and shoving right against the edge instead of hitting the
                # card. Paying for the transition puts a gradient on finishing.
                # Once per episode, so a re-enterable pipe cannot be farmed.
                comp["clear"] = self.clear_bonus
                reward += comp["clear"]
                self._cleared = True
                if self.end_on_clear:
                    terminated = True
        if current_x is not None:
            self.prev_x = current_x
        comp["x"] = current_x

        # Points: coins, power-ups, stomped enemies, and the level-clear time
        # bonus all raise score. Clamp to >=0 so a score reset on death/new
        # level isn't read as a negative reward.
        current_score = info.get("score")
        if current_score is not None and self.prev_score is not None:
            comp["score"] = max(0, current_score - self.prev_score) * self.score_scale
            reward += comp["score"]
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
        # Coins: an explicit counter, so collection is rewarded directly rather
        # than only through the score it happens to grant. Only small positive
        # steps count -- the counter resets at 100 (paying a 1-Up, which
        # life_bonus already covers) and on level change.
        current_coins = self._read_byte(self.coin_address)
        if (self.coin_bonus and current_coins is not None
                and self.prev_coins is not None):
            delta = current_coins - self.prev_coins
            if 0 < delta <= 5:
                comp["coins"] = delta * self.coin_bonus
                reward += comp["coins"]
        if current_coins is not None:
            self.prev_coins = current_coins

        # P-meter reaching FULL. The value counts bars filled (0..6) and then
        # jumps to the full marker (127 on SMB3) when the P indicator lights, so
        # ">= speed_full" is the flight-ready test.
        # Paid on the transition only, not for holding a
        # partial meter: a full meter is the precondition for raccoon flight, and
        # filling it demands a long sustained sprint that random exploration
        # essentially never produces -- so without this the whole "reach the sky"
        # branch of the game has no gradient leading to it. Rewarding the raw
        # value instead would pay for merely jogging, which is farmable.
        current_speed = self._read_byte(self.speed_address)
        if (self.speed_bonus and current_speed is not None
                and self.prev_speed is not None):
            if current_speed >= self.speed_full and self.prev_speed < self.speed_full:
                comp["speed"] = self.speed_bonus
                reward += comp["speed"]
        if current_speed is not None:
            self.prev_speed = current_speed

        # Read lives before the power block: a power drop must be distinguished
        # from a death, where the death penalty already covers it.
        current_lives_now = info.get("lives")
        current_power = self._read_power(info)
        if current_power is not None and self.prev_power is not None and self.power_bonus:
            delta = int(current_power) - int(self.prev_power)
            if delta > 0:
                comp["power"] = delta * self.power_bonus
                reward += comp["power"]
            elif delta < 0:
                # A power DROP is only a hit if play is continuing. The byte is
                # also cleared when the level ends and the game returns to the
                # world map (observed at frame 4756 of a human demo: 1 -> 0 on
                # level exit, not a hit), and on death -- where the death
                # penalty already applies and this would double-count. Charge
                # for it only when neither is happening.
                dying = terminated or truncated or (
                    current_lives_now is not None and self.prev_lives is not None
                    and current_lives_now < self.prev_lives)
                if not dying:
                    comp["power"] = delta * self.power_bonus
                    reward += comp["power"]
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
            comp["life"] = (current_lives - self.prev_lives) * self.life_bonus
            reward += comp["life"]
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
        if ((terminated or truncated) and not info.get("is_stage_clear", False)
                and not self._cleared):
            comp["death"] = -self.death_penalty
            reward -= self.death_penalty
        elif lost_life:
            # Life lost but end_on_life_loss is off and the game didn't end the
            # episode: still penalize the death, play just continues.
            comp["death"] = -self.death_penalty
            reward -= self.death_penalty

        info["shaping"] = comp
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

    def __init__(self, discretizer_env, jump_bonus=0.2, stuck_penalty=0.05, stuck_frames=8,
                 progress_address=None, progress_address_high=None,
                 progress_use_info_x=False):
        super().__init__(discretizer_env)
        self._read_x = make_progress_reader(discretizer_env, progress_address, progress_address_high,
                                            use_info_x=progress_use_info_x)
        self.jump_action_indices = discretizer_env.jump_action_indices
        self.jump_bonus = jump_bonus
        self.stuck_penalty = stuck_penalty
        self.stuck_frames = stuck_frames
        self.prev_x = None
        self.stalled_frames = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = self._read_x(info)
        self.stalled_frames = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        comp = {"jump": 0.0, "stuck": 0.0}
        if action in self.jump_action_indices:
            comp["jump"] = self.jump_bonus
            reward += self.jump_bonus

        current_x = self._read_x(info)
        if current_x is not None and self.prev_x is not None:
            if abs(current_x - self.prev_x) < 0.5:
                self.stalled_frames += 1
                if self.stalled_frames > self.stuck_frames:
                    comp["stuck"] = -self.stuck_penalty
                    reward -= self.stuck_penalty
            else:
                self.stalled_frames = 0
        if current_x is not None:
            self.prev_x = current_x

        info["jump_shaping"] = comp
        return obs, reward, terminated, truncated, info


def make_env(game, state, death_penalty, jump_bonus, render=False, end_on_life_loss=True,
             state_file=None,
             progress_scale=0.1, keep_game_reward=False, stuck_penalty=0.1,
             backtrack_scale=0.5, survival_tick=0.0,
             score_bonus=0.01, time_penalty=0.0, life_bonus=25.0, power_bonus=0.0, powerup_address=None,
             progress_address=None, progress_address_high=None, progress_use_info_x=False,
             playstate_address=None, playstate_value=None,
             coin_address=None, coin_bonus=0.0,
             speed_address=None, speed_full=127, speed_bonus=0.0,
             clear_bonus=0.0, end_on_clear=True,
             auto_advance=False, timer_address=None):
    def _init():
        render_mode = "human" if render else "rgb_array"
        if state_file:
            # A raw emulator state captured by save_state.py, which lets
            # training start anywhere -- level 2, a mid-level checkpoint --
            # instead of only at the integration's single default state. Without
            # this every episode begins at the same spot and the agent's entire
            # experience is one level, which is why it flounders in level 2.
            import gzip
            env = retro.make(game=game, state=retro.State.NONE, render_mode=render_mode)
            with gzip.open(state_file, "rb") as fh:
                env.initial_state = fh.read()
            env.reset()
        else:
            env = retro.make(game=game, state=state or retro.State.DEFAULT, render_mode=render_mode)
        # Order matters. The discretizer must be innermost so everything
        # above it operates at the level of one agent decision. JumpIncentive
        # must wrap the discretizer directly (it reads .jump_action_indices
        # and receives the discrete action). RewardShaper then sits on top so
        # its survival tick / progress / death shaping is applied once per
        # decision, not once per emulator frame (see RewardShaper docstring).
        if auto_advance:
            # Below the discretizer, so its map navigation happens as raw
            # emulator frames inside one step and never becomes a PPO decision.
            def _prog(info):
                try:
                    if progress_use_info_x:
                        low = read_x(info)
                        if low is None:
                            return None
                        low = int(low)
                    elif progress_address is not None:
                        low = int(env.unwrapped.get_ram()[progress_address])
                    else:
                        return read_x(info)
                    if progress_address_high is not None:
                        low += int(env.unwrapped.get_ram()[progress_address_high]) << 8
                    return low
                except Exception:
                    return None

            def _timer(info):
                if timer_address is None:
                    return None
                try:
                    ram = env.unwrapped.get_ram()
                    return (int(ram[timer_address]) * 100 + int(ram[timer_address + 1]) * 10
                            + int(ram[timer_address + 2]))
                except Exception:
                    return None
            env = AutoAdvanceLevel(env, _prog, _timer)
        env = VariableHoldDiscretizer(env, ACTION_TABLE, use_game_reward=keep_game_reward)
        env = JumpIncentiveWrapper(env, jump_bonus=jump_bonus, stuck_penalty=stuck_penalty,
                                   progress_address=progress_address,
                                   progress_address_high=progress_address_high,
                                   progress_use_info_x=progress_use_info_x)
        env = RewardShaper(env, death_penalty=death_penalty, end_on_life_loss=end_on_life_loss,
                           progress_scale=progress_scale,
                           backtrack_scale=backtrack_scale, survival_tick=survival_tick,
                           score_scale=score_bonus, time_penalty=time_penalty,
                           life_bonus=life_bonus, power_bonus=power_bonus,
                           powerup_address=powerup_address,
                           progress_address=progress_address,
                           progress_address_high=progress_address_high,
                           progress_use_info_x=progress_use_info_x,
                           playstate_address=playstate_address,
                           playstate_value=playstate_value,
                           coin_address=coin_address, coin_bonus=coin_bonus,
                           speed_address=speed_address, speed_full=speed_full,
                           speed_bonus=speed_bonus,
                           clear_bonus=clear_bonus, end_on_clear=end_on_clear)
        env = WarpFrame(env)
        return env
    return _init


DEFAULT_GAME_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games.json")


def load_game_config(path, game):
    """Read one game's memory map and reward weights from games.json.

    Keeps per-game knowledge out of the code so the same pipeline trains any
    game: fill in a game's addresses there and it works, leave a variable as
    'unknown' and the shaping that needs it stays inert rather than reading
    garbage. Returns argparse defaults; command-line flags still override.
    """
    if not path or not os.path.exists(path):
        return {}, {}
    try:
        import json
        with open(path) as fh:
            entry = (json.load(fh).get("games") or {}).get(game)
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return {}, {}
    if not entry:
        return {}, {}

    variables = entry.get("variables") or {}
    defaults = dict(entry.get("rewards") or {})

    def addr(name):
        spec = variables.get(name) or {}
        if spec.get("source") in (None, "unknown") or spec.get("address") is None:
            return None
        return int(spec["address"], 0) if isinstance(spec["address"], str) else int(spec["address"])

    mapped = {
        "progress_address": addr("progress"),
        "powerup_address": addr("powerup"),
        "playstate_address": addr("playstate"),
        "coin_address": addr("coins"),
        "speed_address": addr("pmeter"),
    }
    progress = variables.get("progress") or {}
    if progress.get("address_high") is not None:
        mapped["progress_address_high"] = int(str(progress["address_high"]), 0)
    if progress.get("source") == "info16":
        # Low byte from the integration's info key, high byte from RAM.
        mapped["progress_use_info_x"] = True
        if progress.get("address_high") is not None:
            mapped["progress_address_high"] = int(str(progress["address_high"]), 0)
    playstate = variables.get("playstate") or {}
    if playstate.get("in_play_value") is not None:
        mapped["playstate_value"] = int(str(playstate["in_play_value"]), 0)
    meter = variables.get("pmeter") or {}
    if meter.get("full_value") is not None:
        mapped["speed_full"] = int(str(meter["full_value"]), 0)

    defaults.update({k: v for k, v in mapped.items() if v is not None})
    return defaults, variables


# Addresses VERIFIED against video for a game, with how they were confirmed.
# Recorded here so the knowledge lives in the repo rather than in a chat log.
VERIFIED_ADDRESSES = {
    "SuperMarioBros3-Nes-v0": {
        "powerup": (0x00ED, "0=small 1=big 2=fire 3=raccoon; tier flips exactly "
                            "when a mushroom is taken on the recorded video"),
        "coins":   (0x2167, "live coin counter. 0x25A2 mirrors it for the HUD but "
                            "lags: --compare showed 3 divergences with 0x2167 first"),
        "progress_page": (0x0075, "HIGH byte of level position; the low byte is the "
                            "integration's own hpos. Ticked up on 13/13 forward wraps of "
                            "hpos and down on 2/2 backward ones, changing only 16 times "
                            "in 4971 frames. Range 0..32 pages == ~8192px. "
                            "position = hpos + (0x0075 << 8)"),
        "speed":   (0x03DD, "P-meter. Value == number of bars filled (0..6), then "
                            "jumps to 127 when the P indicator lights == full == "
                            "raccoon flight available. Found by correlating with "
                            "sustained run+direction input, confirmed on screen"),
        "timer":   (0x05EE, "3-digit BCD across 0x05EE/0x05EF/0x05F0: value = "
                            "*100 + *10 + *1. Range exactly 0..299 with ZERO "
                            "increases over a whole demo. The integration's `time` "
                            "info key reads something else and is unusable"),
        # progress + playstate: still unknown, see KNOWN_BAD_ADDRESSES.
    },
}


# Addresses proven WRONG for a game, kept so a stale command line can't
# silently poison a run. Each was believed correct at some point and cost real
# training time; the audit that disproved it is named in the reason.
KNOWN_BAD_ADDRESSES = {
    ("SuperMarioBros3-Nes-v0", "progress", 0x00CF):
        "returns to 0 whenever the player stops or reverses -- it is a per-frame "
        "delta/velocity, not accumulated position (audit_ram.py on a human demo: "
        "progress jumped 24 -> 261 -> 284 -> 394 -> 239). It only looked monotonic "
        "under a scripted RIGHT-hold.",
    ("SuperMarioBros3-Nes-v0", "progress", 0x053C):
        "a world-map counter, frozen during actual play -- it was picked from a "
        "RAM scan whose window was two-thirds post-death map frames.",
    ("SuperMarioBros3-Nes-v0", "playstate", 0x07F1):
        "cycles 5 -> 0 -> 8 every ~24 frames DURING normal play (animation/PPU "
        "phase), so it cannot mark 'in play'; gating on it ends every episode "
        "within about a second.",
    ("SuperMarioBros3-Nes-v0", "playstate", 0x0749):
        "flips only ~19 frames before lives decrements -- the END of the death "
        "sequence, so the whole garbage window is still paid as progress.",
}


def reject_known_bad(game, kind, address):
    """Refuse an address already disproven for this game."""
    if address is None:
        return
    reason = KNOWN_BAD_ADDRESSES.get((game, kind, address))
    if reason:
        print(
            f"\nFATAL: --{kind}-address 0x{address:04X} is a KNOWN-BAD address for "
            f"{game}.\n  {reason}\n"
            f"  Training with it wastes the run. Find a working address with "
            f"inspect_progress.py --demo\n  (progress) or probe_after_death.py "
            f"(playstate), and verify with audit_ram.py."
        )
        sys.exit(1)


def describe_value_sources(game, progress_address=None, progress_address_high=None,
                           powerup_address=None, playstate_address=None,
                           playstate_value=None, progress_use_info_x=False,
                           coin_address=None, speed_address=None, speed_full=127):
    """Print exactly where every shaped value is read from.

    info-key addresses come from the integration's own data.json, so this shows
    the real RAM address behind `lives`/`score`/`time`/`hpos` rather than just
    the key name."""
    print("\nVALUE SOURCES")
    integration_addrs = {}
    try:
        import json
        rom = retro.data.get_romfile_path(game, retro.data.Integrations.STABLE)
        with open(os.path.join(os.path.dirname(rom), "data.json")) as fh:
            for name, spec in (json.load(fh).get("info") or {}).items():
                integration_addrs[name] = spec
    except Exception:
        pass

    def info_src(key):
        spec = integration_addrs.get(key) or {}
        addr = spec.get("address")
        if addr is None:
            return f"info['{key}'] (address not readable from data.json)"
        return f"RAM 0x{addr:04X} ({addr}) via info['{key}'], type {spec.get('type', '?')}"

    print(f"  lives     <- {info_src('lives')}")
    print(f"  score     <- {info_src('score')}   [NOTE: one TENTH of the on-screen value]")
    print(f"  time      <- {info_src('time')}")
    print(f"  hpos      <- {info_src('hpos')}   [on-screen x; caps at the scroll threshold]")

    # Report what the reader will ACTUALLY do. Checking only progress_address
    # was wrong: a 16-bit position whose low byte comes from the info key
    # legitimately leaves that field None, and the old text then announced a
    # broken fallback for a perfectly good configuration.
    if progress_use_info_x or progress_address is not None:
        low = "info x/hpos" if progress_use_info_x else f"RAM 0x{progress_address:04X}"
        src = low
        if progress_address_high is not None:
            src = f"{low} + (RAM 0x{progress_address_high:04X} << 8)   [16-bit level position]"
        else:
            src += "   *** LOW BYTE ONLY -- wraps every 256px of travel ***"
        print(f"  PROGRESS  <- {src}")
    else:
        print("  PROGRESS  <- info x/hpos fallback  *** NO progress source set ***")
        print("               Whatever the integration calls x, used raw. If that is")
        print("               an on-screen coordinate or a low byte, progress stops")
        print("               paying partway through a level and training plateaus.")
        print("               Find a source: inspect_progress.py --demo <recording>")

    print(f"  powerup   <- " + (f"RAM 0x{powerup_address:04X} ({powerup_address})  [0=small 1=big 2=fire 3=raccoon]"
                                if powerup_address is not None else "(not set -- power shaping inactive)"))
    print(f"  playstate <- " + (f"RAM 0x{playstate_address:04X} ({playstate_address}), in-play value {playstate_value}"
                                if playstate_address is not None else "(not set -- no death-sequence gating)"))
    print(f"  coins     <- " + (f"RAM 0x{coin_address:04X} ({coin_address})"
                                if coin_address is not None else "(not set -- coins rewarded only via score)"))
    print(f"  P-meter   <- " + (f"RAM 0x{speed_address:04X} ({speed_address}), full at {speed_full}"
                                if speed_address is not None else "(not set -- no signal toward raccoon flight)"))
    print()


def action_index(combo, hold=None):
    """Index of an ACTION_TABLE entry by combo (and optionally hold length)."""
    for i, (c, h) in enumerate(ACTION_TABLE):
        if set(c) == set(combo) and (hold is None or h == hold):
            return i
    return None


def reward_sanity_check(env_fn):
    """Run scripted probes through the EXACT env about to be trained on and
    refuse to train if the reward landscape is broken.

    This exists because every major failure in this project's history --
    the dead x-key, the on-screen-x cap, the oscillation farm, the false
    0x053C address, the hidden scenario -124, the jump-bonus farm -- was
    invisible until behavior had already converged on it, and the standalone
    gate (debug_rewards.py) kept getting skipped between config changes.
    Running it automatically at startup costs ~half a minute and makes
    training on a silently-broken signal impossible.

    Pass condition: RUN must decisively out-earn STAND (the progress stream
    is alive and idling loses). JUMP-LEFT approximating STAND guards against
    per-action bonuses being farmable at the left screen edge."""
    probes = (
        ("STAND", lambda i: action_index([])),
        ("RUN", lambda i: action_index(["RIGHT", "B"])),
        ("JUMP-LEFT", lambda i: action_index(["LEFT", "A"], hold=20)),
    )
    print("Reward sanity check (scripted probes through the exact training env)...")
    env = env_fn()
    results = {}
    rates = {}
    for label, pick in probes:
        env.reset()
        total, n = 0.0, 0
        for i in range(400):
            _obs, r, terminated, truncated, _info = env.step(pick(i))
            total += r
            n += 1
            if terminated or truncated:
                break
        results[label] = total
        rates[label] = total / max(1, n)
        print(f"  {label:<10} {total:+9.2f} over {n:4d} decisions  "
              f"({rates[label]:+.2f}/decision)")
    env.close()

    ok = True
    if results["STAND"] > 10:
        print(
            "\nFATAL: STAND earns a clearly positive total -- something pays for\n"
            "doing nothing (observed once: ~+1/decision of phantom income while\n"
            "idle). An agent with a profitable do-nothing niche will find it.\n"
            "Run debug_rewards.py with these flags: its per-component breakdown\n"
            "names the term. Override with --skip-reward-check."
        )
        ok = False
    if results["RUN"] <= results["STAND"] + 20:
        print(
            "\nFATAL: RUN does not decisively out-earn STAND in the actual training\n"
            "env -- the progress signal is dead or drowned, and training now would\n"
            "converge to idling/degenerate behavior. Check --progress-address /\n"
            "--progress-use-info-x / --progress-address-high (verify with\n"
            "inspect_progress.py --demo) and\n"
            "the rest of the shaping flags. Diagnose with debug_rewards.py using\n"
            "the same flags. Override with --skip-reward-check if you are certain."
        )
        ok = False
    # Compare PER-DECISION, not totals. These policies run for very different
    # numbers of decisions, and one that dies sooner accrues less stuck-penalty
    # bleed -- so on totals alone jump-spam can appear to "beat" standing while
    # earning nothing at all (measured: JUMP-LEFT -28.4 over 27 decisions vs
    # STAND -48.6 over 128, yet -1.05/decision against -0.38, i.e. strictly
    # worse). Rate isolates farming from merely ending the episode early.
    if rates["JUMP-LEFT"] > rates["STAND"] + 0.05:
        print(
            "\nFATAL: JUMP-LEFT earns more PER DECISION than STAND -- some\n"
            "per-action bonus is being farmed by jumping in place at the left\n"
            "screen edge (this exact failure was observed in training). Check\n"
            "--jump-bonus is 0 and that no term pays for motionless actions.\n"
            "Override with --skip-reward-check."
        )
        ok = False
    if ok:
        print("  -> landscape OK: progress pays, idling and jump-spam lose.\n")
    return ok


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

    def __init__(self, milestones, out_dir, autosave_every=25, start_iteration=0, verbose=0,
                 vecnorm=None):
        super().__init__(verbose)
        self.milestones = set(milestones)
        self.out_dir = out_dir
        self.autosave_every = autosave_every
        self.iteration = start_iteration
        self._last_autosave_path = None
        # The VecNormalize layer, so its running reward statistics get saved
        # with every checkpoint -- they're training state: resuming without
        # them re-estimates from scratch and briefly mis-scales rewards.
        self.vecnorm = vecnorm
        os.makedirs(out_dir, exist_ok=True)

    def save_now(self, tag="iter"):
        path = os.path.join(self.out_dir, f"{tag}_{self.iteration}.zip")
        self.model.save(path)
        if self.vecnorm is not None:
            self.vecnorm.save(os.path.join(self.out_dir, "vecnorm_stats.pkl"))
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
    parser.add_argument("--state", default=None, help="Named save state from the integration to start from (default state if omitted)")
    parser.add_argument("--state-file", default=None, help="Path to a raw emulator state captured by save_state.py, used instead of a named state. This is how you train on a level other than the one the integration starts at -- without it every episode begins at the same place and the agent only ever learns that level.")
    parser.add_argument("--iterations", type=int, default=100, help="How many MORE iterations to run this invocation -- the main training-length knob. (default: %(default)s)")
    parser.add_argument("--num-envs", type=int, default=8, help="Parallel emulator instances -- match roughly to your CPU core count. (default: %(default)s)")
    parser.add_argument("--n-steps", type=int, default=128, help="Env steps collected per env before each PPO update. (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="PPO learning rate. Lower this (e.g. 3e-5) when resuming from an imitation-pretrained checkpoint, so RL fine-tuning doesn't wash out what it already learned. (default: %(default)s)")
    parser.add_argument("--n-epochs", type=int, default=4, help="Passes over each rollout batch per update. SB3's default is 10, which with n_steps 128 x 8 envs means ~160 gradient steps on 1024 samples -- enough reuse to push approx_kl and clip_fraction far past healthy (0.15 and 0.66 were observed). Atari PPO conventionally uses 4. (default: %(default)s)")
    parser.add_argument("--target-kl", type=float, default=0.03, help="Stop a rollout's epoch loop early once the policy has moved this far in KL. This is the direct guard against the aggressive updates that precede a training collapse; set to 0 to disable. (default: %(default)s)")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="PPO entropy coefficient -- higher encourages more exploration. Lower this (e.g. 0.001) when fine-tuning a pretrained checkpoint. (default: %(default)s)")
    parser.add_argument("--death-penalty", type=float, default=50.0, help="Reward subtracted on death/episode-end without clearing the stage. (default: %(default)s)")
    parser.add_argument("--jump-bonus", type=float, default=0.0, help="Reward added for choosing a jump action. DEFAULT 0 -- and leave it there once a real progress signal is configured: paying for the ACT of jumping is farmable (observed in training: the policy converged to jumping in place at the left screen edge, where jump bonus minus stuck penalty is risk-free income and nothing else pays). Clearing an obstacle already pays through the progress it unlocks. Only raise this as a temporary crutch on a game with NO working progress signal. (default: %(default)s)")
    parser.add_argument("--stuck-penalty", type=float, default=0.1, help="Per-decision penalty while horizontal position hasn't changed for several decisions -- bleed applied to standing still or running against an obstacle. Must exceed any per-decision bonus (e.g. --jump-bonus) or stalling somewhere and farming that bonus becomes net-positive. (default: %(default)s)")
    parser.add_argument("--score-bonus", type=float, default=0.01, help="Reward per point the game's own score goes up -- coins, power-ups, stomped enemies, and the level-clear leftover-time bonus. This is what makes the agent value points AND finishing fast (faster clear = more time converted to score). Raise it to prioritize collecting/points, lower toward 0 for a pure speedrun-right agent. Tune to your game's score magnitudes. (default: %(default)s)")
    parser.add_argument("--life-bonus", type=float, default=25.0, help="Reward for each extra life GAINED -- 1-Up mushrooms, the 100-coin threshold, score milestones. An extra life is worth far more strategically than the few points a 1-Up adds to score, so --score-bonus alone undervalues it. Set to 0 to disable. Note regular power-up mushrooms can't be rewarded directly (this integration exposes no power-state variable), only via their score. (default: %(default)s)")
    parser.add_argument("--power-bonus", type=float, default=0.0, help="Reward per power-up TIER gained (small->big->fire->raccoon...), and the same penalty per tier lost when you take a hit. Requires the integration to publish the power state in info as `powerup` -- it is NOT exposed by default. Use find_ram_variable.py to locate the RAM address in your own recording, add it to the integration data.json, then set this (try 10-20). Left at 0 it does nothing. (default: %(default)s)")
    parser.add_argument("--backtrack-scale", type=float, default=0.5, help="Fraction of --progress-scale charged for moving backwards over ground already covered. 0 disables the backtrack cost. (default: %(default)s)")
    parser.add_argument("--survival-tick", type=float, default=0.0, help="Flat reward added every decision for still being alive. Default 0: rewarding survival fights a speed objective, and the death penalty already discourages dying. (default: %(default)s)")
    parser.add_argument("--progress-scale", type=float, default=0.1, help="Reward per unit of NEW ground covered. This is the main learning signal and must be balanced against --death-penalty: if a whole run only earns progress*distance while dying costs far more, the death term swamps everything and the agent gets no gradient toward playing better. Measure your game's distance-per-run with inspect_progress.py and scale so a good run is worth at least as much as a death costs. (default: %(default)s)")
    parser.add_argument("--keep-game-reward", action="store_true", help="Keep the integration's own scenario reward on top of the shaping. OFF by default: for SuperMarioBros3-Nes-v0 that reward was measured (debug_rewards.py) to pay an hpos-based progress term that caps out ~1.2s into the level and then a hidden ~-124 at death, swamping the tuned shaping and flattening the reward landscape.")
    parser.add_argument("--progress-address", type=lambda v: int(v, 0), default=None, help="RAM index of the real level-position counter, read directly. STRONGLY recommended: SuperMarioBros3-Nes-v0's published `hpos` is Mario's ON-SCREEN x, which flatlines at the scroll threshold (144), so rewarding it pays only for the first ~1.5s of a level. Find candidates with inspect_progress.py and VERIFY with its --watch flag before trusting one -- an early candidate (0x053C) turned out to be a map-screen counter that never moves during play, because the scan window included post-death map frames. Hex or decimal. (default: %(default)s)")
    parser.add_argument("--progress-use-info-x", action="store_true", help="Take the LOW byte of level position from the integration's own on-screen x (hpos) rather than a RAM address, and combine it with --progress-address-high. This is the SMB3 answer: hpos wraps at 256 and the page byte 0x0075 ticks up on every wrap, so position = hpos + (0x0075 << 8). Replaces the old --progress-add-screen-x, whose additive theory was wrong.")
    parser.add_argument("--progress-address-high", type=lambda v: int(v, 0), default=None, help="High byte of a 16-bit little-endian level position (e.g. 0x053D), combined with --progress-address. A single byte wraps at 255, which a full level exceeds several times. (default: %(default)s)")
    parser.add_argument("--auto-advance", action="store_true", help="On clearing a level, walk the world map automatically and keep training in the NEXT level, instead of ending the episode. The navigation runs as raw emulator frames inside a single step, so no map frames enter the rollout and no scripted action is attributed to the policy. This is what lets one run cover several levels without capturing a save state per level by hand. Needs --timer-address to tell when a level has loaded. Implies --no-end-on-clear.")
    parser.add_argument("--timer-address", type=lambda v: int(v, 0), default=None, help="RAM index of the 3-digit BCD level timer (SMB3: 0x05EE). Used by --auto-advance to detect that a level has actually started, since the timer only ticks in a level.")
    parser.add_argument("--clear-bonus", type=float, default=0.0, help="Reward for finishing a level or entering a new area, detected as level position collapsing WITHOUT losing a life (normal travel never jumps backwards, because the page byte absorbs the low byte's wraps). This fills the dead spot at the end of a level: progress pays for ground gained, so at the goal there is nothing left to earn and the agent shoves right against the edge instead of hitting the goal block. Paid once per episode so a re-enterable pipe cannot be farmed. (default: %(default)s)")
    parser.add_argument("--no-end-on-clear", dest="end_on_clear", action="store_false", help="Keep playing after a level clear instead of ending the episode there. Off by default because the agent has never trained on the world map and only flails there.")
    parser.set_defaults(end_on_clear=True)
    parser.add_argument("--coin-address", type=lambda v: int(v, 0), default=None, help="RAM index of the coin counter, so collecting coins is rewarded directly instead of only through the score it grants. SMB3 verified: 0x2167 (0x25A2 is the HUD mirror and lags a frame). Needs --coin-bonus.")
    parser.add_argument("--coin-bonus", type=float, default=0.0, help="Reward per coin collected. (default: %(default)s)")
    parser.add_argument("--speed-address", type=lambda v: int(v, 0), default=None, help="RAM index of the speed/P-meter. SMB3 verified: 0x03DD, 0..127 where 127 (0b1111111, one bit per segment: 6 arrows + P) means FULL -- the precondition for raccoon flight. Needs --speed-bonus.")
    parser.add_argument("--speed-full", type=lambda v: int(v, 0), default=127, help="Value of --speed-address meaning a full meter. (default: %(default)s)")
    parser.add_argument("--speed-bonus", type=float, default=0.0, help="Reward when the meter REACHES full (paid on the transition, not for holding it). Filling it needs a long sustained sprint that random exploration almost never produces, so without this there is no gradient toward flight at all. Try 20-40. (default: %(default)s)")
    parser.add_argument("--powerup-address", type=lambda v: int(v, 0), default=None, help="RAM index holding the power-up tier, read directly so the integration does not need to publish it (SuperMarioBros3-Nes-v0: 0x00ED, found with find_ram_variable.py -- verify for your own game/version). Accepts hex (0x00ED) or decimal. Needed for --power-bonus to do anything here. (default: %(default)s)")
    parser.add_argument("--time-penalty", type=float, default=0.0, help="Small reward subtracted every decision, to discourage dawdling and push toward finishing the level sooner. Start around 0.01-0.05 if the agent loiters; too high and it rushes into danger. (default: %(default)s)")
    parser.add_argument("--playstate-address", type=lambda v: int(v, 0), default=None, help="RAM index of the game-state/mode byte (find it with probe_after_death.py's state-byte differ). With --playstate-value set, any step where this byte differs from the in-play value is treated as 'not playing': the x-tracker is suspended so death-transition garbage can never be paid as progress, and the episode ends at the moment of the hit instead of when lives finally decrements. Hex or decimal.")
    parser.add_argument("--playstate-value", type=lambda v: int(v, 0), default=None, help="The value the --playstate-address byte holds during NORMAL play.")
    parser.add_argument("--skip-reward-check", action="store_true", help="Skip the automatic startup reward sanity check (scripted STAND/RUN/JUMP-LEFT probes through the exact training env, refusing to train on a broken landscape). Only skip when deliberately experimenting with a configuration the check would reject.")
    parser.add_argument("--no-end-on-death", dest="end_on_life_loss", action="store_false", help="By default the episode ends the instant a life is lost, so training always restarts from the clean in-level state instead of wandering onto the post-death world map / continue screen. Pass this to disable that and let the game's own scenario decide when an episode ends. (Requires the integration to expose 'lives' in info either way.)")
    parser.set_defaults(end_on_life_loss=True)
    parser.add_argument("--checkpoint-iterations", type=int, nargs="+", default=None, help="Which cumulative iterations to snapshot as named milestones. Defaults to [1, <the iteration this run ends on>] if omitted.")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Parent folder -- a subfolder named after --game is created inside it. (default: %(default)s)")
    parser.add_argument("--autosave-every", type=int, default=25, help="Also save a rolling latest_iter_N.zip every N iterations, regardless of --checkpoint-iterations -- your crash/resume safety net. Set to 0 to disable. (default: %(default)s)")
    parser.add_argument("--render", action="store_true", help="Open a live emulator window for ONE of the parallel envs so you can watch training happen. Needs a display (see README) and slows training down. To watch cleanly WITHOUT the slowdown, train headless and periodically play a checkpoint with: play_and_record.py --model <ckpt> --render")
    parser.add_argument("--resume-from", default=None, help="Path to a checkpoint to continue training from -- a previous PPO run, OR an imitation-pretrained checkpoint from pretrain_imitation.py. The cumulative iteration count is read from its filename.")
    parser.add_argument("--start-iteration", type=int, default=None, help="Override the cumulative iteration count when resuming, if the checkpoint filename doesn't encode it (e.g. you renamed it).")
    parser.add_argument("--game-config", default=DEFAULT_GAME_CONFIG, help="JSON file of per-game memory maps and reward weights (default: games.json next to this script). Its values become the defaults for this game; any flag you pass still overrides them. Set to '' to ignore the file entirely.")

    # Two-pass parse: read --game/--game-config first, fold that game's entry in
    # as defaults, then parse for real so explicit flags win over the file.
    prelim, _ = parser.parse_known_args()
    config_defaults, config_vars = load_game_config(prelim.game_config, prelim.game)
    if config_defaults:
        known = {a.dest for a in parser._actions}
        applied = {k: v for k, v in config_defaults.items() if k in known}
        ignored = sorted(set(config_defaults) - set(applied))
        parser.set_defaults(**applied)
        print(f"Loaded {len(applied)} defaults for {prelim.game} from {prelim.game_config}")
        if ignored:
            # A weight in the config with no matching flag would otherwise be
            # dropped in silence -- the config would claim a setting training
            # never used. Two were found this way (backtrack_scale,
            # survival_tick), so say it out loud.
            print(f"  WARNING: {len(ignored)} config key(s) have no matching flag and were "
                  f"IGNORED: {', '.join(ignored)}")
            print(f"  Training is NOT using them. Add a flag in train.py or remove them "
                  f"from {prelim.game_config}.")
    elif prelim.game_config:
        print(f"No entry for {prelim.game} in {prelim.game_config} -- using built-in defaults. "
              f"Add one (copy the _template entry) to make this game reproducible.")

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
    print(f"Reward shaping: keep_game_reward={args.keep_game_reward}, death_penalty={args.death_penalty}, progress_scale={args.progress_scale}, progress_use_info_x={args.progress_use_info_x}, progress_address_high={args.progress_address_high}, jump_bonus={args.jump_bonus}, score_bonus={args.score_bonus}, life_bonus={args.life_bonus}, power_bonus={args.power_bonus}, powerup_address={args.powerup_address}, progress_address={args.progress_address}, time_penalty={args.time_penalty}, end_on_life_loss={args.end_on_life_loss}")
    print(
        "Rough guide from earlier: simple games often reach solid play in "
        "hours, medium-complexity platformers in about a day, on a modern "
        "multi-core machine with several parallel envs. This is a guide, "
        "not a promise -- actual time depends heavily on the game."
    )

    # Refuse addresses already disproven for this game -- a stale command line
    # from shell history must not silently poison another run.
    if args.auto_advance:
        if args.timer_address is None:
            print("--auto-advance needs --timer-address (SMB3: 0x05EE) to tell when a "
                  "level has loaded; without it the navigator cannot know it succeeded.")
            sys.exit(1)
        # The episode must not end at the clear if we intend to play on.
        args.end_on_clear = False
    reject_known_bad(args.game, "progress", args.progress_address)
    reject_known_bad(args.game, "playstate", args.playstate_address)
    describe_value_sources(
        args.game,
        progress_address=args.progress_address,
        progress_address_high=args.progress_address_high,
        powerup_address=args.powerup_address,
        playstate_address=args.playstate_address,
        playstate_value=args.playstate_value,
        progress_use_info_x=args.progress_use_info_x,
        coin_address=args.coin_address,
        speed_address=args.speed_address,
        speed_full=args.speed_full,
    )

    # One kwargs dict shared by the sanity-check probe env and every training
    # env, so the check exercises EXACTLY the configuration training will use
    # (duplicated kwarg lists have already caused silent drift once).
    env_kwargs = dict(
        state_file=args.state_file,
        end_on_life_loss=args.end_on_life_loss,
        score_bonus=args.score_bonus, time_penalty=args.time_penalty,
        life_bonus=args.life_bonus, power_bonus=args.power_bonus,
        powerup_address=args.powerup_address,
        progress_scale=args.progress_scale,
        keep_game_reward=args.keep_game_reward,
        stuck_penalty=args.stuck_penalty,
        backtrack_scale=args.backtrack_scale, survival_tick=args.survival_tick,
        progress_address=args.progress_address,
        progress_address_high=args.progress_address_high,
        progress_use_info_x=args.progress_use_info_x,
        playstate_address=args.playstate_address,
        playstate_value=args.playstate_value,
        coin_address=args.coin_address, coin_bonus=args.coin_bonus,
        speed_address=args.speed_address, speed_full=args.speed_full,
        speed_bonus=args.speed_bonus,
        clear_bonus=args.clear_bonus, end_on_clear=args.end_on_clear,
        auto_advance=args.auto_advance, timer_address=args.timer_address,
    )

    if not args.skip_reward_check:
        if not reward_sanity_check(make_env(args.game, args.state, args.death_penalty,
                                            args.jump_bonus, **env_kwargs)):
            sys.exit(1)

    # --render runs the single env in-process (DummyVecEnv) so its window lives
    # in the main process and stays responsive. Headless training keeps using
    # SubprocVecEnv for true parallelism across --num-envs processes.
    if args.render:
        env = DummyVecEnv([
            make_env(args.game, args.state, args.death_penalty, args.jump_bonus,
                     render=True, **env_kwargs)
        ])
    else:
        env = SubprocVecEnv([
            make_env(args.game, args.state, args.death_penalty, args.jump_bonus,
                     **env_kwargs)
            for _ in range(args.num_envs)
        ])
    # Without a Monitor layer SB3 has no episode statistics at all --
    # rollout/ep_rew_mean and ep_len_mean simply never appear in the console
    # table, leaving no way to tell whether training is working. VecMonitor
    # records per-episode shaped reward and length at the vec-env level.
    # (It sits BELOW VecNormalize, so the logged ep_rew_mean stays in raw,
    # interpretable shaped-reward units, not normalized ones.)
    env = VecMonitor(env)

    # Reward normalization -- the stabilizer this pipeline was missing.
    # With a real progress signal (e.g. --progress-scale 1.0 against a level
    # position counter) episode returns reach magnitude ~100+. Feeding raw
    # returns that large into PPO makes the value-function loss dominate the
    # shared CNN trunk; training then looks healthy early (the agent learns to
    # run) and later COLLAPSES into a degenerate policy (e.g. standing still
    # until an enemy arrives) as the value blowup wrecks the features both
    # heads share. VecNormalize rescales rewards to O(1) by a running estimate
    # of return variance, which is the standard fix. Observations are left
    # alone (uint8 images; the policy normalizes those itself). The running
    # statistics are part of the training state, so they are saved next to
    # every checkpoint and restored on --resume-from.
    vecnorm_stats = os.path.join(checkpoint_dir, "vecnorm_stats.pkl")
    if args.resume_from and os.path.exists(vecnorm_stats):
        env = VecNormalize.load(vecnorm_stats, env)
        env.training = True
        print(f"Restored reward-normalization statistics from {vecnorm_stats}")
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    vecnorm = env
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
                "n_epochs": args.n_epochs,
                "target_kl": (args.target_kl or None),
                # PPO.load restores `verbose` too, and pretrain_imitation.py
                # builds its model with verbose=0 -- so resuming from a BC
                # checkpoint silenced SB3's logger completely: no rollout
                # table, no ep_rew_mean/ep_len_mean, exactly when they're
                # needed to see whether the warm start took. Force it on.
                "verbose": 1,
            },
        )
        # ep_info_buffer is restored as well, so a resume would spend its first
        # ~100 episodes averaging in the PREVIOUS run's stats -- misleading
        # precisely when comparing a warm start against where the last run
        # plateaued. Start the episode statistics fresh for this run.
        model.ep_info_buffer = None
        model.ep_success_buffer = None
    else:
        model = PPO("CnnPolicy", env, n_steps=args.n_steps, learning_rate=args.lr,
                    ent_coef=args.ent_coef, n_epochs=args.n_epochs,
                    target_kl=(args.target_kl or None), verbose=1)

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
        vecnorm=vecnorm,
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
