#!/usr/bin/env python3
"""
A training environment for TetrisTime that reads its state from games.json.

Two things make this different from train.py's SMB3 setup.

FIRST, it does not use retro's data.json at all. The integration for a ROM
imported by hand has an empty data.json, so env.step() returns info={} and a
reward of 0 forever -- there is nothing to train on. Everything here is read
straight out of env.get_ram() using the addresses in games.json, which are the
ones that were verified frame-by-frame against the screen. games.json is the
single source of truth; adding a game means adding its addresses there, not
editing retro's integration files.

SECOND, the agent is PLAYER 2 in a two-player game. A policy trained on the
one-player screen would have to generalise to the two-player screen at deploy
time -- different layout, two wells, a different HUD -- which is exactly the
kind of domain shift that quietly ruins a run. Training as P2 from a two-player
save state removes the problem instead of managing it: the agent sees the
screen it will play on and drives the controls it will drive. Player 1 is left
idle by default; its play does not affect P2's reward, which is read only from
P2's own addresses.

The joint action is MultiBinary(9 * players); P2 occupies bits 9..17. Verified:
driving only that half moves only the right side of the screen.

Usage:
    from tetris_env import make_tetris_env
    env = make_tetris_env()          # P2, level 0, two players
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
import json
import os

import gymnasium as gym
import numpy as np
import stable_retro as retro

import custom_integrations

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES_JSON = os.path.join(HERE, "games.json")

# The control map, on one player's 9-button slice. Tetris needs nothing else:
# no jump, no run. Holding a rotate button does not spin the piece repeatedly --
# the game debounces it -- so a frame-skip that repeats the press is safe.
TETRIS_ACTIONS = [
    [],             # do nothing (let the piece fall)
    ["LEFT"],       # move left
    ["RIGHT"],      # move right
    ["DOWN"],       # soft drop
    ["A"],          # rotate clockwise
    ["B"],          # rotate counter-clockwise
]


class GameVars:
    """Read the verified values for one game out of games.json.

    Handles both shapes games.json records: `tiles` (a run of display digits,
    one per byte, byte = digit + digit_offset, most significant first) and
    `ram` (a plain byte). Nothing here knows about a specific game -- point it
    at another entry and it reads that one.
    """

    def __init__(self, game, sections=("two_player", "variables")):
        with open(GAMES_JSON, encoding="utf-8") as fh:
            entry = (json.load(fh).get("games") or {}).get(game) or {}
        self.spec = {}
        for section in sections:
            for name, s in (entry.get(section) or {}).items():
                if isinstance(s, dict) and s.get("address"):
                    self.spec[name] = s
        self.training = entry.get("training") or {}

    def has(self, name):
        return name in self.spec

    def read(self, ram, name, default=None):
        s = self.spec.get(name)
        if not s:
            return default
        base = int(str(s["address"]), 0)
        if s.get("source") == "tiles":
            off = int(s.get("digit_offset", 48))
            digits = []
            for i in range(int(s.get("length", 1))):
                d = int(ram[base + i]) - off
                # A blank or garbage tile reads as 0 rather than poisoning the
                # whole number; the HUD pads with '0' tiles anyway.
                digits.append(d if 0 <= d <= 9 else 0)
            return int("".join(str(d) for d in digits))
        return int(ram[base])


class TetrisPlayerEnv(gym.Env):
    """One player of a two-player TetrisTime game, as a Gym environment."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, game="TetrisTime-Nes-v0", player=2, state=None,
                 players=2, frameskip=4, opponent=None,
                 line_reward=(0.0, 1.0, 3.0, 5.0, 8.0), score_scale=0.01,
                 death_penalty=-10.0, step_reward=0.001, max_steps=100_000):
        super().__init__()
        custom_integrations.register()
        self.vars = GameVars(game)
        if state is None:
            state = self.vars.training.get("start_state_2p") or retro.State.DEFAULT
        self.env = retro.make(game=game, state=state, players=players,
                              use_restricted_actions=retro.Actions.ALL,
                              render_mode="rgb_array")
        self.buttons = self.env.unwrapped.buttons
        self.n_buttons = self.env.num_buttons
        self.players = players
        self.slot = player - 1              # P2 -> bits 9..17
        self.frameskip = frameskip
        self.opponent = opponent
        self.line_reward = line_reward
        self.score_scale = score_scale
        self.death_penalty = death_penalty
        self.step_reward = step_reward
        self.max_steps = max_steps

        self._idx = {n: i for i, n in enumerate(self.buttons) if n}
        self._combos = [[self._idx[b] for b in combo] for combo in TETRIS_ACTIONS]
        self.action_space = gym.spaces.Discrete(len(TETRIS_ACTIONS))
        self.observation_space = self.env.observation_space

        p = player
        self._k_score = "score_p%d" % p
        self._k_lines = "lines_p%d" % p
        self._k_level = "level_p%d" % p
        self._k_over = "game_over_p%d" % p
        over = self.vars.spec.get(self._k_over) or {}
        self._alive_value = int(over.get("alive_value", 1))

    # -- helpers ---------------------------------------------------------
    def _joint(self, discrete):
        """Put this player's buttons in their own slice and nobody else's."""
        act = [False] * (self.n_buttons * self.players)
        for b in self._combos[int(discrete)]:
            act[self.slot * self.n_buttons + b] = True
        if self.opponent is not None:
            for b in self.opponent(self):
                other = 0 if self.slot else 1
                act[other * self.n_buttons + self._idx[b]] = True
        return act

    def _ram(self):
        return self.env.unwrapped.get_ram()

    def _in_transition(self, ram):
        """True while the level-change animation is playing. The game is not
        being played then: the piece is not under control, the clear bonus is
        ticking up on its own, and treating it as play would both hand the
        agent free reward and pollute its experience with frames it cannot act
        on. It is NOT a game over either -- 0x01D2 stays 0 through every frame
        of a real top-out."""
        return self.vars.read(ram, "level_transition", 0) != 0

    def _alive(self, ram):
        return self.vars.read(ram, self._k_over, self._alive_value) == self._alive_value

    def _stats(self, ram):
        return (self.vars.read(ram, self._k_score, 0),
                self.vars.read(ram, self._k_lines, 0),
                self.vars.read(ram, self._k_level, 0))

    # -- gym API ---------------------------------------------------------
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        ram = self._ram()
        # Roll past a transition if the state happens to land in one.
        obs = self._skip_transition(obs)
        self.prev_score, self.prev_lines, _ = self._stats(self._ram())
        self.steps = 0
        return obs, self._info(self._ram())

    def _skip_transition(self, obs):
        """Advance through the level animation with no input, so the agent
        never sees frames it cannot influence and no time passes for it."""
        guard = 0
        while self._in_transition(self._ram()) and guard < 1200:
            obs, _, _, _, _ = self.env.step([False] * (self.n_buttons * self.players))
            guard += 1
        return obs

    def step(self, action):
        joint = self._joint(action)
        total = 0.0
        terminated = truncated = False
        obs = None
        for _ in range(self.frameskip):
            obs, _r, term, trunc, _i = self.env.step(joint)
            if term or trunc:
                terminated = terminated or bool(term)
                truncated = truncated or bool(trunc)
                break
        ram = self._ram()

        if self._in_transition(ram):
            # Not play time: skip it, and do not pay or charge the agent for it.
            obs = self._skip_transition(obs)
            ram = self._ram()
            # The clear bonus lands during the animation; absorb it silently so
            # the next real step does not read it as a sudden huge gain.
            self.prev_score, self.prev_lines, _ = self._stats(ram)
            return obs, 0.0, False, False, self._info(ram)

        score, lines, _level = self._stats(ram)
        dl = max(0, lines - self.prev_lines)
        if dl:
            total += self.line_reward[min(dl, len(self.line_reward) - 1)]
        total += max(0, score - self.prev_score) * self.score_scale
        total += self.step_reward          # staying alive is worth a little
        self.prev_score, self.prev_lines = score, lines

        if not self._alive(ram):
            total += self.death_penalty
            terminated = True

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
        return obs, total, terminated, truncated, self._info(ram)

    def _info(self, ram):
        score, lines, level = self._stats(ram)
        return {"score": score, "lines": lines, "level": level,
                "alive": self._alive(ram), "transition": self._in_transition(ram)}

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def make_tetris_env(**kwargs):
    return TetrisPlayerEnv(**kwargs)


if __name__ == "__main__":
    # Smoke test: does the env read the right player's numbers and end the
    # episode when THAT player tops out?
    env = make_tetris_env()
    obs, info = env.reset()
    print("reset:", info, "obs", obs.shape, "actions", env.action_space.n)
    import random
    total = 0.0
    for i in range(400):
        obs, r, term, trunc, info = env.step(random.randrange(env.action_space.n))
        total += r
        if i % 100 == 0:
            print("step %4d reward=%.3f total=%.3f %s" % (i, r, total, info))
        if term or trunc:
            print("episode ended at step", i, info)
            break
    env.close()
