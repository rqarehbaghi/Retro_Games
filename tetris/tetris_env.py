#!/usr/bin/env python3
"""
A training environment for TetrisTime, driven by games.json.

Lives under tetris/ because everything in it is specific to this game; the
generic framework stays at the repo root.

Three things make it different from train.py's SMB3 setup.

FIRST, it does not use retro's data.json at all. The integration for a ROM
imported by hand has an empty data.json, so env.step() returns info={} and a
reward of 0 forever -- there is nothing to train on. Everything here is read
straight out of env.get_ram() using the addresses in games.json, which are the
ones verified frame-by-frame against the screen. Adding a game means adding its
addresses there, not editing retro's integration files.

SECOND, the agent is PLAYER 2 in a two-player game. A policy trained on the
one-player screen would have to generalise to the two-player screen at deploy
time -- different layout, two wells, a different HUD. Training as P2 from a
two-player save state removes the problem instead of managing it: the agent
sees the screen it will play on and drives the controls it will drive. Player 1
is idle by default and cannot affect P2's reward, which is read only from P2's
own addresses.

THIRD, reward is shaped from the WELL, not just from lines. Line clears alone
are far too sparse to learn from: a random policy tops out having never cleared
one, so the gradient is almost always zero. The well is stored at 0x0600 as one
16-byte row per line -- P1 in the low 8 bytes, P2 in the high 8 -- four bits per
cell, 13 rows of 10 columns once the three wall nibbles each side are dropped.
From it we can measure holes, height and bumpiness every step, which is the
signal that actually separates a bot that survives from one that flails.

The well holds LOCKED cells only; the falling piece is not in it. That is why
the observation is pixels (which show the piece) while the well drives shaping.
Finding the active piece in RAM would allow a pure-grid observation later.

Usage:
    from tetris.tetris_env import make_tetris_env
    env = make_tetris_env()
"""
import json
import os
import sys

import cv2
import gymnasium as gym
import numpy as np
import stable_retro as retro

# tetris/ sits one level down; the framework and games.json are at the root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import custom_integrations  # noqa: E402

GAMES_JSON = os.path.join(ROOT, "games.json")

# The control map, on one player's 9-button slice. Tetris needs nothing else.
# Holding a rotate button does not spin the piece repeatedly -- the game
# debounces it -- so a frame-skip that repeats the press is safe.
TETRIS_ACTIONS = [
    [],             # do nothing (let the piece fall)
    ["LEFT"],       # move left
    ["RIGHT"],      # move right
    ["DOWN"],       # soft drop
    ["A"],          # rotate clockwise
    ["B"],          # rotate counter-clockwise
]

# Well layout, verified by driving only P2 and watching which half filled.
# The board is DOUBLE BUFFERED: the same well exists at 0x0600 and 0x0700 and
# the game alternates which one it keeps current. Reading a fixed base is a
# trap -- measured directly, a run existed where 0x0700 filled to 7 cells as
# pieces landed while 0x0600 sat flat at 0, and another where 0x0600 was the
# live one. A stale buffer lags BEHIND (it holds fewer cells, in the limit
# none), never ahead, so taking whichever mirror currently holds more filled
# cells always lands on the live board.
WELL_BASES = (0x0600, 0x0700)
WELL_ROW_STRIDE = 16    # one 16-byte line per row, both players side by side
WELL_PLAYER_OFF = 8     # P1 at +0, P2 at +8
WELL_ROWS = 13
WELL_SKIP_NIBBLES = 3   # wall nibbles at each end of the 16-nibble row
WELL_COLS = 10


def _read_well_at(ram, base, player):
    off = WELL_PLAYER_OFF if player == 2 else 0
    grid = np.zeros((WELL_ROWS, WELL_COLS), dtype=np.uint8)
    for r in range(WELL_ROWS):
        b = base + WELL_ROW_STRIDE * r + off
        nib = []
        for x in ram[b:b + 8]:
            nib += [int(x) >> 4, int(x) & 0xF]
        cells = nib[WELL_SKIP_NIBBLES:WELL_SKIP_NIBBLES + WELL_COLS]
        grid[r] = [1 if c else 0 for c in cells]
    return grid


def read_well(ram, player=2):
    """Decode one player's locked board as a 13x10 array of 0/1, from whichever
    of the two mirrors is currently live (see WELL_BASES)."""
    grids = [_read_well_at(ram, b, player) for b in WELL_BASES]
    return max(grids, key=lambda g: int(g.sum()))


def well_features(grid):
    """Holes, total height and bumpiness -- the standard Tetris heuristics.

    A hole is an empty cell with a filled cell somewhere above it in the same
    column: the thing that makes a stack unrecoverable. Bumpiness is the sum of
    height differences between neighbouring columns, which punishes a jagged
    surface that only an I-piece can fix.
    """
    rows, cols = grid.shape
    heights = np.zeros(cols, dtype=np.int32)
    holes = 0
    for c in range(cols):
        col = grid[:, c]
        filled = np.nonzero(col)[0]
        if filled.size:
            top = filled[0]
            heights[c] = rows - top
            holes += int((col[top:] == 0).sum())
    bumpiness = int(np.abs(np.diff(heights)).sum()) if cols > 1 else 0
    return {"holes": holes, "height": int(heights.sum()),
            "max_height": int(heights.max()) if cols else 0,
            "bumpiness": bumpiness}


class GameVars:
    """Read the verified values for one game out of games.json.

    Handles both shapes games.json records: `tiles` (a run of display digits,
    one per byte, byte = digit + digit_offset, most significant first) and
    `ram` (a plain byte). Nothing here knows about a specific game.
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
                digits.append(d if 0 <= d <= 9 else 0)
            return int("".join(str(d) for d in digits))
        return int(ram[base])


class TetrisPlayerEnv(gym.Env):
    """One player of a two-player TetrisTime game, as a Gym environment."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, game="TetrisTime-Nes-v0", player=2, state=None,
                 players=2, frameskip=4, opponent=None,
                 line_reward=(0.0, 2.0, 6.0, 12.0, 25.0), score_scale=0.002,
                 death_penalty=-10.0, step_reward=0.001,
                 hole_penalty=0.5, height_penalty=0.02, bump_penalty=0.02,
                 max_steps=20_000):
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
        self.player = player
        self.slot = player - 1
        self.frameskip = frameskip
        self.opponent = opponent
        self.line_reward = line_reward
        self.score_scale = score_scale
        self.death_penalty = death_penalty
        self.step_reward = step_reward
        self.hole_penalty = hole_penalty
        self.height_penalty = height_penalty
        self.bump_penalty = bump_penalty
        self.max_steps = max_steps

        self._idx = {n: i for i, n in enumerate(self.buttons) if n}
        self._combos = [[self._idx[b] for b in combo] for combo in TETRIS_ACTIONS]
        self.action_space = gym.spaces.Discrete(len(TETRIS_ACTIONS))
        self.observation_space = self.env.observation_space

        self._k_score = "score_p%d" % player
        self._k_lines = "lines_p%d" % player
        self._k_level = "level_p%d" % player
        self._k_over = "game_over_p%d" % player
        over = self.vars.spec.get(self._k_over) or {}
        self._alive_value = int(over.get("alive_value", 1))

    # -- helpers ---------------------------------------------------------
    def _joint(self, discrete):
        act = [False] * (self.n_buttons * self.players)
        for b in self._combos[int(discrete)]:
            act[self.slot * self.n_buttons + b] = True
        if self.opponent is not None:
            other = 0 if self.slot else 1
            for b in self.opponent(self):
                act[other * self.n_buttons + self._idx[b]] = True
        return act

    def _ram(self):
        return self.env.unwrapped.get_ram()

    def _in_transition(self, ram):
        """True while the level-change animation is playing. The game is not
        being played then: the piece is not under control and the clear bonus
        ticks up on its own, so treating it as play would hand the agent free
        reward and fill its buffer with frames it cannot act on. It is NOT a
        game over -- 0x01D2 stays 0 through every frame of a real top-out."""
        return self.vars.read(ram, "level_transition", 0) != 0

    def _alive(self, ram):
        return self.vars.read(ram, self._k_over, self._alive_value) == self._alive_value

    def _stats(self, ram):
        return (self.vars.read(ram, self._k_score, 0),
                self.vars.read(ram, self._k_lines, 0),
                self.vars.read(ram, self._k_level, 0))

    def _shape(self, ram):
        return well_features(read_well(ram, self.player))

    # -- gym API ---------------------------------------------------------
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        obs = self._skip_transition(obs)
        ram = self._ram()
        self.prev_score, self.prev_lines, _ = self._stats(ram)
        self.prev_feat = self._shape(ram)
        self.steps = 0
        return obs, self._info(ram)

    def _skip_transition(self, obs):
        guard = 0
        while self._in_transition(self._ram()) and guard < 1200:
            obs, _, _, _, _ = self.env.step([False] * (self.n_buttons * self.players))
            guard += 1
        return obs

    def step(self, action):
        joint = self._joint(action)
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
            # Not play time: skip it, pay nothing, charge nothing, and re-baseline
            # so the clear bonus does not read as a sudden gain on the next step.
            obs = self._skip_transition(obs)
            ram = self._ram()
            self.prev_score, self.prev_lines, _ = self._stats(ram)
            self.prev_feat = self._shape(ram)
            return obs, 0.0, False, False, self._info(ram)

        score, lines, _lvl = self._stats(ram)
        feat = self._shape(ram)
        total = self.step_reward

        dl = max(0, lines - self.prev_lines)
        if dl:
            total += self.line_reward[min(dl, len(self.line_reward) - 1)]
        # Score is kept deliberately SMALL. Soft-dropping earns drop points, so
        # a generous score weight teaches the bot to slam every piece down for
        # points and ignore where it lands -- measured: one careless episode
        # earned 2088 points, which at 0.01 would be +21 and swamp the -10 for
        # dying. It stays non-zero so DOWN is still worth using, but line clears
        # and board quality have to dominate.
        total += max(0, score - self.prev_score) * self.score_scale

        # Board shaping: pay for the CHANGE, so it is a potential-style signal
        # rather than a constant drag that the agent cannot influence.
        total -= self.hole_penalty * (feat["holes"] - self.prev_feat["holes"])
        total -= self.height_penalty * (feat["height"] - self.prev_feat["height"])
        total -= self.bump_penalty * (feat["bumpiness"] - self.prev_feat["bumpiness"])

        self.prev_score, self.prev_lines, self.prev_feat = score, lines, feat

        if not self._alive(ram):
            total += self.death_penalty
            terminated = True

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
        return obs, float(total), terminated, truncated, self._info(ram)

    def _info(self, ram):
        score, lines, level = self._stats(ram)
        f = self._shape(ram)
        return {"score": score, "lines": lines, "level": level,
                "alive": self._alive(ram), "transition": self._in_transition(ram),
                **f}

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class WarpFrame(gym.ObservationWrapper):
    """Grayscale + resize, the standard cheap observation for pixel control."""

    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width, self.height = width, height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


def make_tetris_env(warp=True, **kwargs):
    env = TetrisPlayerEnv(**kwargs)
    return WarpFrame(env) if warp else env


if __name__ == "__main__":
    import random
    env = make_tetris_env(warp=False)
    obs, info = env.reset()
    print("reset:", info)
    print("obs", obs.shape, "actions", env.action_space.n)
    total = 0.0
    for i in range(4000):
        a = 3 if random.random() < 0.5 else random.randrange(env.action_space.n)
        obs, r, term, trunc, info = env.step(a)
        total += r
        if i % 200 == 0:
            print("step %4d r=%+.3f total=%+.2f holes=%d h=%d bump=%d score=%d lines=%d"
                  % (i, r, total, info["holes"], info["height"], info["bumpiness"],
                     info["score"], info["lines"]))
        if term or trunc:
            print("episode ended at step %d  %s  total=%+.2f" % (i, info, total))
            break
    env.close()
