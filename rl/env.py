#!/usr/bin/env python3
"""
One environment builder that trains any game, configured from games.json.

The old trainer was generic in its addresses but not in its REWARD: it had a
fixed platformer vocabulary (progress, jump, stuck, backtrack) wired into the
code, so a game that does not move rightwards had nothing to say. Tetris is
paid on holes and line clears; Mario on ground covered and a course clear.
Neither is a special case of the other.

So reward is a LIST OF TERMS declared in games.json, each naming a variable
(from the same verified `variables` block everything else reads) or a computed
feature, and how much it is worth:

    "rewards": {
      "terms": [
        {"kind": "delta",        "var": "progress",     "scale": 0.25},
        {"kind": "delta",        "var": "score",        "scale": 0.2},
        {"kind": "event",        "var": "course_clear", "equals": 255,
                                 "scale": 800.0},
        {"kind": "feature_delta","name": "holes",       "scale": -0.5},
        {"kind": "terminal",     "scale": -25.0}
      ]
    }

Term kinds:

    delta         pay scale * (value - previous). `positive_only` (default
                  true) ignores decreases, which is what you want for a score
                  or a distance that resets between lives.
    delta_tiered  for a counter that can jump by more than one, pay
                  tiers[jump] -- a Tetris of four lines is worth far more than
                  four singles, and a flat per-line rate cannot express that.
    feature_delta same, on a value from the game's feature hook.
    feature_level pay scale * the feature itself, every step.
    event         pay scale the step a variable BECOMES `equals`.
    step          a flat amount per decision.
    terminal      paid once when the episode ends by the game's own end
                  condition (not on a truncation, which is our time limit and
                  says nothing about how the agent played).

Everything a game needs beyond this -- which state to start from, which player
to drive, the button map -- lives in its `training` block, so adding a game is
a JSON edit plus, only if its reward needs computing, a feature hook.
"""
import os
import sys

import cv2
import gymnasium as gym
import numpy as np
import stable_retro as retro

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import custom_integrations                      # noqa: E402
from rl import features as feature_hooks        # noqa: E402
from rl.vars import GameVars, load_entry        # noqa: E402

# Used when a game names no action set. Every retro game has these buttons, and
# a game whose controls matter should declare its own.
DEFAULT_ACTIONS = [[], ["LEFT"], ["RIGHT"], ["DOWN"], ["UP"], ["A"], ["B"]]


class TrainingSpec:
    """Everything games.json says about training one game."""

    def __init__(self, game, overrides=None):
        self.game = game
        self.entry = load_entry(game)
        t = dict(self.entry.get("training") or {})
        t.update({k: v for k, v in (overrides or {}).items() if v is not None})
        self.raw = t
        self.players = int(t.get("players", 1))
        self.player = int(t.get("player", 1))
        self.frameskip = int(t.get("frameskip", 4))
        self.features_name = t.get("features")
        # An action is either a plain button list, or {"buttons": [...],
        # "hold": N} when how LONG it is held is part of the move. SMB3 needs
        # the second form: (["A"], 6) is a short hop and (["A"], 20) a full
        # jump, and collapsing both to one frameskip would delete the
        # difference between them.
        self.actions, self.holds = [], []
        for a in (t.get("action_set") or DEFAULT_ACTIONS):
            if isinstance(a, dict):
                self.actions.append(list(a.get("buttons") or []))
                self.holds.append(int(a.get("hold", 0)) or None)
            else:
                self.actions.append(list(a))
                self.holds.append(None)
        self.observation = dict(t.get("observation") or {})
        self.episode_end = t.get("episode_end")     # {"var":..., "equals":...}
        self.skip_while = t.get("skip_while")       # {"var":..., "equals":...}
        self.max_steps = int(t.get("max_steps", 20000))
        # data.json is retro's integration file: it is what fills the `info`
        # dict and what supplies a scenario `done`. A hand-made integration has
        # an empty one, and relying on it means a game only trains if somebody
        # wrote retro config for it. With this off, EVERYTHING comes from this
        # file instead -- RAM variables for values, episode_end for the ending.
        self.use_data_json = bool(t.get("use_data_json", True))
        self.ppo = dict(t.get("ppo") or {})
        self.state = t.get("state") or self._default_state(t)
        self.terms = list((self.entry.get("rewards") or {}).get("terms") or [])

    def _default_state(self, t):
        for key in ("start_state_2p" if self.players >= 2 else "start_state",
                    "start_state", "start_state_2p"):
            if t.get(key):
                return t[key]
        return None

    def states_available(self):
        try:
            d = os.path.join(custom_integrations.INTEGRATIONS_DIR, self.game)
            return sorted(f[:-6] for f in os.listdir(d) if f.endswith(".state"))
        except Exception:                                        # noqa: BLE001
            return []


class RewardModel:
    """Turns a spec's declared terms into a number, every step."""

    def __init__(self, terms, vars, feature_fn, player):
        self.terms = terms
        self.vars = vars
        self.feature_fn = feature_fn
        self.player = player
        self.prev_vars = {}
        self.prev_feats = {}

    def _values(self, ram, info):
        needed = {t["var"] for t in self.terms if t.get("var")}
        return {n: self.vars.read(n, ram, info) for n in needed}

    def features(self, ram, info):
        try:
            return self.feature_fn(self.vars, ram, info, self.player) or {}
        except Exception:                                        # noqa: BLE001
            return {}

    def reset(self, ram, info):
        self.prev_vars = self._values(ram, info)
        self.prev_feats = self.features(ram, info)

    def rebaseline(self, ram, info):
        """Forget the last values without paying for the change.

        Used when the game was not being played -- an animation between levels
        runs the score up on its own, and charging the agent for a jump it did
        not cause is worse than paying nothing.
        """
        self.reset(ram, info)

    @staticmethod
    def label(t):
        kind = t.get("kind")
        who = t.get("var") or t.get("name") or ""
        return ("%s %s" % (kind, who)).strip()

    def step(self, ram, info, terminated):
        now = self._values(ram, info)
        feats = self.features(ram, info)
        total = 0.0
        self.breakdown = {}          # per-term contribution, for --explain-reward
        for t in self.terms:
            before = total
            kind = t.get("kind")
            scale = float(t.get("scale", 0.0))
            if kind == "step":
                total += scale
            elif kind == "terminal":
                if terminated:
                    total += scale
            elif kind in ("delta", "delta_tiered"):
                name = t.get("var")
                a, b = self.prev_vars.get(name), now.get(name)
                if a is None or b is None:
                    continue
                diff = b - a
                if kind == "delta_tiered":
                    tiers = t.get("tiers") or []
                    if diff > 0 and tiers:
                        total += float(tiers[min(int(diff), len(tiers) - 1)])
                else:
                    if t.get("positive_only", True):
                        diff = max(0, diff)
                    total += scale * diff
            elif kind in ("feature_delta", "feature_level"):
                name = t.get("name")
                b = feats.get(name)
                if b is None:
                    continue
                if kind == "feature_level":
                    total += scale * b
                else:
                    a = self.prev_feats.get(name, b)
                    total += scale * (b - a)
            elif kind == "event":
                name = t.get("var")
                want = t.get("equals")
                a, b = self.prev_vars.get(name), now.get(name)
                if b is not None and want is not None:
                    became = int(b) == int(want) and (a is None or int(a) != int(want))
                    if became:
                        total += scale
            if total != before:
                self.breakdown[self.label(t)] = round(total - before, 4)
        self.prev_vars, self.prev_feats = now, feats
        return total, feats


class GenericRetroEnv(gym.Env):
    """A retro game as a Gym env, entirely described by games.json."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, game, overrides=None, render_mode="rgb_array"):
        super().__init__()
        custom_integrations.register()
        self.spec_ = TrainingSpec(game, overrides)
        s = self.spec_
        state = self._resolve_state(s)
        self.env = retro.make(game=game, state=state, players=s.players,
                              use_restricted_actions=retro.Actions.ALL,
                              render_mode=render_mode)
        self.buttons = self.env.unwrapped.buttons
        self.n_buttons = self.env.num_buttons
        self.slot = max(0, s.player - 1)
        self._idx = {n: i for i, n in enumerate(self.buttons) if n}
        self._combos = [[self._idx[b] for b in combo if b in self._idx]
                        for combo in s.actions]
        self._holds = s.holds
        self.action_space = gym.spaces.Discrete(len(self._combos))
        self.observation_space = self.env.observation_space

        self.use_data_json = s.use_data_json
        self.vars = GameVars(game, entry=s.entry)
        if not self.use_data_json:
            self._warn_data_json_off(s)
        self.reward_model = RewardModel(s.terms, self.vars,
                                        feature_hooks.get(s.features_name),
                                        s.player)
        if not s.terms:
            print("WARNING: %s declares no rewards.terms in games.json, so every "
                  "step scores 0 and nothing can be learned." % game)

    def _warn_data_json_off(self, s):
        """Say exactly what stops working, rather than failing silently."""
        print("data.json: OFF -- info is ignored and the integration's own done "
              "signal is not used.")
        blind = sorted({t["var"] for t in s.terms if t.get("var")
                        and (self.vars.spec.get(t["var"]) or {}).get("source")
                        in ("info", "info16")})
        if blind:
            print("  WARNING: these reward terms read `info`, which is empty "
                  "without data.json, so they will never pay: %s"
                  % ", ".join(blind))
            print("  Give each a RAM address in games.json (tools/find_game_vars.py "
                  "finds them) or drop the term.")
        if not s.episode_end:
            print("  WARNING: no episode_end declared, and the integration's done "
                  "is off, so episodes can only ever end by truncation at "
                  "max_steps (%d)." % s.max_steps)

    # -- state -----------------------------------------------------------
    def _resolve_state(self, s):
        have = s.states_available()
        if s.state:
            if have and s.state not in have:
                print("")
                print("No save state named %r for %s." % (s.state, s.game))
                print("Available: %s" % (", ".join(have) if have else "(none)"))
                sys.exit("Pick one with --state.")
            return s.state
        if have:
            return have[0]
        return retro.State.DEFAULT

    # -- helpers ---------------------------------------------------------
    def _joint(self, discrete):
        act = [False] * (self.n_buttons * self.spec_.players)
        for b in self._combos[int(discrete)]:
            act[self.slot * self.n_buttons + b] = True
        return act

    def _ram(self):
        return self.env.unwrapped.get_ram()

    def _seen(self, info):
        """The info the rest of the env is allowed to look at."""
        return info if self.use_data_json else {}

    def _skipping(self, ram, info):
        c = self.spec_.skip_while
        return bool(c) and self.vars.matches(c["var"], ram, info, c.get("equals"))

    def _ended(self, ram, info):
        c = self.spec_.episode_end
        return bool(c) and self.vars.matches(c["var"], ram, info, c.get("equals"))

    def _skip(self, obs, info):
        """Run past a stretch the agent cannot act on, paying nothing for it."""
        guard = 0
        while self._skipping(self._ram(), info) and guard < 2000:
            obs, _r, _t, _tr, info = self.env.step(
                [False] * (self.n_buttons * self.spec_.players))
            guard += 1
        return obs, info

    def _info(self, ram, info, feats=None):
        out = {n: self.vars.read(n, ram, info) for n in self.vars.names()}
        out.update({k: v for k, v in (feats or {}).items()})
        return {k: v for k, v in out.items() if v is not None}

    # -- gym API ---------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, info = self._skip(obs, info)
        ram = self._ram()
        self.reward_model.reset(ram, self._seen(info))
        self.steps = 0
        return obs, self._info(ram, self._seen(info), self.reward_model.prev_feats)

    def step(self, action):
        joint = self._joint(action)
        obs = None
        info = {}
        env_term = env_trunc = False
        hold = self._holds[int(action)] or self.spec_.frameskip
        for _ in range(hold):
            obs, _r, term, trunc, info = self.env.step(joint)
            # Keep the integration's OWN done signal. A standard integration
            # (SMB3) publishes one through scenario.json, and a game whose
            # games.json declares no episode_end has nothing else to end on.
            if self.use_data_json:
                env_term = env_term or bool(term)
                env_trunc = env_trunc or bool(trunc)
            if term or trunc:
                break
        ram = self._ram()

        if self._skipping(ram, self._seen(info)):
            obs, info = self._skip(obs, info)
            ram = self._ram()
            self.reward_model.rebaseline(ram, self._seen(info))
            return obs, 0.0, False, False, self._info(ram, self._seen(info),
                                                      self.reward_model.prev_feats)

        terminated = self._ended(ram, self._seen(info)) or env_term
        reward, feats = self.reward_model.step(ram, self._seen(info), terminated)
        self.steps += 1
        truncated = env_trunc or self.steps >= self.spec_.max_steps
        return obs, float(reward), bool(terminated), bool(truncated), \
            self._info(ram, self._seen(info), feats)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class WarpFrame(gym.ObservationWrapper):
    """Crop to what matters, then grayscale and resize.

    The crop is the important half. A two-player screen spends most of its
    pixels on things this agent cannot act on -- the opponent's board, the HUD,
    the decorative middle -- and after a downscale to 84x84 the agent's own well
    was about two pixels per cell, which is close to unreadable. Cropping to the
    player's own area first spends the whole observation on the part that
    matters and lands nearer six pixels per cell.

    `crop` is [x, y, w, h] in the game's own pixels, declared per game in
    games.json so nothing here knows which game it is looking at.
    """

    def __init__(self, env, width=84, height=84, crop=None):
        super().__init__(env)
        self.width, self.height = width, height
        self.crop = [int(v) for v in crop] if crop else None
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, frame):
        if self.crop:
            x, y, w, h = self.crop
            frame = frame[y:y + h, x:x + w]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


def make_env(game, overrides=None, warp=True, render_mode="rgb_array"):
    """One game, configured from games.json, ready for a vectoriser."""
    env = GenericRetroEnv(game, overrides, render_mode=render_mode)
    obs_cfg = env.spec_.observation
    if warp and obs_cfg.get("kind", "pixels") == "pixels":
        env = WarpFrame(env,
                        width=int(obs_cfg.get("width", 84)),
                        height=int(obs_cfg.get("height", 84)),
                        crop=obs_cfg.get("crop"))
    return env
