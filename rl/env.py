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
    decrease_event pay scale the step a variable DROPS -- a counter resetting,
                  which is how "a piece just landed" or "a life was lost" shows
                  up when there is no fixed value to match on.
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
from rl.macro import MacroPlacementWrapper      # noqa: E402
from rl.vars import GameVars, load_entry        # noqa: E402

# Used when a game names no action set. Every retro game has these buttons, and
# a game whose controls matter should declare its own.
DEFAULT_ACTIONS = [[], ["LEFT"], ["RIGHT"], ["DOWN"], ["UP"], ["A"], ["B"]]


def _fill_player(value, player):
    """Replace {player} with the player number, anywhere in a config tree.

    This is what lets ONE config train either player. A variable name like
    "game_over_p{player}" resolves to game_over_p1 or game_over_p2, so
    --player is the only thing that has to change -- rather than a config that
    names player 2 throughout and a model that is therefore stuck as player 2.

    A game with no per-player variables writes no {player} and is untouched.
    """
    if isinstance(value, str):
        return value.replace("{player}", str(player)) if "{player}" in value else value
    if isinstance(value, dict):
        return {k: _fill_player(v, player) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill_player(v, player) for v in value]
    return value


def _for_player(spec, player):
    """One player's version of a geometry block.

    Where a player's own area of the screen is decides what the agent can see,
    and on a 2-player screen the two are in different places: TetrisTime draws
    player 1's well at x=8 and player 2's at x=153, everything else identical.
    A `per_player` map holds only what differs, keyed by player number, and is
    merged over the shared values -- so the same policy can be pointed at
    either board without a second observation block to keep in sync.

        "grid": {"y": 56, "cell": 8, "cols": 10, "rows": 20,
                 "per_player": {"1": {"x": 8}, "2": {"x": 153}}}
    """
    out = dict(spec or {})
    per = out.pop("per_player", None) or {}
    mine = per.get(str(player)) or per.get(player) or {}
    out.update(mine)
    return out


class TrainingSpec:
    """Everything games.json says about training one game."""

    def __init__(self, game, overrides=None):
        self.game = game
        self.entry = load_entry(game)
        t = dict(self.entry.get("training") or {})
        t.update({k: v for k, v in (overrides or {}).items() if v is not None})
        self.players = int(t.get("players", 1))
        self.player = int(t.get("player", 1))
        # WHICH player this agent is, applied everywhere at once.
        #
        # A config that spells out "game_over_p2" and "lines_p2" trains a model
        # that can only ever be player 2, and moving it to player 1 means
        # hand-editing every one of those names. Write "{player}" instead and
        # --player rewrites them all: "game_over_p{player}" becomes
        # game_over_p1 or game_over_p2 with nothing else changed.
        t = _fill_player(t, self.player)
        self.raw = t
        self.frameskip = int(t.get("frameskip", 4))
        self.features_name = t.get("features")
        self.action_mode = t.get("action_mode", "button_stream")
        self.macro_config = dict(t.get("macro_config") or {})
        # An action is either a plain button list, or {"buttons": [...],
        # "hold": N} when how LONG it is held is part of the move. SMB3 needs
        # the second form: (["A"], 6) is a short hop and (["A"], 20) a full
        # jump, and collapsing both to one frameskip would delete the
        # difference between them.
        self.actions, self.holds, self.releases = [], [], []
        for a in (t.get("action_set") or DEFAULT_ACTIONS):
            if isinstance(a, dict):
                self.actions.append(list(a.get("buttons") or []))
                self.holds.append(int(a.get("hold", 0)) or None)
                self.releases.append(int(a["release"]) if "release" in a else None)
            else:
                self.actions.append(list(a))
                self.holds.append(None)
                self.releases.append(None)
        self.observation = dict(t.get("observation") or {})
        self.grid = _for_player(self.observation.get("grid"), self.player)
        # Board features are only meaningful once a piece has LANDED.
        # feature_gate names a variable whose DROP marks a new piece
        # (the previous one just locked).
        self.feature_gate = t.get("feature_gate")
        # Frames at the END of each decision with NO buttons held. Without one,
        # two identical decisions in a row reach the game as a single sustained
        # press: the second does nothing, and a long one triggers auto-repeat.
        # Measured on Tetris, repeated LEFT presses moved the piece
        # 7,6,6,5,5,4 -- half of them did not register -- while the same total
        # frames with one release frame gave a clean 7,6,5,4,3.
        self.release_frames = int(t.get("release_frames", 0))
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
        self.terms = _fill_player(
            list((self.entry.get("rewards") or {}).get("terms") or []), self.player)

    def _default_state(self, t):
        """The save state to start from, which depends on the player COUNT.

        A two-player save state cannot be loaded into a one-player env, so
        preferred_state may be a map keyed by how many players are running:

            "preferred_state": {"1": "level0_1p", "2": "level8_2p"}

        A plain string still works for a game with only one sensible state.
        --state overrides either.
        """
        for key in ("preferred_state", "training_state",
                    "start_state_2p" if self.players >= 2 else "start_state",
                    "start_state", "start_state_2p"):
            got = t.get(key)
            if isinstance(got, dict):
                got = got.get(str(self.players)) or got.get(self.players)
            if got:
                return got
        return None

    def states_available(self):
        try:
            d = os.path.join(custom_integrations.INTEGRATIONS_DIR, self.game)
            return sorted(f[:-6] for f in os.listdir(d) if f.endswith(".state"))
        except Exception:                                        # noqa: BLE001
            return []


class RewardModel:
    """Turns a spec's declared terms into a number, every step."""

    def __init__(self, terms, vars, feature_fn, player, gate=None):
        self.terms = terms
        self.vars = vars
        self.feature_fn = feature_fn
        self.player = player
        self.gate = gate            # {"var": ...} whose DROP means "a piece landed"
        self.prev_vars = {}
        self.prev_feats = {}
        self._gate_val = None
        self._held = {}

    def _values(self, ram, info):
        needed = {t["var"] for t in self.terms if t.get("var")}
        return {n: self.vars.read(n, ram, info) for n in needed}

    def features(self, ram, info, frame=None, spec=None):
        try:
            return self.feature_fn(self.vars, ram, info, self.player,
                                   frame=frame, spec=spec) or {}
        except TypeError:
            # A hook that predates the frame argument (RAM-only) still works.
            try:
                return self.feature_fn(self.vars, ram, info, self.player) or {}
            except Exception as e:                               # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning("Feature hook failed: %s", e)
                return {}
        except Exception as e:                                   # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning("Feature hook failed: %s", e)
            return {}

    def _gated_features(self, ram, info, frame, spec):
        """Score the board only once a piece has come to rest.

        A board sampled every step includes the FALLING piece, and the empty
        space under it reads as holes that appear and vanish as it descends.
        Measured, that made the holes term swing by +-8 per step against a
        terminal of -10: one step of noise outweighed dying, and the agent was
        being graded almost entirely on where a piece happened to be mid-air.

        With a gate, the features are recomputed only when the gate variable
        DROPS -- a new piece spawning, meaning the last one just locked -- and
        held constant in between, so the deltas are zero until the stack
        actually changes. Without a gate configured this is the old behaviour.
        """
        if not self.gate:
            return self.features(ram, info, frame, spec)
        val = self.vars.read(self.gate.get("var"), ram, info)
        fresh = (self._gate_val is None or val is None or val < self._gate_val)
        self._gate_val = val
        if fresh or not self._held:
            self._held = self.features(ram, info, frame, spec)
        return dict(self._held)

    def reset(self, ram, info, frame=None, spec=None, is_macro=False):
        self.prev_vars = self._values(ram, info)
        self._gate_val = None
        self._held = {}
        if is_macro:
            self.prev_feats = self.features(ram, info, frame, spec)
        else:
            self.prev_feats = self._gated_features(ram, info, frame, spec)

    def rebaseline(self, ram, info, frame=None, spec=None, is_macro=False):
        """Forget the last values without paying for the change.

        Used when the game was not being played -- an animation between levels
        runs the score up on its own, and charging the agent for a jump it did
        not cause is worse than paying nothing.
        """
        self.reset(ram, info, frame, spec, is_macro=is_macro)

    @staticmethod
    def label(t):
        kind = t.get("kind")
        who = t.get("var") or t.get("name") or ""
        return ("%s %s" % (kind, who)).strip()

    def step(self, ram, info, terminated, frame=None, spec=None, is_macro=False):
        now = self._values(ram, info)
        if is_macro:
            feats = self.features(ram, info, frame, spec)
        else:
            feats = self._gated_features(ram, info, frame, spec)
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
                # NOT on the terminating step. The board is read from the
                # screen, and the frame after the game-over flag is the
                # game-over ANIMATION, not a board the agent played: measured
                # on TetrisTime, holes jumped 17 -> 46 and filled cells 58 -> 72
                # in the single frame after 0x004B cleared. Diffing across that
                # made the terminal step pay anywhere from +53.7 to -69.6 when
                # the configured terminal is a flat -10..-20, so dying was
                # sometimes the best-paid move in the episode. What ending an
                # episode is worth is the `terminal` term's job alone.
                if terminated:
                    continue
                name = t.get("name")
                b = feats.get(name)
                if b is None:
                    continue
                if kind == "feature_level":
                    total += scale * b
                else:
                    a = self.prev_feats.get(name, b)
                    total += scale * (b - a)
            elif kind == "decrease_event":
                # Pays when a counter DROPS. In macro mode, each successful piece placement
                # is a landing event (unless the piece topped out and terminated).
                if is_macro:
                    if not terminated:
                        total += scale
                else:
                    name = t.get("var")
                    a, b = self.prev_vars.get(name), now.get(name)
                    if a is not None and b is not None and b < a:
                        total += scale
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
        self._releases = s.releases
        self.action_space = gym.spaces.Discrete(len(self._combos))
        self.steps = self.frames = 0
        self.observation_space = self.env.observation_space

        self.use_data_json = s.use_data_json
        self.vars = GameVars(game, entry=s.entry)
        if not self.use_data_json:
            self._warn_data_json_off(s)
        self.reward_model = RewardModel(s.terms, self.vars,
                                        feature_hooks.get(s.features_name),
                                        s.player, gate=s.feature_gate)
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

    def buttons_to_joint(self, button_names):
        act = [False] * (self.n_buttons * self.spec_.players)
        for b in button_names:
            if b in self._idx:
                act[self.slot * self.n_buttons + self._idx[b]] = True
        return act

    def empty_joint(self):
        return [False] * (self.n_buttons * self.spec_.players)

    def step_raw_frame(self, buttons):
        """Step exactly one emulator frame with specific buttons."""
        if isinstance(buttons, (list, tuple)) and (len(buttons) == 0 or isinstance(buttons[0], str)):
            joint = self.buttons_to_joint(buttons)
        elif isinstance(buttons, (list, tuple, np.ndarray)):
            joint = buttons
        else:
            joint = self.empty_joint()
        obs, r, term, trunc, info = self.env.step(joint)
        self.frames += 1
        ram = self._ram()
        if self._ended(ram, self._seen(info)):
            term = True
        return obs, r, term, trunc, info

    def finalize_macro_step(self, obs, info, env_term, env_trunc):
        """Finalizes one high-level macro placement action and returns step outputs."""
        self.steps += 1
        # Same rule the button path follows: a stretch the agent cannot act on
        # is run past and paid nothing. A level change animates the score up on
        # its own and redraws the board, and charging -- or crediting -- the
        # agent for a jump it did not cause is worse than paying nothing.
        if self._skipping(self._ram(), self._seen(info)):
            obs, info = self._skip(obs, info)
            ram = self._ram()
            self.reward_model.rebaseline(ram, self._seen(info), obs,
                                         self.spec_.grid, is_macro=True)
            return obs, 0.0, False, False, self._info(
                ram, self._seen(info), self.reward_model.prev_feats)
        if not (env_term or env_trunc):
            obs, info = self._settle_board(obs, info)
        ram = self._ram()
        terminated = self._ended(ram, self._seen(info)) or env_term
        reward, feats = self.reward_model.step(ram, self._seen(info), terminated,
                                               obs, self.spec_.grid, is_macro=True)
        truncated = env_trunc or self.steps >= self.spec_.max_steps
        return obs, float(reward), bool(terminated), bool(truncated), \
            self._info(ram, self._seen(info), feats)

    def _settle_board(self, obs, info):
        """Let an animation finish before the board is scored.

        A macro step ends when the agent's next decision becomes possible, and
        on Tetris that is when the next piece spawns -- which happens BEFORE a
        line clear has finished collapsing the rows above it. Scoring there
        reads a board that is mid-animation. Measured: a clear that left 0
        holes and height 2 was scored as 1 hole and height 7, and one that left
        1 hole and height 9 was scored as 3 holes and height 15.

        The cost is not the size of the error but WHERE it lands. The step that
        clears the lines is charged for holes that do not exist, and the next
        step is refunded for removing them -- so the single most valuable action
        in the game is taxed and an unrelated later one is paid for it.

        Waiting on the game's own FEATURES rather than a fixed frame count
        keeps this generic: any game with a feature hook settles when its
        features stop moving, and a game without one does nothing here.
        """
        cfg = dict(self.spec_.raw.get("settle_board") or {})
        if not cfg:
            return obs, info
        want = int(cfg.get("stable_frames", 2))
        prev, same = None, 0
        for _ in range(int(cfg.get("max_frames", 40))):
            cur = self.reward_model.features(self._ram(), self._seen(info),
                                             obs, self.spec_.grid)
            if prev is not None and cur == prev:
                same += 1
                if same >= want:
                    break
            else:
                same = 0
            prev = cur
            obs, _r, term, trunc, info = self.env.step(self.empty_joint())
            self.frames += 1
            if term or trunc:
                break
        return obs, info

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
            self.frames += 1
            guard += 1
        return obs, info

    def _info(self, ram, info, feats=None):
        out = {n: self.vars.read(n, ram, info) for n in self.vars.names()}
        out.update({k: v for k, v in (feats or {}).items()})
        out = {k: v for k, v in out.items() if v is not None}
        # Emulator frames since reset. A decision is NOT a fixed number of
        # frames once a game uses macro actions -- a Tetris placement runs 18
        # to 54 -- so multiplying decisions by frameskip to get survival time
        # is wrong by about 10x. Anything comparing runs must read this.
        out["frames"] = self.frames
        return out

    # -- gym API ---------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, info = self._skip(obs, info)
        ram = self._ram()
        is_macro = (getattr(self.spec_, "action_mode", "button_stream") == "macro_placement")
        self.reward_model.reset(ram, self._seen(info), obs, self.spec_.grid, is_macro=is_macro)
        self.steps = 0
        self.frames = 0
        return obs, self._info(ram, self._seen(info), self.reward_model.prev_feats)

    def step(self, action):
        joint = self._joint(action)
        obs = None
        info = {}
        env_term = env_trunc = False
        hold = self._holds[int(action)] or self.spec_.frameskip
        act_rel = self._releases[int(action)]
        base_rel = act_rel if act_rel is not None else self.spec_.release_frames
        release = min(max(0, base_rel), max(0, hold - 1)) if hold > 1 else 0
        for _ in range(hold - release):
            obs, _r, term, trunc, info = self.env.step(joint)
            self.frames += 1
            # Keep the integration's OWN done signal. A standard integration
            # (SMB3) publishes one through scenario.json, and a game whose
            # games.json declares no episode_end has nothing else to end on.
            if self.use_data_json:
                env_term = env_term or bool(term)
                env_trunc = env_trunc or bool(trunc)
            if term or trunc:
                break
        # Let go, so the next decision is a fresh press rather than a hold.
        for _ in range(release):
            if env_term or env_trunc:
                break
            obs, _r, term, trunc, info = self.env.step(
                [False] * (self.n_buttons * self.spec_.players))
            self.frames += 1
            if self.use_data_json:
                env_term = env_term or bool(term)
                env_trunc = env_trunc or bool(trunc)
        ram = self._ram()

        if self._skipping(ram, self._seen(info)):
            obs, info = self._skip(obs, info)
            ram = self._ram()
            self.reward_model.rebaseline(ram, self._seen(info), obs, self.spec_.grid)
            return obs, 0.0, False, False, self._info(ram, self._seen(info),
                                                      self.reward_model.prev_feats)

        terminated = self._ended(ram, self._seen(info)) or env_term
        reward, feats = self.reward_model.step(ram, self._seen(info), terminated,
                                               obs, self.spec_.grid)
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


def build_grid_observation(frame, ram, gspec, vars, player):
    """Canonical builder for grid observations shared by training and inference.

    Produces a 1D vector consisting of:
      1. Flattened rows x cols board (optionally masking the active falling piece
         if mask_piece is enabled in gspec, so the grid represents settled stack only).
      2. Active piece one-hot (7 floats).
      3. Piece coordinates: normalized rotation, column, and row (3 floats).
    """
    from rl import features as fh
    gspec = gspec or {}
    cols = int(gspec.get("cols", 10))
    g = fh.grid_from_frame(frame, gspec)
    if gspec.get("mask_piece", False) and ram is not None:
        g = fh.mask_piece(g, ram, vars, player, gspec)
    g = g.astype(np.float32).ravel()

    def rd(name, default=0):
        if ram is None:
            return default
        v = vars.read("%s_p%d" % (name, player), ram, {}, default)
        return default if v is None else int(v)

    onehot = np.zeros(7, dtype=np.float32)
    t = rd("piece_type", 0)
    if 0 <= t < 7:
        onehot[t] = 1.0

    extra = np.concatenate([
        onehot,
        np.array([
            rd("piece_rot", 0) / 3.0,
            rd("piece_col", 0) / max(1, cols),
            rd("piece_row", 0) / 24.0,
        ], dtype=np.float32)
    ])
    return np.concatenate([g, extra]).astype(np.float32)


class GridObservation(gym.ObservationWrapper):
    """The board as numbers instead of a picture, plus the piece being placed.

    Pixels were tried first and did not learn: three separate fixes to the crop,
    the aspect and the board read left survival flat. A CNN has to rediscover
    "these 8x8 blobs are a 10-wide grid" from scratch, which is most of the
    problem, and Tetris then needs the piece and the board related to each
    other. Handing over the grid removes that entirely.

    The board comes from the rendered frame rather than RAM -- see
    features.grid_from_frame for why -- and the piece (type, rotation, column,
    row) from the verified addresses in games.json, since it is NOT in the board.
    """

    def __init__(self, env, grid_spec, vars, player):
        super().__init__(env)
        self.gspec = dict(grid_spec or {})
        self.vars = vars
        self.player = player
        self.rows = int(self.gspec.get("rows", 20))
        self.cols = int(self.gspec.get("cols", 10))
        # board cells + piece type (one-hot 7) + rotation + column + row
        self.extra = 7 + 3
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.rows * self.cols + self.extra,),
            dtype=np.float32)

    def _get_ram(self):
        """Robustly retrieve RAM across any wrappers."""
        if hasattr(self.env, "_ram"):
            return self.env._ram()
        unwrapped = getattr(self.env, "unwrapped", self.env)
        if hasattr(unwrapped, "_ram"):
            return unwrapped._ram()
        if hasattr(unwrapped, "get_ram"):
            return unwrapped.get_ram()
        if hasattr(unwrapped, "env"):
            inner = unwrapped.env
            if hasattr(inner, "_ram"):
                return inner._ram()
            if hasattr(inner, "get_ram"):
                return inner.get_ram()
        return getattr(self.env, "get_ram", lambda: None)()

    def observation(self, frame):
        ram = self._get_ram()
        return build_grid_observation(frame, ram, self.gspec, self.vars, self.player)


def make_env(game, overrides=None, warp=True, render_mode="rgb_array"):
    """One game, configured from games.json, ready for a vectoriser."""
    env = GenericRetroEnv(game, overrides, render_mode=render_mode)
    if env.spec_.action_mode == "macro_placement":
        from rl.macro import MacroPlacementWrapper
        env = MacroPlacementWrapper(env, env.spec_.macro_config, env.vars, env.spec_.player)
    obs_cfg = env.spec_.observation
    if obs_cfg.get("kind") == "grid":
        return GridObservation(env, env.spec_.grid, env.vars, env.spec_.player)
    if warp and obs_cfg.get("kind", "pixels") == "pixels":
        env = WarpFrame(env,
                        width=int(obs_cfg.get("width", 84)),
                        height=int(obs_cfg.get("height", 84)),
                        crop=obs_cfg.get("crop"))
    return env


def make_observer(game, overrides=None, n_stack=4):
    """Build observations the way TRAINING builds them, for anything outside it.

    A policy is only valid on the observation it learned from. play_engine used
    to construct its own -- a fixed grayscale image -- which silently diverged
    the moment the crop, the size, or the whole observation KIND changed here.
    Both now come from this one function, so they cannot drift apart.

    Returns (process, stacked_shape):
        process(frame, ram) -> one observation
        stacked_shape        what the model sees after n_stack of them
    """
    spec = TrainingSpec(game, overrides)
    o = spec.observation or {}
    kind = o.get("kind", "pixels")
    vars = GameVars(game, entry=spec.entry)
    player = spec.player

    if kind == "grid":
        gs = dict(spec.grid or {})
        rows = int(gs.get("rows", 20))
        cols = int(gs.get("cols", 10))

        def process(frame, ram):
            return build_grid_observation(frame, ram, gs, vars, player)

        # A grid observation is a single 1D vector (rows*cols + piece info), not stacked.
        return process, (rows * cols + 10,)

    crop = o.get("crop")
    width = int(o.get("width", 84))
    height = int(o.get("height", 84))

    def process(frame, ram=None):
        if crop:
            x, y, w, h = [int(v) for v in crop]
            frame = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    return process, (n_stack, height, width)
