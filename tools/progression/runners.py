"""Play one game headless and hand every emulator frame to a callback.

The panel and grid machinery must not care how an agent chooses its moves, so
the policy lives here and everything else -- cropping, labelling, freezing,
composing -- stays generic. Two runners, picked by the game's own declared
`algorithm` in games.json:

    afterstate   a move is a (rotation, column) placement chosen by a value net
                 over simulated afterstates -- TetrisTime and any other
                 falling-block game with an `afterstate` block
    ppo          a move is a button combo chosen every step by a policy net --
                 SuperMarioBros3 and everything else

Both report the same four things: frames (through the callback), a couple of
labelled numbers to burn onto the panel, a decision count, and why the game
ended. Supporting a third algorithm means adding a runner here and changing
nothing in video.py.

Nothing game-specific belongs in this file. The numbers on the panel come from
the game's `report_stats`; the well geometry comes from its `observation.grid`;
the checkpoint layout comes from the CLI.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CONTROL = "control"          # the untrained net, as a column name
# cap <= 0 means PLAY TO THE END: no move budget, the game stops when the
# agent loses. Honest, and unbounded -- an agent that does not lose does not
# stop, which is what a cap exists to prevent.
FRAME_STACK = 4              # train.py's default, and a policy only works on
                             # the observation shape it was trained with


class FrameTap:
    """Wraps the raw retro env so EVERY emulator frame is seen.

    A training env steps the emulator internally while skipping transition
    animations, and those frames never come back from env.step -- a video built
    from step returns would jump over every level change.

    It also stops at the end of the FIRST episode: SB3's vector envs auto-reset
    on done, and without this the panel would end on a few frames of a freshly
    restarted game instead of on the game over."""

    def __init__(self, inner, sink):
        self._inner, self._sink, self._alive = inner, sink, True

    def step(self, action):
        out = self._inner.step(action)
        if self._alive:
            self._sink(out[0])
        return out

    def reset(self, **kwargs):
        self._alive = False                  # an auto-reset: the episode is over
        return self._inner.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def spec_for(game, player=0, state=None):
    """The game's training spec -- video.py reads its grid to size the crop."""
    import custom_integrations                                    # noqa: F401
    from rl.env import TrainingSpec
    overrides = {}
    if player:
        overrides["player"] = player
    if state:
        overrides["state"] = state
    return TrainingSpec(game, overrides)


def counter_modulus(gv, *names):
    """When does this counter roll over, according to its OWN declaration?

    A `tiles` variable is read from N digit tiles on the HUD, so it counts to
    10^N - 1 and then starts again -- the width is already recorded in
    games.json with its evidence, so the modulus is derived from it rather
    than guessed. 0 means no known modulus and no correction is applied."""
    for name in names:
        s = gv.spec.get(name) or {}
        if s.get("source") == "tiles" and s.get("length"):
            return 10 ** int(s["length"])
    return 0


def _stats_from_info(info, names, player, limit=2):
    """The game's own declared report_stats, as (LABEL, value) pairs.

    Two of them: a panel is one well wide, and a third line ran into the board.
    Which two is the game's choice -- report_stats is ordered."""
    out = []
    for raw in names:
        value = info.get(raw.replace("{player}", str(player)))
        if value is None:
            continue
        label = raw.replace("_p{player}", "").replace("{player}", "").upper()
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        out.append((label, value))
        if len(out) == limit:
            break
    return out


def run_afterstate(spec, checkpoint, state, cap, player, on_frame):
    """A move is a placement, so a decision is a placement.

    The headline number is lines gained SINCE the state was loaded: a restored
    mid-game state does not start at zero, so the raw counter is not the
    measurement."""
    from rl.env import make_env
    from rl.vars import GameVars
    from rl.simulators import get_simulator
    from rl.afterstate import AfterstateAgent

    cfg = spec.afterstate_config
    sim = get_simulator(cfg.get("simulator") or spec.features_name or spec.game, config=cfg)
    agent = AfterstateAgent(input_dim=sim.feature_dim, device="cpu",
                            model_type=cfg.get("model_type", "mlp"),
                            zero_init_value=(checkpoint == CONTROL))
    if checkpoint != CONTROL:
        agent.load(checkpoint)
    gv = GameVars(spec.game, entry=spec.entry)
    lines_var = cfg.get("lines_var", "lines_p{player}").replace("{player}", str(player))

    env = make_env(spec.game, {"player": player, "state": state})
    core = env.unwrapped
    obs, info = env.reset()
    box = {"stats": [("LINES", 0), ("PIECES", 0)]}
    on_frame(np.asarray(core.env.unwrapped.render()), box)
    core.env = FrameTap(core.env, lambda frame: on_frame(frame, box))

    lines0 = raw_previous = int(info.get(lines_var, 0) or 0)
    modulus = counter_modulus(gv, lines_var, "lines")
    wraps = 0
    decisions, pending, end = 0, False, "placement_cap"
    while cap <= 0 or decisions < cap:
        action = None
        if not pending:
            cands = sim.get_candidates(obs=obs, ram=getattr(core, "ram", None),
                                       info=info, vars=gv, spec=spec)
            if cands:
                action = agent.select_action(cands, epsilon=0.0)[0]
                pending, decisions = True, decisions + 1
        obs, _r, term, trunc, info = env.step(action)
        # The counter is a fixed number of DIGITS on the HUD, so it rolls over
        # -- and a counter that only ever counts up cannot decrease for any
        # other reason. A 100k checkpoint that cleared 1057 lines was recorded
        # as 57, which put it below checkpoints it had beaten several times
        # over. Half the modulus is the threshold so an ordinary clear of four
        # lines is never mistaken for a roll.
        raw = int(info.get(lines_var, 0) or 0)
        if modulus and raw < raw_previous - modulus // 2:
            wraps += 1
        raw_previous = raw
        box["stats"] = [("LINES", wraps * modulus + raw - lines0),
                        ("PIECES", decisions)]
        if info.get("afterstate_ready", True):
            pending = False
        if term or trunc:
            end = "game_over" if term else "truncated"
            break
    last = np.asarray(core.env.unwrapped.render())
    env.close()
    return {"end_reason": end, "decisions": decisions, "stats": box["stats"], "last": last}


def run_policy(spec, checkpoint, state, cap, player, on_frame):
    """A move is a button combo, chosen every step by a policy network.

    The env is built by train.build_envs and the observation by
    train.policy_for, because a policy is only valid on the exact observation
    it was trained on -- same wrappers, same frame stack, same channel order."""
    from stable_baselines3 import PPO
    from train import build_envs, policy_for

    policy, image = policy_for(spec)
    vec = build_envs(spec.game, {"player": player, "state": state}, 1, FRAME_STACK, image)
    base = vec
    while hasattr(base, "venv"):          # VecTransposeImage -> VecFrameStack -> DummyVecEnv
        base = base.venv
    inner = base.envs[0]                  # Monitor(make_env(...))
    core = inner.unwrapped
    model = None if checkpoint == CONTROL else PPO.load(checkpoint, device="cpu")

    obs = vec.reset()
    box = {"stats": []}
    on_frame(np.asarray(core.env.unwrapped.render()), box)
    core.env = FrameTap(core.env, lambda frame: on_frame(frame, box))

    decisions, end = 0, "placement_cap"
    while cap <= 0 or decisions < cap:
        if model is None:
            # The control column for a policy game: the untrained equivalent is
            # a uniformly random button press, not a zeroed net -- an untrained
            # PPO policy IS approximately that, without needing a checkpoint.
            action = np.array([vec.action_space.sample()])
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _r, dones, infos = vec.step(action)
        decisions += 1
        info = infos[0] if len(infos) else {}
        box["stats"] = (_stats_from_info(info, spec.report_stats or [], player)
                        or [("STEPS", decisions)])
        if dones[0]:
            end = "truncated" if info.get("TimeLimit.truncated") else "game_over"
            break
    last = np.asarray(core.env.unwrapped.render())
    vec.close()
    return {"end_reason": end, "decisions": decisions, "stats": box["stats"], "last": last}


def run(game, checkpoint, state, cap, player, on_frame):
    """Play one (checkpoint, state) pair, whatever kind of game it is.

    `on_frame(frame, box)` is called for every emulator frame; `box["stats"]`
    holds the current (LABEL, value) pairs to burn onto it."""
    spec = spec_for(game, player, state)
    runner = run_afterstate if spec.algorithm == "afterstate" else run_policy
    return runner(spec, checkpoint, state, cap, player, on_frame)
