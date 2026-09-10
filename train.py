#!/usr/bin/env python3
"""
Train a PPO agent on ANY game in games.json. One CLI, one set of flags.

    # what can this game do?
    python train.py --game TetrisTime-Nes-v0 --describe

    # train
    python train.py --game TetrisTime-Nes-v0 --state level8_2p --timesteps 500000 --n-envs 4

    # watch a checkpoint play
    python train.py --game TetrisTime-Nes-v0 --state level8_2p \
        --play checkpoints/TetrisTime-Nes-v0/final.zip

Nothing here knows about a specific game. Everything game-shaped -- which
player to drive, the button map, which save state, when an episode ends, and
what earns reward -- is read from that game's entry in games.json (or another
file via --config). Adding a game is a JSON edit, plus a feature hook in
rl/features.py only if its reward has to be computed rather than read.

The reward vocabulary is declarative, which is what makes one trainer enough
for games as different as Tetris and Mario: see rl/env.py for the term kinds.

Per-game PPO defaults come from that game's "training": {"ppo": {...}} and any
flag given here overrides them, so a tuned game stays tuned without hard-coding
its numbers.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO                                  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback   # noqa: E402
from stable_baselines3.common.monitor import Monitor                # noqa: E402
from stable_baselines3.common.vec_env import (                      # noqa: E402
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage)

from rl import vars as rlvars                                       # noqa: E402
from rl.env import TrainingSpec, make_env                           # noqa: E402


def _factory(seed, game, overrides):
    def _init():
        env = make_env(game, overrides)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def build_envs(game, overrides, n_envs, frame_stack):
    fns = [_factory(i, game, overrides) for i in range(n_envs)]
    # stable-retro allows ONE emulator per process, so parallel envs must be
    # separate processes; a single env stays in-process.
    vec = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    if frame_stack > 1:
        vec = VecFrameStack(vec, n_stack=frame_stack)   # so motion is visible
    return VecTransposeImage(vec)


def overrides_from(args):
    """CLI values that override the game's training block (None = leave it)."""
    o = {"state": args.state, "player": args.player, "players": args.players,
         "frameskip": args.frameskip, "max_steps": args.max_steps}
    if args.no_data_json:
        o["use_data_json"] = False
    for item in args.set or []:
        if "=" not in item:
            sys.exit("--set expects key=value, got %r" % item)
        k, v = item.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass                     # leave it a string
        o[k] = v
    return {k: v for k, v in o.items() if v is not None}


def describe(game, spec):
    print("game        : %s" % game)
    print("players     : %d (agent drives player %d)" % (spec.players, spec.player))
    print("state       : %s" % (spec.state or "(integration default)"))
    print("available   : %s" % (", ".join(spec.states_available()) or "(none)"))
    print("frameskip   : %d" % spec.frameskip)
    print("actions     : %d  %s" % (len(spec.actions), spec.actions))
    print("features    : %s" % (spec.features_name or "(none)"))
    print("data.json   : %s" % ("used" if spec.use_data_json else "OFF (config only)"))
    print("episode end : %s" % (spec.episode_end or "(never -- only truncation)"))
    print("skip while  : %s" % (spec.skip_while or "(nothing)"))
    print("ppo defaults: %s" % (spec.ppo or "(library defaults)"))
    print("reward terms:")
    for t in spec.terms:
        print("   %s" % t)
    if not spec.terms:
        print("   (none -- every step scores 0, nothing can be learned)")
    if not spec.use_data_json:
        from rl.vars import GameVars
        gv = GameVars(game, entry=spec.entry)
        blind = sorted({t["var"] for t in spec.terms if t.get("var")
                        and (gv.spec.get(t["var"]) or {}).get("source")
                        in ("info", "info16")})
        if blind:
            print("WARNING: without data.json these terms read `info` and will "
                  "never pay: %s" % ", ".join(blind))
            print("         give each a RAM address in games.json "
                  "(tools/find_game_vars.py finds them), or drop the term.")
        if not spec.episode_end:
            print("WARNING: no episode_end declared and the integration's done is "
                  "off, so episodes end only by truncation at max_steps.")


def hyper(args, spec):
    """PPO settings: game defaults from games.json, CLI wins."""
    p = {"n_steps": 512, "batch_size": 256, "learning_rate": 2.5e-4,
         "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01,
         "clip_range": 0.1, "n_epochs": 4}
    p.update(spec.ppo)
    for cli, key in (("n_steps", "n_steps"), ("batch_size", "batch_size"),
                     ("lr", "learning_rate"), ("gamma", "gamma"),
                     ("ent_coef", "ent_coef"), ("n_epochs", "n_epochs")):
        v = getattr(args, cli)
        if v is not None:
            p[key] = v
    return p


def train(args, spec, overrides):
    env = build_envs(args.game, overrides, args.n_envs, args.frame_stack)
    save_dir = args.save_dir or os.path.join("checkpoints", args.game)
    os.makedirs(save_dir, exist_ok=True)
    if args.resume and os.path.exists(args.resume):
        print("resuming from", args.resume)
        model = PPO.load(args.resume, env=env)
    else:
        p = hyper(args, spec)
        print("ppo:", p)
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=args.tb, **p)
    cb = CheckpointCallback(save_freq=max(1, args.save_every // args.n_envs),
                            save_path=save_dir, name_prefix="ckpt")
    model.learn(total_timesteps=args.timesteps, callback=cb,
                progress_bar=args.progress)
    final = os.path.join(save_dir, "final.zip")
    model.save(final)
    print("saved", final)
    env.close()


def play(args, spec, overrides):
    env = build_envs(args.game, overrides, 1, args.frame_stack)
    model = PPO.load(args.play, env=env)
    obs = env.reset()
    ep = steps = 0
    while ep < args.episodes:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, _r, done, info = env.step(action)
        steps += 1
        if done[0]:
            print("episode %d: %d steps  %s" % (ep, steps, info[0]))
            ep += 1
            steps = 0
    env.close()


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True, help="stable-retro game id, e.g. TetrisTime-Nes-v0")
    p.add_argument("--config", default=None, help="games.json to read the game's setup from (default: the repo's)")
    p.add_argument("--describe", action="store_true", help="Print what games.json says about this game and exit")
    p.add_argument("--list-states", action="store_true", help="Print the game's save states and exit")

    p.add_argument("--state", default=None, help="Save state every episode starts from (see --list-states)")
    p.add_argument("--player", type=int, default=None, help="Which player the agent drives")
    p.add_argument("--players", type=int, default=None, help="How many players the env runs")
    p.add_argument("--frameskip", type=int, default=None, help="Frames one decision is held for")
    p.add_argument("--max-steps", type=int, default=None, help="Truncate an episode after this many decisions")
    p.add_argument("--no-data-json", action="store_true",
                   help="Ignore retro's data.json entirely: no `info` values and no "
                        "scenario done signal, so the game is defined ONLY by this "
                        "repo's config -- RAM variables for every value and an "
                        "episode_end for the ending. Use it when you do not want to "
                        "depend on integration files somebody else has to write.")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="Override any field of the game's training block, repeatable")

    p.add_argument("--timesteps", type=int, default=200_000, help="Total decisions to train for, across all envs")
    p.add_argument("--n-envs", type=int, default=1, help="Parallel emulators (separate processes)")
    p.add_argument("--frame-stack", type=int, default=4, help="Frames stacked into one observation")
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--ent-coef", type=float, default=None)
    p.add_argument("--n-epochs", type=int, default=None)

    p.add_argument("--save-dir", default=None, help="Default: checkpoints/<game>")
    p.add_argument("--save-every", type=int, default=50_000, help="Checkpoint every N total timesteps")
    p.add_argument("--resume", default=None, help="Checkpoint to continue from")
    p.add_argument("--tb", default=None, help="TensorBoard log dir")
    p.add_argument("--progress", action="store_true")

    p.add_argument("--play", default=None, metavar="CHECKPOINT.zip", help="Watch a checkpoint instead of training")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--deterministic", action="store_true", help="Greedy actions when playing")
    args = p.parse_args()

    if args.config:
        rlvars.set_config_path(args.config)
    overrides = overrides_from(args)
    spec = TrainingSpec(args.game, overrides)

    if args.list_states:
        for n in spec.states_available():
            print(n)
        return
    if args.describe:
        describe(args.game, spec)
        return
    print("game %s | state %s | player %d/%d | %d actions"
          % (args.game, spec.state, spec.player, spec.players, len(spec.actions)))
    play(args, spec, overrides) if args.play else train(args, spec, overrides)


if __name__ == "__main__":
    main()
