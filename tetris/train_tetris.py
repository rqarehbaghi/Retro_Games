#!/usr/bin/env python3
"""
Train a bot to play TetrisTime as PLAYER 2, and watch a checkpoint play.

    # train (checkpoints land in ./checkpoints_tetris)
    python tetris/train_tetris.py --timesteps 500000 --n-envs 4

    # watch what a checkpoint does
    python tetris/train_tetris.py --play checkpoints_tetris/tetris_100000_steps.zip

Why SubprocVecEnv rather than DummyVecEnv: stable-retro allows exactly ONE
emulator per process, so parallel environments have to be separate processes.
With --n-envs 1 it stays in-process.

What to expect. This is Tetris from pixels, which is a hard exploration problem:
line clears are rare enough that a policy learning from them alone gets almost
no gradient. The board shaping in tetris_env (holes, height, bumpiness) is what
makes early progress visible -- watch `holes` and `max_height` in the rollout
stats before you look at `lines`. Survival time should climb first, clears much
later.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO                                # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor             # noqa: E402
from stable_baselines3.common.vec_env import (                   # noqa: E402
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage)

from tetris.tetris_env import make_tetris_env                    # noqa: E402


def _factory(seed, **kw):
    def _init():
        env = make_tetris_env(**kw)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def build_envs(n_envs, **kw):
    fns = [_factory(i, **kw) for i in range(n_envs)]
    # One emulator per PROCESS: more than one env means more than one process.
    vec = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    vec = VecFrameStack(vec, n_stack=4)   # the agent needs to see motion
    return VecTransposeImage(vec)


def train(args):
    env = build_envs(args.n_envs, player=args.player, frameskip=args.frameskip)
    if args.resume and os.path.exists(args.resume):
        print("resuming from", args.resume)
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "CnnPolicy", env, verbose=1,
            n_steps=args.n_steps, batch_size=args.batch_size,
            learning_rate=args.lr, gamma=0.99, gae_lambda=0.95,
            ent_coef=0.01,           # Tetris needs exploration; clears are rare
            clip_range=0.1, n_epochs=4,
            tensorboard_log=args.tb,
        )
    os.makedirs(args.save_dir, exist_ok=True)
    cb = CheckpointCallback(save_freq=max(1, args.save_every // args.n_envs),
                            save_path=args.save_dir, name_prefix="tetris")
    model.learn(total_timesteps=args.timesteps, callback=cb,
                progress_bar=args.progress)
    final = os.path.join(args.save_dir, "tetris_final.zip")
    model.save(final)
    print("saved", final)
    env.close()


def play(args):
    env = build_envs(1, player=args.player, frameskip=args.frameskip)
    model = PPO.load(args.play, env=env)
    obs = env.reset()
    ep, steps, best = 0, 0, {}
    while ep < args.episodes:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        steps += 1
        if done[0]:
            i = info[0]
            print("episode %d: %d steps  score=%s lines=%s holes=%s max_h=%s"
                  % (ep, steps, i.get("score"), i.get("lines"),
                     i.get("holes"), i.get("max_height")))
            ep += 1
            steps = 0
    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=1, help="parallel emulators (separate processes)")
    p.add_argument("--player", type=int, default=2, help="which player the bot drives (default: %(default)s)")
    p.add_argument("--frameskip", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--save-dir", default="./checkpoints_tetris")
    p.add_argument("--save-every", type=int, default=50_000)
    p.add_argument("--resume", default=None)
    p.add_argument("--tb", default=None, help="tensorboard log dir")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--play", default=None, metavar="CHECKPOINT.zip")
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()
    play(args) if args.play else train(args)


if __name__ == "__main__":
    main()
