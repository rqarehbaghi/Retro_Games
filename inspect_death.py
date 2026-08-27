#!/usr/bin/env python3
"""
Diagnostic: watch which info-dict variable actually changes when the agent
dies, so reward shaping / episodic-life can key off the right one.

Some stable-retro integrations expose a 'lives' variable that doesn't
actually update on death -- wrong RAM address, BCD encoding, or it only
changes once you're kicked to the world map. When that happens, a
life-loss death penalty and "end episode on life loss" both silently do
nothing. Run this, let the agent play until it dies, and read which field
changes at the moment of death in the terminal.

It prints a line only when one of (lives, time, score, terminated) changes
-- not every frame -- so the death moment stands out. Correlate it with the
window (with --render) to see exactly what the game reports on death.

Usage (random agent, headless -- may take a while to die):
    python inspect_death.py --game SuperMarioBros3-Nes-v0

    # Driven by your pretrained/trained model so it actually reaches a death,
    # in a visible window:
    python inspect_death.py --game SuperMarioBros3-Nes-v0 \\
        --model ./checkpoints/SuperMarioBros3-Nes-v0/pretrained_human_bc1.zip --render
"""
import argparse

import stable_retro as retro

TRACK = ("lives", "hpos", "time", "score")


def summary(info, done):
    # Change-detection key: ignore hpos (changes every frame) so we only
    # print on the interesting transitions -- death, timer reset, score.
    return (info.get("lives"), info.get("time"), info.get("score"), bool(done))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True)
    p.add_argument("--state", default=None)
    p.add_argument("--model", default=None, help="Optional trained/pretrained PPO checkpoint so the agent actually reaches a death")
    p.add_argument("--render", action="store_true", help="Open a window (needs a display)")
    p.add_argument("--max-steps", type=int, default=100000)
    args = p.parse_args()

    render_mode = "human" if args.render else "rgb_array"

    if args.model:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

        from train import ACTION_TABLE, VariableHoldDiscretizer, WarpFrame

        base = retro.make(game=args.game, state=args.state or retro.State.DEFAULT, render_mode=render_mode)
        wrapped = WarpFrame(VariableHoldDiscretizer(base, ACTION_TABLE))
        venv = VecFrameStack(DummyVecEnv([lambda: wrapped]), n_stack=4)
        model = PPO.load(args.model)

        obs = venv.reset()
        prev = None
        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            i = info[0]
            key = summary(i, done[0])
            if key != prev:
                shown = {k: i.get(k) for k in TRACK}
                print(f"step {step:6d} terminated={bool(done[0])!s:5} {shown}")
                prev = key
            if done[0]:
                print("    >>> the env reported the episode ENDED and auto-reset <<<")
        venv.close()
    else:
        env = retro.make(game=args.game, state=args.state or retro.State.DEFAULT, render_mode=render_mode)
        obs, info = env.reset()
        prev = None
        for step in range(args.max_steps):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            key = summary(info, done)
            if key != prev:
                shown = {k: info.get(k) for k in TRACK}
                print(f"step {step:6d} term={terminated} trunc={truncated} {shown}")
                prev = key
            if args.render:
                env.render()
            if done:
                print("    >>> the env reported the episode ENDED <<<")
                obs, info = env.reset()
        env.close()


if __name__ == "__main__":
    main()
