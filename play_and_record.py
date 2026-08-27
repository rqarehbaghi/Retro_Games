#!/usr/bin/env python3
"""
Play an NES (or any stable-retro-supported) game -- either an AI agent or
you with a keyboard -- and export a finished MP4 (video + audio, synced)
of the session.

How it works:
  1. stable-retro plays the game and records button presses to a tiny
     .bk2 replay file (built-in feature, not custom code).
  2. Once the episode ends, stable-retro's own playback tool re-runs the
     replay through the emulator and encodes it to .mp4 via ffmpeg.

This means the "video capture" is exact and deterministic -- it's not a
screen-grab, it's the emulator re-rendering the actual run.

Examples:
    # Random agent (no trained model) plays until game over, then exports mp4
    python play_and_record.py --game SuperMarioBros-Nes

    # Use a trained Stable-Baselines3 model, start from a specific save state
    python play_and_record.py --game SuperMarioBros-Nes --state Level1-1 \\
        --model ppo_mario.zip --max-steps 5400

    # You play it yourself with a keyboard (needs a display -- see README)
    python play_and_record.py --game SuperMarioBros-Nes --human
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import stable_retro as retro


def build_policy(model_path, action_space):
    """Return a callable obs -> action for the RAW (no --model) path.

    Falls back to random actions if no model is given, so this script
    works out of the box on any imported game with zero training. The
    --model path is handled separately in play_agent_episode(), since it
    needs the env itself wrapped to match training, not just the policy
    function.
    """
    def policy(obs):
        return action_space.sample()
    return policy


def play_agent_episode(game, state, model_path, max_steps, record_dir, render):
    # stable-retro defaults to render_mode='human', which pulls in pyglet/OpenGL
    # and opens a window on env.reset() -- that needs GLU + a display, which a
    # headless WSL2/EC2 box doesn't have. Only ask for 'human' when --render was
    # explicitly passed; otherwise stay fully headless ('rgb_array').
    render_mode = "human" if render else "rgb_array"
    base_env = retro.make(
        game=game,
        state=state or retro.State.DEFAULT,
        record=record_dir,
        render_mode=render_mode,
    )

    if model_path:
        # A trained checkpoint expects EXACTLY the preprocessing it was
        # trained with (see train.py): the same duration-aware discrete
        # actions, grayscale 84x84 frames, and 4 frames stacked. Raw
        # emulator pixels don't even match the network's expected shape.
        # Reward-shaping wrappers (RewardShaper, JumpIncentiveWrapper)
        # are deliberately NOT applied here -- they only affect the
        # reward signal used during training, not the observation or
        # action space, so they make no difference to how a trained
        # model plays and would only distort the reward number printed
        # at the end.
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
        from train import ACTION_TABLE, VariableHoldDiscretizer, WarpFrame

        wrapped = VariableHoldDiscretizer(base_env, ACTION_TABLE)
        wrapped = WarpFrame(wrapped)
        env = VecFrameStack(DummyVecEnv([lambda: wrapped]), n_stack=4)

        model = PPO.load(model_path)

        obs = env.reset()
        steps = 0
        frames = 0
        total_reward = 0.0
        start = time.time()

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            frames += info[0].get("frames_this_step", 1)

            if render:
                env.render()

            if done[0] or frames >= max_steps:
                break

        elapsed = time.time() - start
        env.close()

    else:
        env = base_env
        policy = build_policy(model_path, env.action_space)
        obs, info = env.reset()

        steps = 0
        total_reward = 0.0
        start = time.time()

        while True:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if render:
                env.render()

            if terminated or truncated or steps >= max_steps:
                break

        elapsed = time.time() - start
        env.close()

    print(
        f"Episode finished: {steps} steps, reward={total_reward:.2f}, "
        f"wall-clock={elapsed:.1f}s"
    )


def play_human_episode(game, state, record_dir):
    """Hands off to stable-retro's own interactive tool for the actual
    keyboard-to-controller mapping (console-aware, already correct for
    every system stable-retro supports) rather than reimplementing it."""
    cmd = [
        sys.executable, "-m", "stable_retro.examples.interactive",
        "--game", game,
        "--record", record_dir,
    ]
    if state:
        cmd += ["--state", state]

    print("Launching interactive play window -- close it (or reach game over) when you're done.")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\nThe interactive tool exited with an error.")
        print("If it complained about an unrecognized --record flag, check its actual")
        print("options with:")
        print("    python3 -m stable_retro.examples.interactive --help")
        sys.exit(result.returncode)


def find_new_bk2(record_dir, before):
    """The .bk2 stable-retro just wrote is the one that wasn't there before.

    Returns None if no NEW file appeared. We deliberately do NOT fall back to
    the newest pre-existing .bk2: if this run failed to record, rendering a
    stale replay from a previous session as if it were this one is worse than
    reporting the failure."""
    candidates = glob.glob(os.path.join(record_dir, "*.bk2"))
    new_files = [f for f in candidates if f not in before]
    if not new_files:
        return None
    return max(new_files, key=os.path.getmtime)


def render_to_mp4(bk2_path):
    """Calls stable-retro's built-in playback script. Requires ffmpeg on PATH.
    Writes a .mp4 next to the .bk2 with video and audio synced.

    Note: this writes at the emulator's native resolution (NES: ~256x224) --
    stable-retro's playback tool has no scaling option. See upscale_mp4()
    for making the result not look tiny/blurry on a modern screen."""
    print(f"Rendering {bk2_path} to MP4 (this replays the run through the emulator)...")
    subprocess.run(
        [sys.executable, "-m", "stable_retro.scripts.playback_movie", bk2_path],
        check=True,
    )
    mp4_path = os.path.splitext(bk2_path)[0] + ".mp4"
    return mp4_path if os.path.exists(mp4_path) else None


def upscale_mp4(mp4_path, factor, mode):
    """Re-encodes mp4_path at factor-x its native resolution, writing
    <name>_HD.mp4 next to it. This doesn't add real detail (NES only ever
    rendered at ~256x224) -- it just controls HOW that gets stretched to
    fill a modern screen, instead of leaving it to whatever blurry default
    scaling your video player applies.

    mode='sharp'  -> nearest-neighbor: crisp, blocky retro-pixel look
    mode='smooth' -> lanczos: soft anti-aliased upscale, less blocky
    """
    algo = "neighbor" if mode == "sharp" else "lanczos"
    out_path = os.path.splitext(mp4_path)[0] + "_HD.mp4"

    print(f"Upscaling {factor}x ({mode}) -> {out_path} ...")
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-i", mp4_path,
            "-vf", f"scale=iw*{factor}:ih*{factor}:flags={algo}",
            "-c:a", "copy",
            "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
            out_path,
        ],
        check=True,
    )
    return out_path if os.path.exists(out_path) else None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--game", required=True, help="stable-retro game id, e.g. SuperMarioBros-Nes (see list_games.py)")
    parser.add_argument("--state", default=None, help="Save state to start from (default state if omitted)")
    parser.add_argument("--human", action="store_true", help="You play with a keyboard (opens a window) instead of an agent. Needs a display.")
    parser.add_argument("--model", default=None, help="Path to a trained Stable-Baselines3 PPO model (.zip). Ignored with --human. Omit for random play.")
    parser.add_argument("--max-steps", type=int, default=10800, help="Safety cap in emulator frames for agent play (60fps NES -> 10800 = ~3 min). Ignored with --human -- that runs until you close the window. (default: %(default)s)")
    parser.add_argument("--record-dir", default="./recordings", help="Where .bk2/.mp4 files land (default: %(default)s)")
    parser.add_argument("--render", action="store_true", help="Also show a live window while an agent plays (slower). Ignored with --human, which always shows a window. Needs a display.")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor for the final video, e.g. 4 turns ~256x224 into ~1024x896. Set to 1 to skip upscaling and keep the native-resolution file. (default: %(default)s)")
    parser.add_argument("--scale-mode", choices=["sharp", "smooth"], default="sharp", help="'sharp' = crisp nearest-neighbor (retro pixel look). 'smooth' = anti-aliased lanczos (softer, less blocky). (default: %(default)s)")
    args = parser.parse_args()

    os.makedirs(args.record_dir, exist_ok=True)
    before = set(glob.glob(os.path.join(args.record_dir, "*.bk2")))

    if args.human:
        play_human_episode(game=args.game, state=args.state, record_dir=args.record_dir)
    else:
        play_agent_episode(
            game=args.game,
            state=args.state,
            model_path=args.model,
            max_steps=args.max_steps,
            record_dir=args.record_dir,
            render=args.render,
        )

    bk2_path = find_new_bk2(args.record_dir, before)
    if bk2_path is None:
        print("No .bk2 recording found -- something went wrong with recording.")
        sys.exit(1)

    mp4_path = render_to_mp4(bk2_path)
    if not mp4_path:
        print("\nbk2 saved but MP4 render failed -- check that ffmpeg is installed and on PATH.")
        print(f"Raw replay file: {bk2_path}")
        return

    if args.scale > 1:
        final_path = upscale_mp4(mp4_path, args.scale, args.scale_mode)
        if final_path:
            print(f"\nDone! Video ready at: {final_path}")
            print(f"(native-resolution version also kept at: {mp4_path})")
        else:
            print("\nUpscale step failed -- native-resolution video is still fine:")
            print(f"Video ready at: {mp4_path}")
    else:
        print(f"\nDone! Video ready at: {mp4_path}")


if __name__ == "__main__":
    main()
