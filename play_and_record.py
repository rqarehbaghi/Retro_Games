#!/usr/bin/env python3
"""
Play an NES (or any stable-retro-supported) game -- either an AI agent or
you with a keyboard -- and export a finished MP4 (video + audio, synced)
of the session.

Fixed: Observation shape wrappers automatically applied when --model is loaded,
matching train.py's WarpFrame, FrameSkip, Discretizer, and VecFrameStack.

Examples:
    # Random agent (no trained model) plays until game over, then exports mp4
    python play_and_record.py --game SuperMarioBros-Nes

    # Use a trained Stable-Baselines3 model, start from a specific save state
    python play_and_record.py --game SuperMarioBros3-Nes-v0 --state Level1-1 \\
        --model ./checkpoints/SuperMarioBros3-Nes-v0/iter_100.zip --max-steps 5400

    # You play it yourself with a keyboard (needs a display -- see README)
    python play_and_record.py --game SuperMarioBros3-Nes-v0 --human
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import stable_retro as retro
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from train import DEFAULT_COMBOS, Discretizer, FrameSkip, WarpFrame


def play_agent_episode(game, state, model_path, max_steps, record_dir, render):
    render_mode = "human" if render else "rgb_array"
    
    if model_path is None:
        # Fallback for random actions (raw environment)
        env = retro.make(
            game=game,
            state=state or retro.State.DEFAULT,
            record=record_dir,
            render_mode=render_mode,
        )
        obs, info = env.reset()

        steps = 0
        total_reward = 0.0
        start = time.time()

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if render:
                env.render()

            if terminated or truncated or steps >= max_steps:
                break

        elapsed = time.time() - start
        env.close()

    else:
        # Trained model requires the exact preprocessing pipeline (Discretizer, WarpFrame, FrameSkip, VecFrameStack)
        from stable_baselines3 import PPO

        def make_env():
            e = retro.make(
                game=game,
                state=state or retro.State.DEFAULT,
                record=record_dir,
                render_mode=render_mode,
            )
            e = Discretizer(e, DEFAULT_COMBOS)
            e = WarpFrame(e)
            e = FrameSkip(e, skip=4)
            return e

        venv = DummyVecEnv([make_env])
        venv = VecFrameStack(venv, n_stack=4)

        model = PPO.load(model_path)
        obs = venv.reset()

        steps = 0
        total_reward = 0.0
        start = time.time()

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            total_reward += reward[0]
            steps += 1

            if render:
                venv.render()

            if dones[0] or steps >= (max_steps // 4):
                break

        elapsed = time.time() - start
        venv.close()

    print(
        f"Episode finished: {steps} steps, reward={total_reward:.2f}, "
        f"wall-clock={elapsed:.1f}s"
    )


def play_human_episode(game, state, record_dir):
    """Hands off to stable-retro's own interactive tool for keyboard-to-controller mapping."""
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
        sys.exit(result.returncode)


def find_new_bk2(record_dir, before):
    """Finds newly created .bk2 replay file."""
    candidates = glob.glob(os.path.join(record_dir, "*.bk2"))
    new_files = [f for f in candidates if f not in before]
    if new_files:
        return max(new_files, key=os.path.getmtime)
    return max(candidates, key=os.path.getmtime) if candidates else None


def render_to_mp4(bk2_path):
    """Renders .bk2 recording to native .mp4 using stable-retro playback."""
    print(f"Rendering {bk2_path} to MP4 (this replays the run through the emulator)...")
    subprocess.run(
        [sys.executable, "-m", "stable_retro.scripts.playback_movie", bk2_path],
        check=True,
    )
    mp4_path = os.path.splitext(bk2_path)[0] + ".mp4"
    return mp4_path if os.path.exists(mp4_path) else None


def upscale_mp4(mp4_path, factor, mode):
    """Upscales native resolution to HD using nearest-neighbor (sharp) or lanczos (smooth)."""
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
    parser.add_argument("--game", required=True, help="stable-retro game id, e.g. SuperMarioBros3-Nes-v0")
    parser.add_argument("--state", default=None, help="Save state to start from")
    parser.add_argument("--human", action="store_true", help="You play with a keyboard instead of an agent.")
    parser.add_argument("--model", default=None, help="Path to trained PPO model (.zip).")
    parser.add_argument("--max-steps", type=int, default=10800, help="Safety cap in emulator frames.")
    parser.add_argument("--record-dir", default="./recordings", help="Where .bk2/.mp4 files land.")
    parser.add_argument("--render", action="store_true", help="Show live window while agent plays.")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor for final video (default: 4).")
    parser.add_argument("--scale-mode", choices=["sharp", "smooth"], default="sharp", help="'sharp' or 'smooth'.")
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
        print("No .bk2 recording found.")
        sys.exit(1)

    mp4_path = render_to_mp4(bk2_path)
    if not mp4_path:
        print("\nbk2 saved but MP4 render failed.")
        return

    if args.scale > 1:
        final_path = upscale_mp4(mp4_path, args.scale, args.scale_mode)
        if final_path:
            print(f"\nDone! Video ready at: {final_path}")
        else:
            print(f"Video ready at: {mp4_path}")
    else:
        print(f"\nDone! Video ready at: {mp4_path}")


if __name__ == "__main__":
    main()
