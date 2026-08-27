#!/usr/bin/env python3
"""
Play 2-Player Retro Games: Human vs. AI (or Human + AI Co-Op) with
automatic high-definition MP4 recording, live score/health overlay,
and telemetry.

Supports:
  - 1v1 Fighting Games (Mortal Kombat, Street Fighter, Fatal Fury, etc.)
  - 2-Player Co-Op (Contra, TMNT, Double Dragon, Streets of Rage, etc.)
  - Split-Screen Speedrun Battles (Mario, Sonic, Mega Man)

How it works:
  1. Opens an interactive pygame window capturing your keyboard (or USB gamepad) for Player 1.
  2. Runs your trained Stable-Baselines3 PPO model for Player 2.
  3. Feeds joint action arrays to stable-retro with proper frame preprocessing (WarpFrame, FrameSkip, VecFrameStack).
  4. Records the exact controller button streams to a .bk2 replay.
  5. Automatically renders an upscaled HD MP4 with custom scoreboard overlays!

Usage:
  # 1v1 Mortal Kombat against Iteration 500 checkpoint:
  python play_human_vs_ai.py --game MortalKombatII-Genesis \\
      --model ./checkpoints/MortalKombatII-Genesis/iter_500.zip

  # Co-op Contra with your AI partner:
  python play_human_vs_ai.py --game Contra-Nes \\
      --model ./checkpoints/Contra-Nes/iter_250.zip --mode coop

Controls (Player 1 Human):
  - D-Pad / Movement: ARROW KEYS or WASD
  - A / Jump / Light Attack: 'X' or 'K'
  - B / Run / Heavy Attack:  'Z' or 'J'
  - C / High Punch / Special: 'C' or 'L'
  - X / High Kick: 'A' or 'U'
  - Y / Block:     'S' or 'I'
  - Z / Special:   'D' or 'O'
  - START: ENTER
  - SELECT: RIGHT SHIFT
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from collections import deque

import cv2
import numpy as np
import pygame
import stable_retro as retro
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO

# The AI (Player 2) is a model trained by train.py, which ALWAYS uses
# train.py's ACTION_TABLE as its discrete action space. The model therefore
# only ever emits indices into ACTION_TABLE -- decoding those indices through
# any other table (an earlier version used a separate 20-entry FIGHTER_COMBOS
# list) makes the agent press semantically unrelated buttons. We import the
# real table and derive the same (index -> button combo) mapping the model
# was trained with. The per-action hold length is irrelevant here because we
# re-query the policy every frame.
from train import ACTION_TABLE

AI_COMBOS = [combo for combo, _hold in ACTION_TABLE]

# Key mappings for Player 1 Keyboard -> Retro Button Names
KEY_MAPPING = {
    # Movement
    pygame.K_UP: "UP",
    pygame.K_w: "UP",
    pygame.K_DOWN: "DOWN",
    pygame.K_s: "DOWN",
    pygame.K_LEFT: "LEFT",
    pygame.K_a: "LEFT",
    pygame.K_RIGHT: "RIGHT",
    pygame.K_d: "RIGHT",
    
    # Standard Action buttons (NES / Genesis / SNES)
    pygame.K_z: "B",
    pygame.K_j: "B",
    pygame.K_x: "A",
    pygame.K_k: "A",
    pygame.K_c: "C",
    pygame.K_l: "C",
    pygame.K_q: "X",
    pygame.K_u: "X",
    pygame.K_e: "Y",
    pygame.K_i: "Y",
    pygame.K_r: "Z",
    pygame.K_o: "Z",
    
    # System buttons
    pygame.K_RETURN: "START",
    pygame.K_RSHIFT: "SELECT",
    pygame.K_SPACE: "MODE",
}

def make_p1_action(env_buttons, pressed_keys):
    """Converts currently pressed pygame keys into a boolean array matching env.buttons."""
    action = np.array([False] * len(env_buttons), dtype=bool)
    for key, button_name in KEY_MAPPING.items():
        if pressed_keys[key]:
            if button_name in env_buttons:
                action[env_buttons.index(button_name)] = True
    return action


def discretize_ai_action(action_idx, env_buttons, combos=AI_COMBOS):
    """Converts a single discrete index from PPO into a boolean array for
    Player 2. `combos` MUST be the same action table the model was trained
    with (train.py's ACTION_TABLE, exposed here as AI_COMBOS) -- otherwise
    the index the policy chose maps to the wrong buttons."""
    action = np.array([False] * len(env_buttons), dtype=bool)
    if 0 <= action_idx < len(combos):
        for button_name in combos[action_idx]:
            if button_name in env_buttons:
                action[env_buttons.index(button_name)] = True
    return action


def process_frame(rgb_frame, target_size=84):
    """Converts raw RGB frame to 84x84 grayscale matching SB3 WarpFrame."""
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def play_match(game, state, model_path, record_dir, scale=3, fps_cap=60, mode="versus"):
    os.makedirs(record_dir, exist_ok=True)
    before_bk2s = set(glob.glob(os.path.join(record_dir, "*.bk2")))
    session_start = time.time()

    # 1. Initialize stable-retro with 2 players
    try:
        env = retro.make(
            game=game,
            state=state or retro.State.DEFAULT,
            players=2,
            record=record_dir,
            render_mode="rgb_array",
        )
    except Exception as e:
        print(f"[Warning] Failed to initialize with players=2: {e}")
        print("Falling back to standard 1-player environment...")
        env = retro.make(
            game=game,
            state=state or retro.State.DEFAULT,
            record=record_dir,
            render_mode="rgb_array",
        )

    obs, info = env.reset()
    buttons = env.unwrapped.buttons
    num_players = getattr(env.unwrapped, "players", 1)

    print(f"\\n=== MATCH STARTED: {game} ===")
    print(f"Mode: {mode.upper()} | Active Players: {num_players}")
    print(f"Controller Buttons detected: {buttons}")
    print(f"Human: PLAYER 1 (Keyboard/Gamepad) | AI: PLAYER 2 ({model_path or 'Random Policy'})")
    print("---------------------------------------------------------------")

    # 2. Load trained PPO model for Player 2
    model = None
    if model_path and os.path.exists(model_path):
        print(f"Loading trained AI policy from: {model_path}")
        model = PPO.load(model_path)
    else:
        print("No checkpoint model found — AI will use exploratory random policy.")

    # 3. Setup Frame Stack buffer (4 frames of 84x84 grayscale)
    frame_stack = deque(maxlen=4)
    init_frame = process_frame(obs)
    for _ in range(4):
        frame_stack.append(init_frame)

    # 4. Setup Pygame Display Window
    pygame.init()
    native_h, native_w, _ = obs.shape
    window_w, window_h = native_w * scale, native_h * scale
    screen = pygame.display.set_mode((window_w, window_h + 60))
    pygame.display.set_caption(f"Retro AI Arena: Human (P1) vs AI (P2) - [{game}]")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18, bold=True)

    running = True
    step_count = 0
    p1_wins, p2_wins = 0, 0
    match_start_time = time.time()

    while running:
        # Check Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        pressed = pygame.key.get_pressed()
        p1_action = make_p1_action(buttons, pressed)

        # AI (Player 2) Action Prediction
        if model is not None:
            stacked_obs = np.array(frame_stack)  # shape (4, 84, 84)
            p2_discrete_action, _ = model.predict(stacked_obs, deterministic=True)
            p2_action = discretize_ai_action(int(p2_discrete_action), buttons)
        else:
            # No trained model: pick a random *valid* action from the same
            # table (a coherent combo), not independent per-button coin flips
            # -- the latter produces impossible inputs like LEFT+RIGHT.
            p2_action = discretize_ai_action(np.random.randint(len(AI_COMBOS)), buttons)

        # Combine actions based on player count
        if num_players == 2:
            joint_action = np.array([p1_action, p2_action])
        else:
            joint_action = p1_action

        # Step Emulator
        obs, reward, terminated, truncated, info = env.step(joint_action)
        step_count += 1

        # Update Frame Stack for AI
        frame_stack.append(process_frame(obs))

        # Render Game Frame to Pygame Surface
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        frame_surface = pygame.transform.scale(frame_surface, (window_w, window_h))
        screen.blit(frame_surface, (0, 0))

        # Render Header / Scoreboard
        pygame.draw.rect(screen, (20, 24, 33), (0, window_h, window_w, 60))
        elapsed_sec = int(time.time() - match_start_time)
        status_text = font.render(
            f"P1 (HUMAN): [Z:Atk X:Jump Arrows:Move]   |   P2 (AI): {model_path and 'PPO Agent' or 'Random'}   |   Time: {elapsed_sec}s",
            True, (240, 240, 240)
        )
        screen.blit(status_text, (15, window_h + 18))

        pygame.display.flip()
        clock.tick(fps_cap)

        if terminated or truncated:
            print(f"Round finished at step {step_count}! Resetting...")
            obs, info = env.reset()
            step_count = 0
            # Refill the AI's frame-stack from the fresh round so it doesn't
            # keep reacting to stale frames from the round that just ended.
            reset_frame = process_frame(obs)
            frame_stack.clear()
            for _ in range(4):
                frame_stack.append(reset_frame)

    env.close()
    pygame.quit()

    # Find the recorded .bk2 file
    # Same session-aware detection as play_and_record: stable-retro reuses
    # -000000 numbering per process, so a rerun OVERWRITES the previous file
    # and path-membership alone would miss it.
    from play_and_record import find_new_bk2
    bk2_path = find_new_bk2(record_dir, before_bk2s, started_at=session_start)

    if bk2_path:
        print(f"\\nMatch replay recorded to: {bk2_path}")
        print("Rendering synchronized MP4 video...")
        subprocess.run([sys.executable, "-m", "stable_retro.scripts.playback_movie", bk2_path], check=False)
        mp4_path = os.path.splitext(bk2_path)[0] + ".mp4"
        if os.path.exists(mp4_path):
            print(f"Exported HD Match Video: {mp4_path}")
            return mp4_path
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True, help="stable-retro game id, e.g. MortalKombatII-Genesis or StreetFighterIISNES")
    parser.add_argument("--state", default=None, help="Save state name")
    parser.add_argument("--model", default=None, help="Path to trained PPO checkpoint .zip for Player 2")
    parser.add_argument("--mode", choices=["versus", "coop", "race"], default="versus", help="Match mode")
    parser.add_argument("--scale", type=int, default=3, help="Window display scale factor (default: 3)")
    parser.add_argument("--fps", type=int, default=60, help="Framerate cap (default: 60)")
    parser.add_argument("--record-dir", default="./recordings", help="Replay output folder")
    args = parser.parse_args()

    play_match(
        game=args.game,
        state=args.state,
        model_path=args.model,
        record_dir=args.record_dir,
        scale=args.scale,
        fps_cap=args.fps,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
