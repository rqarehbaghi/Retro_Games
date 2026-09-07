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
  1. Opens an interactive pygame window reading your keyboard AND any USB
     gamepad SDL can see, for Player 1.
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

# Buttons are read by MEANING, not by index. SDL's GameController layer maps a
# recognised pad onto a standard layout (A/B/X/Y, START, BACK, DPAD_*) from its
# own per-GUID database. Guessing raw indices is what broke this: the previous
# table assumed START was button 6-9, but the pads here report
#   a=b0  b=b1  x=b3  y=b4  back=b10  start=b11  dpad=hat0
# so START was simply never bound and pressing Start at the 1P/2P menu did
# nothing, while the d-pad and face buttons appeared to work.
#
# NES has only A (jump) and B (run/fire), so both lower face buttons map to A
# and both upper ones to B -- the diamond a retro pad is shaped for.
# Only the D-PAD is taken from SDL's profile. Everything else is read by raw
# index, because SDL's profile for these pads is demonstrably wrong about them.
CONTROLLER_MAP = {
    "UP":     ("CONTROLLER_BUTTON_DPAD_UP",),
    "DOWN":   ("CONTROLLER_BUTTON_DPAD_DOWN",),
    "LEFT":   ("CONTROLLER_BUTTON_DPAD_LEFT",),
    "RIGHT":  ("CONTROLLER_BUTTON_DPAD_RIGHT",),
}

# FACE BUTTONS, by raw index. The pad is a SNES-shaped four-button diamond
# labelled X Y A B, which enumerates as b0=X, b1=A, b2=B, b3=Y. SDL's profile
# claims a=b0 b=b1 x=b3 y=b4 and skips b2 entirely -- so routing these through
# the GameController layer left the physical B button dead, which is exactly
# what was reported.
#
# The pairing is the NES-era convention: the console has only A and B, so a
# SNES-shaped pad wires the two spare buttons as turbo duplicates -- X repeats
# A, Y repeats B. Grouping X with A and Y with B keeps that.
PAD_BUTTONS = {0: "A", 1: "A", 2: "B", 3: "B"}

# START and SELECT, also by raw index -- SDL puts start=b11 / back=b10, which
# are not the buttons this pad labels START and SELECT.
PAD_SYSTEM_BUTTONS = {6: "SELECT", 7: "START", 8: "SELECT", 9: "START"}
PAD_DEADZONE = 0.5


class Pad:
    """One gamepad, read semantically when SDL recognises it."""

    def __init__(self, index):
        self.index = index
        self.ctrl = None
        self.joy = None
        try:
            from pygame._sdl2 import controller as sdl_controller
            sdl_controller.init()
            if sdl_controller.is_controller(index):
                self.ctrl = sdl_controller.Controller(index)
        except Exception:                                        # noqa: BLE001
            self.ctrl = None
        # The joystick view is opened ALONGSIDE the controller view, not just
        # as a fallback: START/SELECT are read by raw index on both paths.
        try:
            self.joy = pygame.joystick.Joystick(index)
            try:
                self.joy.init()
            except Exception:                                    # noqa: BLE001
                pass
        except Exception:                                        # noqa: BLE001
            self.joy = None

    def describe(self):
        if self.ctrl is not None:
            return (f"{self.ctrl.name} -- SDL profile for the d-pad, "
                    "raw index for face buttons and START/SELECT")
        if self.joy is not None:
            return (f"{self.joy.get_name()} -- {self.joy.get_numbuttons()} buttons, "
                    "raw index mapping (SDL has no profile for this pad)")
        return "unreadable pad"

    def apply(self, action, env_buttons):
        def press(name):
            if name in env_buttons:
                action[env_buttons.index(name)] = True

        if self.ctrl is not None:
            for nes_name, consts in CONTROLLER_MAP.items():
                for const in consts:
                    code = getattr(pygame, const, None)
                    if code is not None and self.ctrl.get_button(code):
                        press(nes_name)
                        break
            # SDL reports controller axes over the full int16 range.
            lx = getattr(pygame, "CONTROLLER_AXIS_LEFTX", 0)
            ly = getattr(pygame, "CONTROLLER_AXIS_LEFTY", 1)
            ax = self.ctrl.get_axis(lx) / 32768.0
            ay = self.ctrl.get_axis(ly) / 32768.0
        else:
            joy = self.joy
            if joy.get_numhats():
                hx, hy = joy.get_hat(0)
                if hx < 0:
                    press("LEFT")
                elif hx > 0:
                    press("RIGHT")
                if hy > 0:
                    press("UP")
                elif hy < 0:
                    press("DOWN")
            ax, ay = ((joy.get_axis(0), joy.get_axis(1))
                      if joy.get_numaxes() >= 2 else (0.0, 0.0))

        # Face buttons AND start/select by RAW index on BOTH paths: SDL's
        # profile mislabels the diamond and skips b2 (the physical B).
        if self.joy is not None:
            n = self.joy.get_numbuttons()
            for table in (PAD_BUTTONS, PAD_SYSTEM_BUTTONS):
                for index, name in table.items():
                    if index < n and self.joy.get_button(index):
                        press(name)

        # The analogue stick doubles as a d-pad on either path.
        if ax < -PAD_DEADZONE:
            press("LEFT")
        elif ax > PAD_DEADZONE:
            press("RIGHT")
        if ay < -PAD_DEADZONE:
            press("UP")
        elif ay > PAD_DEADZONE:
            press("DOWN")
        return action


def find_pads():
    """Every gamepad SDL can see, wrapped so buttons read by meaning."""
    # Joysticks cannot be enumerated before pygame is initialised; play_match
    # used to call this first, so get_count() returned 0 and pads were dropped.
    pygame.init()
    pygame.joystick.init()
    return [Pad(i) for i in range(pygame.joystick.get_count())]


def apply_pad(action, env_buttons, pad):
    """Fold a gamepad's current state into an action array."""
    return pad.apply(action, env_buttons)


# Player 2 on the keyboard uses the NUMERIC KEYPAD, which collides with none of
# Player 1's keys above -- so two people can share one keyboard when a second
# gamepad is not around. With two pads, P2 just uses pad 2 and ignores this.
KEY_MAPPING_P2 = {
    pygame.K_KP8: "UP",
    pygame.K_KP2: "DOWN",
    pygame.K_KP4: "LEFT",
    pygame.K_KP6: "RIGHT",
    pygame.K_KP7: "B",
    pygame.K_KP9: "A",
    pygame.K_KP0: "A",
    pygame.K_KP_ENTER: "START",
    pygame.K_KP_PLUS: "SELECT",
}


def keys_to_action(env_buttons, held_keys, key_map, pad=None):
    """A set of held pygame key codes (+ an optional pad) -> a button array.

    held_keys is a SET, tested with `in`. It used to be the sequence from
    pygame.key.get_pressed(), but that returns nothing while the window lacks
    focus and was unreliable under WSLg, so the caller now also tracks
    KEYDOWN/KEYUP events and passes the union."""
    action = np.array([False] * len(env_buttons), dtype=bool)
    for key, button_name in key_map.items():
        if key in held_keys and button_name in env_buttons:
            action[env_buttons.index(button_name)] = True
    if pad is not None:
        action = apply_pad(action, env_buttons, pad)
    return action


def make_p1_action(env_buttons, held_keys, pad=None):
    """Player 1: the arrow/WASD keyboard layout (KEY_MAPPING) plus pad 1."""
    return keys_to_action(env_buttons, held_keys, KEY_MAPPING, pad)


def make_p2_action(env_buttons, held_keys, pad=None):
    """Player 2 (second human): the numeric keypad (KEY_MAPPING_P2) plus pad 2."""
    return keys_to_action(env_buttons, held_keys, KEY_MAPPING_P2, pad)


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


class AudioStreamer:
    """Play the emulator's per-frame audio through pygame.mixer, in real time.

    stable-retro renders video to an array and hands back audio as (N, 2) int16
    samples per step via em.get_audio(), but nothing ever played them -- so the
    window was silent. This owns one mixer channel and keeps it fed. It is
    deliberately NON-FATAL: on a box with no audio device it disables itself and
    the session keeps running (silently) rather than crashing the game."""

    def __init__(self, rate):
        self.ok = False
        self.pending = []
        try:
            pygame.mixer.quit()      # drop the default 44.1k mixer pygame.init made
            pygame.mixer.init(frequency=int(round(rate)), size=-16, channels=2,
                              buffer=512)
            self.channel = pygame.mixer.Channel(0)
            self.ok = True
            print(f"Audio: on ({int(round(rate))} Hz).")
        except Exception as exc:                                  # noqa: BLE001
            print(f"Audio: off ({exc.__class__.__name__}: {exc}).")

    def feed(self, samples):
        """Queue one step's samples. While a block is still waiting to play it
        keeps accumulating, so the channel is handed one continuous block at a
        time instead of 60 tiny clips a second (which click)."""
        if not self.ok:
            return
        try:
            self.pending.append(np.ascontiguousarray(samples, dtype=np.int16))
            if self.channel.get_queue() is not None:
                return               # a block is already queued; pile onto it
            block = (self.pending[0] if len(self.pending) == 1
                     else np.concatenate(self.pending))
            self.pending = []
            snd = pygame.sndarray.make_sound(block)
            if self.channel.get_busy():
                self.channel.queue(snd)
            else:
                self.channel.play(snd)
        except Exception:                                        # noqa: BLE001
            self.ok = False          # never let audio kill the play session

    def close(self):
        try:
            pygame.mixer.quit()
        except Exception:                                        # noqa: BLE001
            pass


def play_match(game, state, model_path, record_dir, scale=3, fps_cap=60,
               mode="versus", players=2, boot_screen=False, p2_human=False,
               render_mp4=True):
    os.makedirs(record_dir, exist_ok=True)
    before_bk2s = set(glob.glob(os.path.join(record_dir, "*.bk2")))
    session_start = time.time()

    if boot_screen or (isinstance(state, str) and state.upper() == "NONE"):
        state_val = retro.State.NONE
    else:
        state_val = state or retro.State.DEFAULT

    # 1. Initialize stable-retro with 2 players
    try:
        env = retro.make(
            game=game,
            state=state_val,
            players=players,
            record=record_dir,
            render_mode="rgb_array",
        )
    except Exception as e:
        print(f"[Warning] Failed to initialize with players={players}: {e}")
        print("Falling back to standard 1-player environment...")
        env = retro.make(
            game=game,
            state=state_val,
            record=record_dir,
            render_mode="rgb_array",
        )

    # Cold boot (State.NONE) needs two things fixed up before reset(), and both
    # only bite when recording is on:
    #   - a state NAME: stable-retro builds the .bk2 filename from it inside
    #     reset(), so with None it dies on os.path.basename(None);
    #   - a starting STATE embedded in the movie -- the power-on snapshot. A
    #     .bk2 with no start state records fine but cannot be REPLAYED
    #     ("Could not load movie"), which breaks the studio pipeline
    #     downstream, since it replays the .bk2 to render and to scan events.
    if getattr(env.unwrapped, "statename", None) is None:
        env.unwrapped.statename = "boot"
    if not getattr(env.unwrapped, "initial_state", None):
        env.unwrapped.initial_state = env.unwrapped.em.get_state()

    obs, info = env.reset()
    buttons = env.unwrapped.buttons
    num_players = getattr(env.unwrapped, "players", 1)

    print(f"\n=== MATCH STARTED: {game} ===")
    print(f"Mode: {mode.upper()} | Active Players: {num_players}")
    print(f"Controller Buttons detected: {buttons}")
    pads = find_pads()
    pad1 = pads[0] if len(pads) >= 1 else None
    pad2 = pads[1] if len(pads) >= 2 else None
    for idx, pd in ((1, pad1), (2, pad2)):
        if pd is not None:
            print(f"Gamepad {idx}: {pd.describe()}")
    if pad1 is None:
        print("Gamepad: none detected, keyboard only.")
        print("         On WSL a USB pad is not visible until it is attached")
        print("         with usbipd-win from an admin PowerShell:")
        print("           usbipd list")
        print("           usbipd bind --busid <id>")
        print("           usbipd attach --wsl --busid <id>")
    if p2_human:
        print("TWO HUMANS -- no AI.")
        print("  Player 1: keyboard arrows + Z(B)/X(A), START=Enter"
              + ("  and Gamepad 1" if pad1 else ""))
        if pad2 is not None:
            print("  Player 2: Gamepad 2")
        else:
            print("  Player 2: numeric keypad (8/2/4/6 = move, 7=B, 9/0=A, KP-Enter=START)")
            print("            Plug in a SECOND controller for a proper two-pad game.")
    else:
        print(f"Human: PLAYER 1 (Keyboard/Gamepad) | AI: PLAYER 2 ({model_path or 'Random Policy'})")
    print("---------------------------------------------------------------")

    # 2. Load trained PPO model for Player 2
    model = None
    if not p2_human:
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
    caption = (f"Retro AI Arena: 2-Player Local - [{game}]" if p2_human
               else f"Retro AI Arena: Human (P1) vs AI (P2) - [{game}]")
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18, bold=True)
    audio = AudioStreamer(env.unwrapped.em.get_audio_rate())

    running = True
    step_count = 0
    p1_wins, p2_wins = 0, 0
    match_start_time = time.time()

    # Keys currently held, tracked from KEYDOWN/KEYUP events. get_pressed()
    # alone was the whole "input not recognized" bug: with no window focus it
    # returns all-False, so keyboard AND (through the same dead action) the
    # whole window felt unresponsive. Events are the robust path; the two are
    # unioned so whichever the platform actually delivers gets through.
    held_keys = set()

    while running:
        # Check Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    held_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                held_keys.discard(event.key)

        polled = pygame.key.get_pressed()
        combined = set(held_keys)
        combined.update(k for k in KEY_MAPPING if polled[k])
        p1_action = make_p1_action(buttons, combined, pad1)

        # Player 2: a second human (two controllers / hot-seat) or the AI.
        if p2_human:
            p2_action = make_p2_action(buttons, combined, pad2)
        elif model is not None:
            stacked_obs = np.array(frame_stack)  # shape (4, 84, 84)
            p2_discrete_action, _ = model.predict(stacked_obs, deterministic=True)
            p2_action = discretize_ai_action(int(p2_discrete_action), buttons)
        else:
            # No trained model: pick a random *valid* action from the same
            # table (a coherent combo), not independent per-button coin flips
            # -- the latter produces impossible inputs like LEFT+RIGHT.
            p2_action = discretize_ai_action(np.random.randint(len(AI_COMBOS)), buttons)

        # Combine actions based on player count.
        #
        # stable-retro wants ONE FLAT array of num_buttons * players, laid out
        # player-major (all of P1's buttons, then all of P2's) -- not a 2-D
        # (players, buttons) array. Stacking produced shape (2, 9), and
        # retro_env.action_to_array then sliced a row where it expected a
        # single button, so int(ap[i]) got a 9-element array:
        #   TypeError: only 0-dimensional arrays can be converted to Python scalars
        # The same flat player-major layout is what movie.get_key uses when a
        # 2-player recording is replayed (see studio.read_events).
        if num_players == 2:
            joint_action = np.concatenate([p1_action, p2_action])
        else:
            joint_action = p1_action

        # Step Emulator
        obs, reward, terminated, truncated, info = env.step(joint_action)
        step_count += 1
        audio.feed(env.unwrapped.em.get_audio())

        # Update Frame Stack for AI
        frame_stack.append(process_frame(obs))

        # Render Game Frame to Pygame Surface
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        frame_surface = pygame.transform.scale(frame_surface, (window_w, window_h))
        screen.blit(frame_surface, (0, 0))

        # Render Header / Scoreboard
        pygame.draw.rect(screen, (20, 24, 33), (0, window_h, window_w, 60))
        elapsed_sec = int(time.time() - match_start_time)
        # LIVE P1 input readout. If this shows "none" while you press keys, the
        # game is not getting your input -- click the window to give it focus,
        # or the pad is not seen (see the "Gamepad:" line printed at startup).
        # If it lights up (e.g. ['START']), input IS reaching the game and you
        # are just navigating the boot/title/menu, where only START advances.
        p1_pressed = [name for name, on in zip(buttons, p1_action) if on]
        if p1_pressed and not getattr(play_match, "_input_seen", False):
            play_match._input_seen = True
            print(f"[input OK] first P1 input detected: {p1_pressed}")
        if p2_human:
            p2_pressed = [name for name, on in zip(buttons, p2_action) if on]
            header = f"P1: {p1_pressed or '-'}     P2: {p2_pressed or '-'}     {elapsed_sec}s"
        else:
            header = (f"P1 INPUT: {p1_pressed or 'none'}   |   "
                      f"START = Enter / pad-Start   |   {elapsed_sec}s")
        status_text = font.render(header, True, (240, 240, 240))
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

    # stable-retro's env.close() does NOT finalize the movie -- the .bk2 is
    # only written when the Movie object is closed, which stop_record does.
    # Without this the file appears only later (when the env is garbage
    # collected), so the find_new_bk2 below saw nothing and no video rendered.
    if hasattr(env.unwrapped, "stop_record"):
        env.unwrapped.stop_record()
    audio.close()
    env.close()
    pygame.quit()

    # Find the recorded .bk2 file
    # Same session-aware detection as play_and_record: stable-retro reuses
    # -000000 numbering per process, so a rerun OVERWRITES the previous file
    # and path-membership alone would miss it.
    from play_and_record import find_new_bk2
    bk2_path = find_new_bk2(record_dir, before_bk2s, started_at=session_start)

    if bk2_path:
        print(f"\nMatch replay recorded to: {bk2_path}")
        if not render_mp4:
            return bk2_path
        print("Rendering synchronized MP4 video...")
        subprocess.run([sys.executable, "-m", "stable_retro.scripts.playback_movie", bk2_path], check=False)
        mp4_path = os.path.splitext(bk2_path)[0] + ".mp4"
        if os.path.exists(mp4_path):
            print(f"Exported HD Match Video: {mp4_path}")
            return mp4_path
        return bk2_path
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True, help="stable-retro game id, e.g. MortalKombatII-Genesis or StreetFighterIISNES")
    parser.add_argument("--state", default=None, help="Save state name (or NONE for cold boot)")
    parser.add_argument("--boot-screen", action="store_true", help="Start from cold boot / title screen (state=retro.State.NONE)")
    parser.add_argument("--two-human", action="store_true", help="Two human players (P1: keyboard/pad1, P2: numpad/pad2)")
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
        boot_screen=args.boot_screen,
        p2_human=args.two_human,
        players=2 if args.two_human else (2 if args.model else 1),
    )


if __name__ == "__main__":
    main()
