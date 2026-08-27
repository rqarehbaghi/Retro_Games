#!/usr/bin/env python3
"""
Custom Gym Wrappers for Stable-Retro Reinforcement Learning:
- JumpIncentiveWrapper: Forces jump exploration within the first 100 iterations
- DeathPenaltyWrapper: Stops suicidal rushing into obstacles
- Velocity & Distance Progress Shaper
"""
import cv2
import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper, Wrapper
from gymnasium.spaces import Box, Discrete

class JumpIncentiveWrapper(Wrapper):
    """
    Solves the 'No jump attempts in 100 iterations' problem:
    1. Rewards the agent for exploring running jumps ('RIGHT' + 'A').
    2. Detects when horizontal progress is blocked (stuck on obstacle/pipe)
       and rewards triggering a jump to overcome the blockage.
    3. Rewards vertical airtime while moving horizontally.
    """
    def __init__(self, env, jump_bonus=0.2, stuck_penalty=0.05, jump_button="A"):
        super().__init__(env)
        self.jump_bonus = jump_bonus
        self.stuck_penalty = stuck_penalty
        self.jump_button = jump_button
        self.prev_x = 0
        self.stalled_frames = 0
        self.prev_y = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = info.get("x", info.get("x_pos", 0))
        self.prev_y = info.get("y", info.get("y_pos", 0))
        self.stalled_frames = 0
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        current_x = info.get("x", info.get("x_pos", None))
        current_y = info.get("y", info.get("y_pos", None))

        # Check if the agent is actively pressing the jump button or airborne
        buttons = getattr(self.env.unwrapped, "buttons", [])
        jump_idx = buttons.index(self.jump_button) if self.jump_button in buttons else None

        # Detect forward jump attempt
        if jump_idx is not None and isinstance(action, (list, np.ndarray)) and len(action) > jump_idx:
            if action[jump_idx]:
                reward += self.jump_bonus

        # Detect if horizontal progress is stalled (wall/obstacle collision)
        if current_x is not None and self.prev_x is not None:
            if abs(current_x - self.prev_x) < 0.5:
                self.stalled_frames += 1
                if self.stalled_frames > 8:
                    reward -= self.stuck_penalty  # Discourage standing/running into walls
            else:
                self.stalled_frames = 0
            self.prev_x = current_x

        # Bonus for vertical height / airtime while clearing
        if current_y is not None and self.prev_y is not None:
            if current_y > self.prev_y:
                reward += self.jump_bonus * 0.5
            self.prev_y = current_y

        return obs, reward, term, trunc, info

class DeathPenaltyWrapper(Wrapper):
    """
    Penalizes the agent when it dies or loses health, stopping the agent from
    blindly sprinting into the first enemy or pit just to rack up quick x-distance.
    """
    def __init__(self, env, penalty=50.0, survival_tick=0.01):
        super().__init__(env)
        self.penalty = penalty
        self.survival_tick = survival_tick
        self.prev_lives = None
        self.prev_health = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_lives = info.get("lives", None)
        self.prev_health = info.get("health", None)
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        # Encourage staying alive
        reward += self.survival_tick

        current_lives = info.get("lives", None)
        current_health = info.get("health", None)

        if self.prev_lives is not None and current_lives is not None:
            if current_lives < self.prev_lives:
                reward -= self.penalty
            self.prev_lives = current_lives

        if self.prev_health is not None and current_health is not None:
            if current_health < self.prev_health:
                reward -= (self.prev_health - current_health) * 2.0
            self.prev_health = current_health

        if (term or trunc) and not info.get("is_stage_clear", False):
            reward -= self.penalty

        return obs, reward, term, trunc, info

