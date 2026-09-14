"""Macro-action execution for turn-based, round-based, and grid-placement games.

Continuous physics games (Super Mario, Mega Man, Contra) require micro-step
button streams to control variable jump height, inertia, and sub-second reactions.

Decision-based games (Tetris, Dr. Mario, Yoshi, Columns) do not benefit from
micro-button mashing. Instead, a placement is a single strategic decision:
(rotation, column). This module translates that discrete decision into the
required sequence of controller taps and soft-drops until the piece settles.
"""

try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        class _GymFallback:
            class Wrapper:
                def __init__(self, env):
                    self.env = env
            class spaces:
                class Discrete:
                    def __init__(self, n):
                        self.n = n
        gym = _GymFallback()


class MacroPlanGenerator:
    """Generates frame-by-frame button sequences for grid placement games."""

    @staticmethod
    def plan(action, config, current_rot=0, current_col=None, piece_type=None):
        rotations = int(config.get("rotations", 4))
        columns = int(config.get("columns", 10))
        target_rot = int(action) // columns
        target_col = int(action) % columns

        rotate_button = config.get("rotate_button", "A")
        left_button = config.get("left_button", "LEFT")
        right_button = config.get("right_button", "RIGHT")
        commit_button = config.get("commit_button", "DOWN")
        tap_hold = int(config.get("tap_hold", 2))
        tap_release = int(config.get("tap_release", 2))
        spawn_col = int(config.get("spawn_col", 4))

        if current_col is None:
            current_col = spawn_col

        frames = []

        # 1. Rotations (tap the rotate button until the counter reads target)
        #
        # How far ONE tap moves the RAM rotation counter is a property of the
        # game, not an assumption. Measured on TetrisTime: tapping A ran the
        # counter 0 -> 3 -> 2 -> 1 -> 0, so it DECREMENTS. The taps needed are
        # therefore (current - target) mod 4, and the old (target - current)
        # asked for 3 taps where 1 was wanted. It went unnoticed because a
        # piece always spawns at rotation 0, which makes the error a fixed
        # permutation of the action space that a policy can learn around -- but
        # only until a piece is already rotated when a plan is made.
        rot_step = int(config.get("rotate_step", -1)) or -1
        rot_taps = ((target_rot - (current_rot or 0)) * rot_step) % rotations
        for _ in range(rot_taps):
            for _ in range(tap_hold):
                frames.append([rotate_button])
            for _ in range(tap_release):
                frames.append([])

        # 2. Horizontal placement (tap LEFT or RIGHT)
        offsets = config.get("piece_col_offsets", {}).get(str(piece_type))
        origin_shift = int(offsets[target_rot]) - int(config.get("col_offset", 0)) if offsets is not None else 0
        col_diff = target_col - current_col + origin_shift
        if col_diff < 0:
            btn = left_button
            taps = abs(col_diff)
        elif col_diff > 0:
            btn = right_button
            taps = col_diff
        else:
            btn = None
            taps = 0

        if btn is not None:
            for _ in range(taps):
                for _ in range(tap_hold):
                    frames.append([btn])
                for _ in range(tap_release):
                    frames.append([])

        return {
            "target_rot": target_rot,
            "target_col": target_col,
            "frames": frames,
            "commit_button": commit_button,
        }


class MacroPlacementWrapper(gym.Wrapper):
    """Turns micro-step button mashing into strategic macro placement decisions.

    Action space: Discrete(rotations * columns).
    Each step executes the full piece placement:
    1. Taps rotate_button to target rotation.
    2. Taps left/right to target column.
    3. Soft-drops (commit_button) until settle_detect triggers (piece locks and next piece spawns).
    4. Computes board delta rewards and returns the clean resting observation for the next piece.
    """

    def __init__(self, env, config, vars, player=1):
        super().__init__(env)
        self.cfg = dict(config or {})
        self.vars = vars
        self.player = player
        self.rotations = int(self.cfg.get("rotations", 4))
        self.columns = int(self.cfg.get("columns", 10))
        self.action_space = gym.spaces.Discrete(self.rotations * self.columns)

        self.rotate_button = self.cfg.get("rotate_button", "A")
        self.left_button = self.cfg.get("left_button", "LEFT")
        self.right_button = self.cfg.get("right_button", "RIGHT")
        self.commit_button = self.cfg.get("commit_button", "DOWN")
        self.settle_detect = dict(self.cfg.get("settle_detect") or {})

        self.tap_hold = int(self.cfg.get("tap_hold", 2))
        self.tap_release = int(self.cfg.get("tap_release", 2))
        self.max_settle_frames = int(self.cfg.get("max_settle_frames", 600))
        self.col_offset = int(self.cfg.get("col_offset", 3))
        self.spawn_col = int(self.cfg.get("spawn_col", 4))

        self.rot_var = self.cfg.get("rot_var", "piece_rot_p%d" % player)
        self.col_var = self.cfg.get("col_var", "piece_col_p%d" % player)
        self.row_var = self.settle_detect.get("var", self.cfg.get("row_var", "piece_row_p%d" % player))
        self.piece_type_var = self.cfg.get("piece_type_var", "piece_type_p%d" % player)

        # Opt-in frame capture for the visual unit test (train.py --unittest).
        # Off in normal training; when on, step() records the rendered frame at
        # the moment the piece LOCKS, which is otherwise gone by the time step()
        # returns (the emulator has run on to spawn the next piece).
        self.capture_lock = False
        self.lock_frame = None

    def _render_frame(self):
        try:
            return self.env.render()
        except Exception:
            return None

    def _ram(self):
        if hasattr(self.env, "_ram"):
            return self.env._ram()
        unwrapped = getattr(self.env, "unwrapped", self.env)
        if hasattr(unwrapped, "get_ram"):
            return unwrapped.get_ram()
        return getattr(self.env, "get_ram", lambda: None)()

    @property
    def spec_(self):
        return getattr(self.env, "spec_", None)

    def _read_var(self, name, default=0):
        if not name or self.vars is None:
            return default
        ram = self._ram()
        if ram is None:
            return default
        v = self.vars.read(name, ram, {}, default)
        return default if v is None else int(v)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Execute at most one placement. None waits without issuing controls."""
        current_rot = self._read_var(self.rot_var, 0)
        col_raw = self._read_var(self.col_var, None)
        current_col = col_raw - self.col_offset if col_raw is not None else self.spawn_col
        initial_type = self._read_var(self.piece_type_var, None)
        peak_row = self._read_var(self.row_var, None)
        min_row = int(self.settle_detect.get("min_previous", 0))
        valid_types = self.cfg.get("valid_piece_types")
        completed = False
        respawned = False
        env_term = env_trunc = False
        obs, info = None, {}
        self.lock_frame = None

        def tick(buttons):
            nonlocal obs, info, env_term, env_trunc, completed, respawned, peak_row
            obs, _r, term, trunc, info = self.env.step_raw_frame(buttons)
            env_term = env_term or bool(term)
            env_trunc = env_trunc or bool(trunc)
            row = self._read_var(self.row_var, None)
            kind = self._read_var(self.piece_type_var, None)
            dropped = row is not None and peak_row is not None and row < peak_row and peak_row >= min_row
            changed = kind is not None and initial_type is not None and kind != initial_type
            if dropped or changed:
                if not completed and self.capture_lock:
                    self.lock_frame = self._render_frame()
                completed = True
            respawned = respawned or dropped
            if row is not None:
                peak_row = row if peak_row is None else max(peak_row, row)

        def active_type():
            kind = self._read_var(self.piece_type_var, None)
            return kind is not None and (valid_types is None or kind in valid_types)

        if action is None:
            # Missing candidates are not action zero (which moves and drops).
            tick([])
        else:
            target_rot, target_col = divmod(int(action), self.columns)
            # Rotate first, then read the column again: games may shift the
            # anchor during rotation, and normalized shape origins can differ.
            rotation_plan = MacroPlanGenerator.plan(
                target_rot * self.columns, self.cfg, current_rot, 0)
            for buttons in rotation_plan["frames"]:
                tick(buttons)
                if completed or env_term or env_trunc:
                    break
            if not (completed or env_term or env_trunc):
                offsets = self.cfg.get("piece_col_offsets", {}).get(str(initial_type))
                offset = int(offsets[target_rot]) if offsets is not None else self.col_offset
                raw_col = self._read_var(self.col_var, None)
                current_col = raw_col - offset if raw_col is not None else self.spawn_col
                move_plan = MacroPlanGenerator.plan(action, self.cfg, target_rot, current_col)
                for buttons in move_plan["frames"]:
                    tick(buttons)
                    if completed or env_term or env_trunc:
                        break
            if not (completed or env_term or env_trunc):
                tick([])
            for _ in range(self.max_settle_frames):
                if completed or env_term or env_trunc:
                    break
                tick([self.commit_button])

            # Release controls after lock. If the row already reset, do not wait
            # for a SECOND reset and let the next piece fall unattended.
            if completed and not (env_term or env_trunc):
                for _ in range(int(self.cfg.get("spawn_wait_frames", 30))):
                    tick([])
                    if env_term or env_trunc or (respawned and active_type()):
                        break
            elif not (env_term or env_trunc):
                # A timeout is not a completed placement and must not train on
                # the falling piece's board. End collection without death.
                env_trunc = True

        ready = (action is None or (completed and respawned)) and active_type()
        if hasattr(self.env, "finalize_macro_step"):
            result = self.env.finalize_macro_step(obs, info, env_term, env_trunc)
        else:
            result = obs, 0.0, env_term, env_trunc, info
        obs, reward, term, trunc, info = result
        info = dict(info)
        info["afterstate_ready"] = bool(ready and not term)
        return obs, reward, term, trunc, info
