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


class MacroPlacementController:
    """ONE placement, executed one frame at a time.

    This is the single implementation of how a (rotation, column) decision
    becomes controller input. Training drives it in a loop inside
    MacroPlacementWrapper; live play (play_engine) drives it one frame per
    render loop, interleaved with the human player. Neither owns a copy.

    That matters because the copies DID drift: play_engine kept its own state
    machine, the rotation-specific column re-read landed only in the wrapper,
    and the same checkpoint scored 605 lines through training and 1 line through
    the studio path. The observation was unified for exactly this reason (see
    rl.env.make_observer); this does the same for execution.

    Nothing here is game-specific. Buttons, tap timing, rotation step, column
    offsets, settle detection and spawn waiting all come from the game's
    macro_config, so any game with a "place a piece by rotating and sliding"
    control style can use it by declaring those values.

    Usage (the caller owns the emulator):

        ctrl = MacroPlacementController(cfg, read_var, player)
        ctrl.begin(action)                  # action None = neutral wait
        while (buttons := ctrl.next_buttons()) is not None:
            ...step the emulator one frame with `buttons`...
            if terminated or truncated:
                ctrl.abort(); break
            ctrl.observe()                  # updates lock/respawn detection
    """

    def __init__(self, config, read_var, player=1, on_lock=None):
        cfg = dict(config or {})
        self.cfg = cfg
        self.read = read_var
        self.on_lock = on_lock
        self.rotations = int(cfg.get("rotations", 4))
        self.columns = int(cfg.get("columns", 10))
        self.commit_button = cfg.get("commit_button", "DOWN")
        self.settle_detect = dict(cfg.get("settle_detect") or {})
        self.max_settle_frames = int(cfg.get("max_settle_frames", 600))
        self.spawn_wait_frames = int(cfg.get("spawn_wait_frames", 30))
        self.col_offset = int(cfg.get("col_offset", 3))
        self.spawn_col = int(cfg.get("spawn_col", 4))
        self.min_row = int(self.settle_detect.get("min_previous", 0))
        self.valid_types = cfg.get("valid_piece_types")
        self.rot_var = cfg.get("rot_var", "piece_rot_p%d" % player)
        self.col_var = cfg.get("col_var", "piece_col_p%d" % player)
        self.row_var = self.settle_detect.get("var", cfg.get("row_var", "piece_row_p%d" % player))
        self.piece_type_var = cfg.get("piece_type_var", "piece_type_p%d" % player)
        self._reset_state()

    @classmethod
    def for_wrapper(cls, wrapper, on_lock=None):
        """Build a controller from a wrapper's ALREADY-RESOLVED configuration.

        The wrapper has done the {player} substitution and default resolution,
        so those values win over re-reading the raw config here. This keeps one
        source of truth for the running placement while letting a caller with no
        wrapper (live play) construct one straight from macro_config.
        """
        ctrl = cls(wrapper.cfg, wrapper._read_var,
                   getattr(wrapper, "player", 1), on_lock=on_lock)
        ctrl.rot_var = wrapper.rot_var
        ctrl.col_var = wrapper.col_var
        ctrl.row_var = wrapper.row_var
        ctrl.piece_type_var = wrapper.piece_type_var
        ctrl.commit_button = wrapper.commit_button
        ctrl.max_settle_frames = wrapper.max_settle_frames
        ctrl.col_offset = wrapper.col_offset
        ctrl.spawn_col = wrapper.spawn_col
        ctrl.columns = wrapper.columns
        ctrl.min_row = int(dict(getattr(wrapper, "settle_detect", {}) or {}).get("min_previous", 0))
        return ctrl

    def _reset_state(self):
        self.action = None
        self.completed = False
        self.respawned = False
        self.aborted = False
        self.timed_out = False
        self.lock_frame = None
        self.initial_type = None
        self.peak_row = None
        self._gen = None

    def begin(self, action):
        """Start one placement. `action` None means wait without issuing input."""
        self._reset_state()
        self.action = action
        self.initial_type = self.read(self.piece_type_var, None)
        self.peak_row = self.read(self.row_var, None)
        self._gen = self._frames()

    def abort(self):
        self.aborted = True

    def active_type(self):
        kind = self.read(self.piece_type_var, None)
        return kind is not None and (self.valid_types is None or kind in self.valid_types)

    def observe(self):
        """Update lock/respawn detection. Call after each stepped frame."""
        row = self.read(self.row_var, None)
        kind = self.read(self.piece_type_var, None)
        dropped = (row is not None and self.peak_row is not None
                   and row < self.peak_row and self.peak_row >= self.min_row)
        changed = (kind is not None and self.initial_type is not None
                   and kind != self.initial_type)
        if dropped or changed:
            if not self.completed and self.on_lock is not None:
                self.lock_frame = self.on_lock()
            self.completed = True
        self.respawned = self.respawned or dropped
        if row is not None:
            self.peak_row = row if self.peak_row is None else max(self.peak_row, row)

    def next_buttons(self):
        """Buttons for this frame, or None when the placement is finished."""
        if self._gen is None:
            return None
        try:
            return next(self._gen)
        except StopIteration:
            self._gen = None
            return None

    def ready(self):
        """True when a settled, playable state is available to act on."""
        return ((self.action is None or (self.completed and self.respawned))
                and self.active_type())

    def _replacement_present(self):
        """Is the NEXT piece really on the board?

        Either the row counter reset, or a different valid piece type is now
        current. Requiring the row reset ALONE left the controller emitting
        neutral input while the replacement was already falling: measured over
        400 placements, every one completed by type change rather than row
        reset, and the tail after completion averaged 7.9 frames and reached the
        full spawn_wait_frames. At high gravity that is several rows of
        uncontrolled fall -- the first piece after a level change was the worst
        case, because the row reset that normally releases the wait a frame or
        two later does not arrive cleanly there.

        The wait itself is still needed: handing back before the replacement
        exists made downstream readers see the LOCKED piece's row/column, which
        erased part of the just-placed piece from the masked board.
        """
        if not self.active_type():
            return False
        if self.respawned:
            return True
        kind = self.read(self.piece_type_var, None)
        return (kind is not None and self.initial_type is not None
                and kind != self.initial_type)

    def _stop(self):
        return self.completed or self.aborted

    def _frames(self):
        if self.action is None:
            # A missing decision is a neutral wait, never action zero (which
            # would rotate and drop a piece the caller never chose).
            yield []
            return

        target_rot, _target_col = divmod(int(self.action), self.columns)
        current_rot = self.read(self.rot_var, 0)
        # Rotate FIRST, then read the column again: games may shift the piece
        # anchor during rotation, and normalised shape origins differ per
        # rotation. Planning both from the pre-rotation column is the defect
        # that put vertical pieces one column off.
        rotation = MacroPlanGenerator.plan(target_rot * self.columns, self.cfg,
                                           current_rot, 0)
        for buttons in rotation["frames"]:
            if self._stop():
                break
            yield buttons

        if not self._stop():
            offsets = self.cfg.get("piece_col_offsets", {}).get(str(self.initial_type))
            offset = int(offsets[target_rot]) if offsets is not None else self.col_offset
            raw_col = self.read(self.col_var, None)
            current_col = raw_col - offset if raw_col is not None else self.spawn_col
            movement = MacroPlanGenerator.plan(self.action, self.cfg, target_rot, current_col)
            for buttons in movement["frames"]:
                if self._stop():
                    break
                yield buttons

        if not self._stop():
            yield []                       # release, so the next press is fresh

        frames = 0
        while not self._stop() and frames < self.max_settle_frames:
            frames += 1
            yield [self.commit_button]

        if self.aborted:
            return
        if self.completed:
            # Release after the lock. Step first, THEN test: the replacement
            # piece is only observable after a frame has run. Stop as soon as it
            # is actually there; do not wait for a SECOND row reset.
            for _ in range(self.spawn_wait_frames):
                yield []
                if self.aborted or self._replacement_present():
                    return
        else:
            # A timeout is not a completed placement, and must not be reported
            # as one -- nor as a death.
            self.timed_out = True


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
        """Execute at most one placement. None waits without issuing controls.

        The placement itself is run by the shared MacroPlacementController, so
        training and live play cannot diverge; this method only owns stepping
        the emulator and finalising the env contract.
        """
        ctrl = MacroPlacementController.for_wrapper(
            self, on_lock=self._render_frame if self.capture_lock else None)
        ctrl.begin(action)
        self.lock_frame = None

        obs, info = None, {}
        env_term = env_trunc = False
        while True:
            buttons = ctrl.next_buttons()
            if buttons is None:
                break
            obs, _r, term, trunc, info = self.env.step_raw_frame(buttons)
            env_term = env_term or bool(term)
            env_trunc = env_trunc or bool(trunc)
            if env_term or env_trunc:
                ctrl.abort()
                break
            ctrl.observe()

        # A settle timeout is a truncation, never a completed placement and
        # never a death; the runner must not train on a falling piece's board.
        if ctrl.timed_out:
            env_trunc = True
        self.lock_frame = ctrl.lock_frame
        ready = ctrl.ready()

        if hasattr(self.env, "finalize_macro_step"):
            result = self.env.finalize_macro_step(obs, info, env_term, env_trunc)
        else:
            result = obs, 0.0, env_term, env_trunc, info
        obs, reward, term, trunc, info = result
        info = dict(info)
        info["afterstate_ready"] = bool(ready and not term)
        return obs, reward, term, trunc, info
