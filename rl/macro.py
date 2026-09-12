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
    def plan(action, config, current_rot=0, current_col=None):
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
        col_diff = target_col - current_col
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
        target_rot = int(action) // self.columns
        target_col = int(action) % self.columns

        current_rot = self._read_var(self.rot_var, 0)
        col_raw = self._read_var(self.col_var, None)
        current_col = (col_raw - self.col_offset) if col_raw is not None else self.spawn_col
        initial_piece_type = self._read_var(self.piece_type_var, None)

        plan = MacroPlanGenerator.plan(
            action, self.cfg, current_rot=current_rot, current_col=current_col
        )

        obs = None
        info = {}
        env_term = env_trunc = False

        # Phase 1: Execute rotation and horizontal movement frames
        #
        # Stop early if the piece this plan was made for is no longer the one
        # falling. The taps run up to 32 frames (measured: 3 rotations plus 9
        # columns), and a piece falls a row every ~7 frames at level 8, so once
        # the stack is high enough that a piece locks after 4 rows the taps
        # outlast it -- and every remaining tap then moves the NEXT piece,
        # which this action was not chosen for. It cannot happen on a low board
        # (a piece takes ~125 frames to fall from spawn) which is why it went
        # unseen, but the high board is exactly where placement matters.
        for btn_list in plan["frames"]:
            obs, _r, term, trunc, info = self.env.step_raw_frame(btn_list)
            if term or trunc:
                env_term = env_term or bool(term)
                env_trunc = env_trunc or bool(trunc)
                break
            now_type = self._read_var(self.piece_type_var, None)
            if (now_type is not None and initial_piece_type is not None
                    and now_type != initial_piece_type):
                break

        # Neutral frame so subsequent DOWN press is recognized as a fresh press
        if not (env_term or env_trunc):
            obs, _r, term, trunc, info = self.env.step_raw_frame([])
            if term or trunc:
                env_term = env_term or bool(term)
                env_trunc = env_trunc or bool(trunc)

        # Phase 2: Soft drop until piece settles
        if not (env_term or env_trunc):
            settled = False
            prev_row = self._read_var(self.row_var, None)
            frame_count = 0
            while not settled and frame_count < self.max_settle_frames:
                obs, _r, term, trunc, info = self.env.step_raw_frame([self.commit_button])
                frame_count += 1
                if term or trunc:
                    env_term = env_term or bool(term)
                    env_trunc = env_trunc or bool(trunc)
                    settled = True
                    break

                curr_row = self._read_var(self.row_var, None)
                curr_type = self._read_var(self.piece_type_var, None)
                # Settle triggered if piece type changes (new spawn) or row drops
                if curr_type is not None and initial_piece_type is not None and curr_type != initial_piece_type:
                    settled = True
                    break
                if curr_row is not None and prev_row is not None:
                    if curr_row < prev_row and prev_row >= 5:
                        settled = True
                        break
                    prev_row = max(prev_row, curr_row)
                elif curr_row is not None:
                    prev_row = curr_row

            # Wait for the NEXT piece to actually spawn before handing back.
            #
            # Settle fires on the LOCK frame, and the replacement piece appears
            # a frame or two later. One neutral frame was not always enough,
            # and when it was not, every reader downstream saw the LOCKED
            # piece's row and column instead of the new one's. Measured on ~14%
            # of placements: the board mask then erased a 5x4 box in the middle
            # of the stack, deleting 5-7 cells of the piece that was just
            # placed -- so the holes it had created sat under nothing and were
            # not counted, height was understated, and the next step's delta
            # was taken against a board that never existed. The same stale read
            # also made the NEXT plan start from the wrong column and rotation.
            #
            # Also serves as the release frame: no buttons are held here.
            settle_row = self._read_var(self.row_var, None)
            for _ in range(int(self.cfg.get("spawn_wait_frames", 30))):
                obs, _r, term, trunc, info = self.env.step_raw_frame([])
                if term or trunc:
                    env_term = env_term or bool(term)
                    env_trunc = env_trunc or bool(trunc)
                    break
                row_now = self._read_var(self.row_var, None)
                if (row_now is not None and settle_row is not None
                        and row_now < settle_row):
                    break

        # Phase 3: Finalize macro step in GenericRetroEnv (calculates reward, checks game over)
        if hasattr(self.env, "finalize_macro_step"):
            return self.env.finalize_macro_step(obs, info, env_term, env_trunc)
        return obs, 0.0, bool(env_term), bool(env_trunc), info
