# NES Play & Record

> **Making videos?** `studio.py` is the whole pipeline — play a level, get
> captioned 16:9 and 9:16 cuts, a clean HD master, spoken commentary and the
> upload copy, in one self-contained folder. See [CLAUDE.md](CLAUDE.md) for how
> it fits together and what not to break. The training half below is on hold.


Plays an NES game (or anything else stable-retro supports) with an agent,
then automatically exports a finished MP4 with video and audio.

**Environment:** built for Ubuntu/Debian-based Linux -- run it under WSL2
on Windows now, and the exact same steps work unchanged on an AWS EC2
Ubuntu instance later. Commands below are plain bash.

It works by leaning on stable-retro's built-in replay system rather than
screen/audio grabbing by hand:

1. `play_and_record.py` runs the episode and writes a tiny `.bk2` replay
   file (just the button presses -- a few KB).
2. It then calls stable-retro's own `playback_movie` tool, which re-runs
   that replay through the emulator and encodes it straight to `.mp4` via
   ffmpeg, with audio synced automatically.

This means the export is exact -- it's the emulator re-rendering the real
run, not a screen capture -- and it works for **any** game you've imported,
not just NES.

## 0. Get onto Ubuntu (WSL2)

If you're setting this up fresh on Windows, install WSL2 + Ubuntu first --
see the step-by-step walkthrough for that. Everything from here on runs
inside the Ubuntu shell, not PowerShell.

Work out of your Linux home directory (e.g. `~/projects/nes-play-record`),
not `/mnt/c/...` -- I/O across the Windows/WSL bridge is noticeably slower.

## 1. Install system dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip build-essential cmake ffmpeg git \
    libglu1-mesa libgl1 freeglut3-dev mesa-utils
```

`ffmpeg` is what actually encodes the final video. The `libglu1-mesa` /
`libgl1` / `freeglut3-dev` / `mesa-utils` group is OpenGL support --
`--human` needs it to open its play window (agent play without `--human`
or `--render` doesn't). Confirm ffmpeg is there:

```bash
ffmpeg -version
```

## 2. Install Python dependencies

```bash
bash setup.sh
source venv/bin/activate
```

`setup.sh` creates a `venv` and installs everything in `requirements.txt`.
Run `source venv/bin/activate` again any time you open a new shell.

## 3. Import your ROM

stable-retro doesn't ship with commercial ROMs -- you need to point it at
one you already own a legal copy of:

```bash
python -m stable_retro.import /path/to/your/ROMs/
```

This scans the folder (including inside zip files), matches ROMs by hash
against stable-retro's known game integrations, and copies matches into
place.

Check what's available/imported:

```bash
python list_games.py
```

Game IDs generally look like `SuperMarioBros-Nes`, `Contra-Nes`,
`Zelda-Nes`, etc.

## 4. Play it yourself, or let an agent play

Same script, one flag decides which:

**You play it (keyboard, window opens, recorded automatically):**

```bash
python play_and_record.py --game SuperMarioBros3-Nes-v0 --human
```

A window opens, you play with the keyboard, and closing the window (or
hitting game over) automatically renders your session to `.mp4`. This
needs a real display -- see "Getting a display in WSL2" below if nothing
opens.

**An agent plays it for you (headless, no window needed):**

No trained model (random actions) -- works immediately on any game:

```bash
python play_and_record.py --game SuperMarioBros-Nes
```

**With a trained Stable-Baselines3 PPO model** (from the training setup we
discussed earlier -- this is what actually gives you *good* play instead of
flailing):

```bash
python play_and_record.py --game SuperMarioBros-Nes --model ppo_mario.zip
```

**Useful flags:**

| Flag | Purpose |
|---|---|
| `--state some-real-state-name` | Start from a specific save state instead of the game's default -- see "Finding valid state names" below before using this, don't guess |
| `--max-steps 10800` | Safety cap in emulator frames (60fps NES -> 10800 = ~3 min). Episode still ends early on game over/win. |
| `--record-dir ./recordings` | Where the `.bk2` and final `.mp4` land |
| `--render` | Pop up a live window while an agent plays (slower, lets you watch) -- needs a display, see below |
| `--scale 4` | Upscale factor for the final video (default `4`). NES renders at ~256x224 -- stable-retro's own export tool has no scaling option, so this is a second pass that fixes it. Set to `1` to skip and keep the tiny native-resolution file. |
| `--scale-mode sharp` | `sharp` (default) = crisp nearest-neighbor, the classic retro-pixel look. `smooth` = anti-aliased lanczos, softer and less blocky. |

Output: an `.mp4` file in `--record-dir`, printed at the end of the run. With scaling on (the default), you'll see two files -- `<name>.mp4` (native resolution) and `<name>_HD.mp4` (upscaled) -- the printed "Done!" path is the one to use.

**Worth knowing:** upscaling doesn't add real detail -- the NES only ever rendered ~256x224 pixels, full stop. What `--scale` controls is *how* that gets stretched to fill a modern screen: nearest-neighbor keeps hard pixel edges (what you want for anything meant to look "retro"), instead of leaving it to whatever blurry default scaling your video player or a platform's re-encode applies.

**Note:** the script runs fully headless by default (no `--render`, no
`--human`) -- no display needed, which is what makes agent play work on a
plain WSL2 or EC2 box with no GUI at all. `--human` and `--render` both
need an actual display, since you're meant to see/control the game live.

## Getting a display in WSL2

Both `--human` and `--render` open a real window, so you need WSL2 to be
able to show one. Check whether that already works:

```bash
sudo apt install -y x11-apps
xeyes
```

If a little eyeball window pops up on your Windows desktop, you're done --
skip the rest of this section.

**If nothing happens or you get a "cannot open display" error:** you're on
WSLg (default on Windows 11 and updated Windows 10), it's just out of
date. From an elevated **Windows** PowerShell (not inside Ubuntu):

```powershell
wsl --update
wsl --shutdown
```

Reopen Ubuntu and try `xeyes` again.

**If that still doesn't work** (older Windows 10 without WSLg support):
install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) on Windows,
launch it via XLaunch with "Disable access control" checked, then inside
Ubuntu:

```bash
echo 'export DISPLAY=$(grep nameserver /etc/resolv.conf | awk "{print \$2}"):0' >> ~/.bashrc
source ~/.bashrc
xeyes
```

Once `xeyes` shows a window, `--human` and `--render` will too.

## 5. Generic RL Architecture & Training

The framework provides a generic, configurable reinforcement learning pipeline designed to work across **any game genre** in the retro library (Platformers, Racers, Shooters, Puzzle/Grid games, Turn-Based RPGs, and Board games).

The architecture separates **How the Agent Interacts** (`action_mode`) from **How the Agent Thinks** (`algorithm`):

```
                  ┌──────────────────────────────────────────────────────────┐
                  │                      GAMES.JSON                          │
                  │  Specifies the contract between Game, Agent, and Control │
                  └─────────────┬──────────────────────────────┬─────────────┘
                                │                              │
                1. CONTROL LAYER                               2. BRAIN LAYER
                ("action_mode")                                ("algorithm")
        ┌───────────────────────┴──────────────┐       ┌───────┴───────────────────────┐
        ▼                                      ▼       ▼                               ▼
  button_stream                         macro_action  ppo                         afterstate
(Frame-by-frame                     (High-level goal  (Model-free actor-critic   (Evaluates projected
 button presses)                     executed over     learning directly from     post-action states
                                     multiple frames)  observations)              via simulator)
```

---

### The Two Dimensions

#### 1. Control Layer (`action_mode`)
* **`button_stream`**: The agent selects a low-level controller combo (e.g. `["A", "RIGHT"]` or `["NOOP"]`) every frame or frameskip. Ideal for reflex and physics-heavy games (Platformers, Racers, Fighting games, Shooters).
* **`macro_action` / `macro_placement`**: The agent outputs an abstract decision index (e.g. target placement `(rotation, column)` or menu selection). A state-machine wrapper intercepts the intent, coordinates multi-frame button taps and holds on the controller, and yields control only when the maneuver settles. Ideal for puzzle, tile, board, and turn-based games.

#### 2. Decision Layer (`algorithm`)
* **`ppo` (Model-Free Actor-Critic)**: Uses Stable-Baselines3 PPO. Maps camera pixels or RAM features directly to action probabilities $\pi(A_t \mid S_t)$ through trial-and-error reward learning. Use whenever environment forward simulation is intractable.
* **`afterstate` (Model-Based Lookahead)**: Uses a registered Python game simulator (`rl/simulators/`) to project all legal moves one decision step forward ($S' = \text{transition}(S, A)$). A Deep Neural Value Network $V_\theta(S')$ scores the prospective post-action states, picking $\arg\max_A [R_{\text{imm}} + \gamma V_\theta(S')]$. Temporal Difference (TD) learning trains the value network. This disentangles the agent's choice from environmental stochasticity (random future spawns/draws).

---

### Game Taxonomy Matrix

| Archetype | Typical Genres | Control Mode (`action_mode`) | Decision Algorithm (`algorithm`) | Example Games |
| :--- | :--- | :--- | :--- | :--- |
| **Real-Time Reflex** | Platformers, Racers, Shooters | `button_stream` | `ppo` | *Super Mario Bros*, *F-Zero*, *Contra* |
| **Discrete Puzzle & Tile** | Drop puzzles, match-3, board games | `macro_placement` | `afterstate` | *Tetris*, *Dr. Mario*, *Puyo Puyo*, *Columns* |
| **Turn-Based Strategy** | RPGs, tactical strategy, card games | `macro_action` | `ppo` or `afterstate` | *Pokemon*, *Final Fantasy*, *Othello* |
| **Frame-Data Combat** | Fighting games | `button_stream` or `macro_action` | `ppo` | *Street Fighter II*, *Mortal Kombat* |

---

### Unified Training with `train.py`

Training is completely driven by `games.json`. Running `train.py` automatically detects which algorithm is configured for that game:

```bash
# 1. Train a model-free PPO game (e.g. platformer)
python train.py --game SuperMarioBros3-Nes-v0 --timesteps 1000000 --save-every 50000

# 2. Train an afterstate lookahead game (e.g. puzzle game across multiple start states)
python train.py --game TetrisTime-Nes-v0 \
  --state level0_2p,level8_2p,rs_01,rs_02,rs_03,rs_04 \
  --timesteps 3000000 \
  --lr 2.5e-4 \
  --ent-coef 0.20 \
  --ent-coef-final 0.01 \
  --save-every 100000 \
  --device cuda
```

**Common Flags for `train.py`:**

| Flag | Purpose |
| :--- | :--- |
| `--game <id>` | The game integration ID (e.g. `TetrisTime-Nes-v0`, `SuperMarioBros3-Nes-v0`). |
| `--timesteps <N>` | Total environment decisions to train across all episodes. |
| `--state <list>` | Single state name or comma-separated list of save states to sample randomly per episode. |
| `--lr <float>` | Learning rate (default: `2.5e-4`). |
| `--gamma <float>` | Discount factor $\gamma$ (default: `0.99`). |
| `--ent-coef <float>` | Exploration parameter (entropy coefficient for PPO, initial $\epsilon$ for Afterstate). |
| `--ent-coef-final <float>` | Final exploration value to smoothly anneal toward over the training duration. |
| `--save-dir <path>` | Destination folder for checkpoint files (defaults to `checkpoints/<game>/`). |
| `--save-every <N>` | Save a `.zip` checkpoint every $N$ decisions. |
| `--resume <ckpt.zip>` | Resume training seamlessly from an existing checkpoint `.zip`. |
| `--describe` | Inspect the game's full configuration, action table, and reward terms from `games.json` and exit. |
| `--explain-reward <N>` | Step $N$ random decisions in the environment and print which reward terms pay what in real-time. |
| `--set key=value` | Override any parameter from `games.json` on the fly without editing the file (e.g. `--set algorithm=ppo`). |

---

### Universal Checkpoint & Playback Format (`.zip`)

All trained checkpoints are saved in the universal `.zip` format:
* **PPO Archives**: contain `policy.pth`, `data`, and SB3 metadata.
* **Afterstate Archives**: contain `value_net.pth` and model configuration metadata.

Both **`studio.py`** and **`play_engine.py`** automatically inspect the `.zip` archive on startup and instantiate the matching policy pipeline:

```bash
# Play live in Pygame, record .bk2, and generate captioned videos with studio.py:
python studio.py --game TetrisTime-Nes-v0 --model checkpoints/TetrisTime-Nes-v0/ckpt_500000_steps.zip
```

---

### Extensible Simulator Plugin Contract

To add Afterstate Lookahead support for any new puzzle, grid, or strategy game, implement a single simulator class in `rl/simulators/`:

```python
from rl.simulators import register_simulator, BaseSimulator

@register_simulator("my_game_sim")
class MyGameSimulator(BaseSimulator):
    feature_dim: int = 16  # Dimensions of the state descriptor

    def get_candidates(self, obs, ram, info, vars, spec):
        """
        Enumerate all legal candidate moves from the current board state.
        Returns a list of tuples:
            (action_index, prospective_feature_vector, immediate_reward)
        """
        ...
```

Then in `games.json`, simply declare:
```json
"algorithm": "afterstate",
"action_mode": "macro_placement",
"afterstate": {
  "simulator": "my_game_sim"
}
```
The entire framework (`train.py`, `studio.py`, `play_engine.py`) immediately supports the game with zero additional changes.

---

### Human-coach pretraining (warm-start from your own play)

If pure RL keeps converging on "run into the first obstacle and die,"
`pretrain_imitation.py` lets you show it the right idea directly instead
of waiting for it to stumble onto good play by chance:

```bash
# 1. Record yourself playing 1-2 levels
python play_and_record.py --game SuperMarioBros3-Nes-v0 \
    --human --record-dir ./human_demos

# 2. Train the network to imitate what you did
python pretrain_imitation.py --game SuperMarioBros3-Nes-v0 \
    --demo-dir ./human_demos --epochs 25 \
    --output ./checkpoints/SuperMarioBros3-Nes-v0/pretrained_human_bc.zip

# 3. Fine-tune with RL from there, gently so it doesn't unlearn the demo
python train.py --game SuperMarioBros3-Nes-v0 \
    --resume-from ./checkpoints/SuperMarioBros3-Nes-v0/pretrained_human_bc.zip \
    --lr 3e-5 --ent-coef 0.001 --iterations 300
```

This walks your recording frame-by-frame and detects how long you
actually held each button, matching it against `ACTION_TABLE`'s
short/long entries -- a quick tap of jump becomes a different training
label than a long hold, matching how you actually played it, not a
sampled/averaged approximation of it. `--jump-weight` (default `8.0`)
upweights jump examples in the training loss, since running dominates
typical demo data (often 85%+ of frames) while jumping is comparatively
rare -- without this, imitation learning mostly just learns to run.

Note: this makes training no longer "from scratch" in the strict sense
-- the network starts from what you demonstrated rather than random
weights. That's a deliberate tradeoff (faster to competent play, less
purely emergent), not an accident; skip this step entirely if you want
the from-scratch guarantee instead.

### Finding valid state names

"State" means a save point stable-retro can start from -- e.g. the
beginning of a specific level, instead of power-on/title screen. Each
game's integration ships its own set of state files with its own names
chosen by whoever built that integration -- there's no universal naming
scheme, so `--state SomeGuessedName` can fail even when it looks
reasonable. Ask stable-retro directly instead of guessing:

```bash
python3 -c "
import stable_retro as retro, os
rom = retro.data.get_romfile_path('SuperMarioBros3-Nes-v0', retro.data.Integrations.STABLE)
data_dir = os.path.dirname(rom)
print('Integration folder:', data_dir)
print()
print('Files in it:')
for f in sorted(os.listdir(data_dir)):
    print(' ', f)
"
```

Swap in your own game id if different. This lists everything stable-retro
actually has for that game. Example of what this might print:

```
Integration folder: /home/reza/RetroGames/venv/lib/python3.14/site-packages/stable_retro/data/stable/SuperMarioBros3-Nes-v0

Files in it:
  SuperMarioBros3-Nes-v0.state
  data.json
  metadata.json
  rom.sha
  scenario.json
```

Any file ending in `.state` (or `.state.gz`) is a valid `--state` value
-- use the filename with that extension stripped. In this example
there's only *one* state file, named after the game itself. That's a
completely normal, common setup: it means this particular integration
doesn't ship separate per-level states at all -- there's just the one
built-in starting point (typically the very beginning of the game). If
your listing looks like that, `--state` has nothing else to point to,
and simply omitting the flag (using the default state) is correct and
complete -- not a workaround, the actual answer. Only integrations built
for specific research use (Sonic's contest states are the well-known
example) tend to ship many labeled per-level states.

**Reward signal:** stable-retro drives reward/episode-end off of a
"scenario" file bundled with each game's integration (usually built from
score, since that's the standard pattern across the shipped
integrations), and `train.py`'s own reward shaping (above) reads `info`
values from that same integration. If both seem to go nowhere (flat
reward, no visible improvement across checkpoints), that's the first
thing worth checking -- via the integration UI (`Game > Data` menu)
rather than assuming the training code is broken.

## 6. Turn checkpoints into a progress-reel video

Once you've got checkpoints, `make_progress_reel.py` plays a short clip
from each one (using the exact same agent-play + bk2-to-mp4 pipeline as
`play_and_record.py`), labels each clip with its iteration number, and
assembles two finished, edited videos in one run:

```bash
python make_progress_reel.py --game SuperMarioBros3-Nes-v0 \
    --checkpoint-dir ./checkpoints --iterations 1 100
```

`--iterations` here must match milestone checkpoints that actually exist
for this game (the default `1 100` matches `train.py`'s own default run
length -- adjust both together if you change `--checkpoint-iterations`
when training).

Output in `./progress_reels/`:

- `progress_youtube.mp4` -- 1920x1080 landscape, `--clip-seconds` (default `8`) per checkpoint
- `progress_shorts.mp4` -- 1080x1920 portrait, `--clip-seconds-short` (default `4`) per checkpoint -- Instagram/TikTok, also works as a YouTube Short

With the default 4 checkpoints that's 32s and 16s total respectively --
well within any platform's limits. Adjust `--clip-seconds` /
`--clip-seconds-short` to taste, or pass a different `--iterations` list
(it just needs to match checkpoint filenames that actually exist in
`--checkpoint-dir`).

Text overlays use the DejaVu Sans font -- add it if you haven't already:

```bash
sudo apt install -y fonts-dejavu-core
```

## Moving to AWS EC2 later

Because everything above is stock Ubuntu, `setup.sh` and the apt-install
line work unchanged on an EC2 Ubuntu instance -- copy the project over
(`git clone` or `scp`) and rerun steps 1-4 as-is.

The one thing that differs: GPU driver setup.
- **WSL2** uses the Windows-side NVIDIA driver plus the WSL-specific CUDA
  toolkit (see the setup walkthrough).
- **EC2** uses the standard Linux NVIDIA driver, or comes with it
  preinstalled if you pick an AWS Deep Learning AMI.

Also worth knowing before you plan a move purely for more power: EC2
doesn't offer RTX 4090 instances (that's a consumer card). The closest
GPU options for training are `g5`/`g6` instances (A10G/L4) or `p4`/`p5`
(A100/H100) for something well beyond a 4090.

## Notes

- **Play quality without `--model` will look random** -- that's expected,
  since there's no trained policy behind it, just random button mashing.
  For a run that actually clears levels, train an agent first (`train.py`)
  and pass the resulting `.zip` as `--model`.
- **Works for any imported game, any console stable-retro supports** --
  nothing here is NES-specific except the flag examples.
- If `playback_movie` fails, it's almost always ffmpeg missing from PATH --
  the `.bk2` replay is still saved either way, so nothing is lost.
- `train.py`'s default action set (`DEFAULT_COMBOS`) is tuned for
  platformer-style games. For a very different genre (a fighting game, a
  puzzle game), edit that list to match -- it looks up button names
  dynamically, so any valid NES button name works.
