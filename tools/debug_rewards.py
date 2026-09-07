#!/usr/bin/env python3
"""
Verify the TRAINING reward pipeline end to end -- no guessing.

Builds the exact same wrapped env stack train.py trains on (same make_env,
same flags) and runs three scripted policies through it:

    stand     -- always no-op
    run       -- always RIGHT+B
    run+jump  -- RIGHT+B with a periodic full forward jump

and prints each one's per-decision shaped rewards and episode totals. What the
totals MUST show if the shaping is healthy:

    run  >>  stand          (progress actually pays)
    run+jump >= run         (jumping over the first obstacle pays even more)

If `stand` scores about the same as `run`, the progress signal is dead inside
the training stack (whatever diagnostics said about raw RAM) and there's a
plumbing bug to hunt. If `run` clearly wins here but a trained agent still
converges to standing, the reward is fine and the problem is optimization
(e.g. value-function blowup from unnormalized returns), which is fixed in
train.py by reward normalization, not by touching the shaping.

Usage -- addresses and weights come from games.json, the same file training
reads, so this exercises exactly the configuration that will train:

    python debug_rewards.py --game SuperMarioBros3-Nes-v0

Any flag still overrides the file, for trying a weight before committing to it:

    python debug_rewards.py --game SuperMarioBros3-Nes-v0 --death-penalty 100
"""
import argparse

from train import ACTION_TABLE, DEFAULT_GAME_CONFIG, load_game_config, make_env

NOOP = 0
RUN_RIGHT = 3        # ["RIGHT", "B"]
JUMP_RIGHT_LONG = 8  # ["RIGHT", "A"] held 20 frames
JUMP_LEFT_LONG = 10  # ["LEFT", "A"] held 20 frames -- the observed exploit:
                     # at the left screen edge this moves nowhere, so it tests
                     # whether pure jump-spam earns anything


def run_policy(env, pick_action, label, max_decisions=600, show_first=40):
    obs, info = env.reset()
    total = 0.0
    rewards = []
    detail = []  # (x, progress_component) per decision, for the tail table
    comp_totals = {}
    x_first = x_last = None
    for i in range(max_decisions):
        obs, r, terminated, truncated, info = env.step(pick_action(i))
        total += r
        rewards.append(r)
        shaping = info.get("shaping", {})
        detail.append((shaping.get("x"), shaping.get("progress", 0.0)))
        # Component ledgers published by RewardShaper / JumpIncentiveWrapper --
        # this is what attributes every point of the total to a specific term.
        for src in ("shaping", "jump_shaping"):
            for key, val in info.get(src, {}).items():
                if key == "x":
                    if val is not None:
                        if x_first is None:
                            x_first = val
                        x_last = val
                elif val:
                    comp_totals[key] = comp_totals.get(key, 0.0) + val
        if terminated or truncated:
            break
    print(f"\n--- {label}: {len(rewards)} decisions, TOTAL shaped reward = {total:+.2f} ---")
    comps = "  ".join(f"{k}={v:+.2f}" for k, v in sorted(comp_totals.items()))
    print(f"components: {comps if comps else '(none fired)'}")
    print(f"x (progress reader) went {x_first} -> {x_last}")
    print(f"first {min(show_first, len(rewards))} per-decision rewards:")
    line = "  "
    for i, r in enumerate(rewards[:show_first]):
        line += f"{r:+7.2f}"
        if (i + 1) % 8 == 0:
            print(line)
            line = "  "
    if line.strip():
        print(line)
    # Tail table: the death sequence lives at the END of the episode, so this
    # is where phantom progress shows itself -- x values swinging while paid
    # progress appears despite the policy not moving.
    tail = detail[-24:]
    start_i = len(detail) - len(tail)
    print(f"last {len(tail)} decisions (i, x, paid progress):")
    print("  " + "  ".join(
        f"[{start_i + j}] x={x if x is not None else '-'} p={p:+.1f}"
        for j, (x, p) in enumerate(tail)))
    return total, len(rewards)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True)
    p.add_argument("--state", default=None)
    p.add_argument("--death-penalty", type=float, default=50.0)
    p.add_argument("--jump-bonus", type=float, default=0.0)
    p.add_argument("--stuck-penalty", type=float, default=0.1)
    p.add_argument("--progress-scale", type=float, default=0.1)
    p.add_argument("--score-bonus", type=float, default=0.01)
    p.add_argument("--life-bonus", type=float, default=25.0)
    p.add_argument("--power-bonus", type=float, default=0.0)
    p.add_argument("--powerup-address", type=lambda v: int(v, 0), default=None)
    p.add_argument("--progress-address", type=lambda v: int(v, 0), default=None)
    p.add_argument("--progress-address-high", type=lambda v: int(v, 0), default=None)
    p.add_argument("--progress-use-info-x", action="store_true")
    p.add_argument("--backtrack-scale", type=float, default=0.5)
    p.add_argument("--survival-tick", type=float, default=0.0)
    p.add_argument("--coin-address", type=lambda v: int(v, 0), default=None)
    p.add_argument("--coin-bonus", type=float, default=0.0)
    p.add_argument("--speed-address", type=lambda v: int(v, 0), default=None)
    p.add_argument("--speed-full", type=lambda v: int(v, 0), default=127)
    p.add_argument("--speed-bonus", type=float, default=0.0)
    p.add_argument("--clear-bonus", type=float, default=0.0)
    p.add_argument("--no-end-on-clear", dest="end_on_clear", action="store_false")
    p.set_defaults(end_on_clear=True)
    p.add_argument("--time-penalty", type=float, default=0.0)
    p.add_argument("--playstate-address", type=lambda v: int(v, 0), default=None)
    p.add_argument("--playstate-value", type=lambda v: int(v, 0), default=None)
    p.add_argument("--no-end-on-death", dest="end_on_life_loss", action="store_false", help="Match train.py: don't end the episode at a lost life.")
    p.set_defaults(end_on_life_loss=True)
    p.add_argument("--game-config", default=DEFAULT_GAME_CONFIG, help="Same games.json training reads. Its values become the defaults here so this gate exercises EXACTLY the training configuration; flags still override. Set to '' to ignore it.")

    # Load the game's config the same way train.py does, so the two cannot
    # drift apart -- a gate that tests different weights than the run is worse
    # than no gate at all.
    prelim, _ = p.parse_known_args()
    config_defaults, _vars = load_game_config(prelim.game_config, prelim.game)
    if config_defaults:
        known = {a.dest for a in p._actions}
        applied = {k: v for k, v in config_defaults.items() if k in known}
        p.set_defaults(**applied)
        print(f"Loaded {len(applied)} settings for {prelim.game} from {prelim.game_config}")
    args = p.parse_args()

    env = make_env(
        args.game, args.state, args.death_penalty, args.jump_bonus,
        stuck_penalty=args.stuck_penalty,
        progress_scale=args.progress_scale, score_bonus=args.score_bonus,
        time_penalty=args.time_penalty, life_bonus=args.life_bonus,
        power_bonus=args.power_bonus, powerup_address=args.powerup_address,
        progress_address=args.progress_address,
        progress_address_high=args.progress_address_high,
        progress_use_info_x=args.progress_use_info_x,
        backtrack_scale=args.backtrack_scale,
        survival_tick=args.survival_tick,
        coin_address=args.coin_address, coin_bonus=args.coin_bonus,
        speed_address=args.speed_address, speed_full=args.speed_full,
        speed_bonus=args.speed_bonus,
        playstate_address=args.playstate_address,
        playstate_value=args.playstate_value,
        end_on_life_loss=args.end_on_life_loss,
        clear_bonus=args.clear_bonus, end_on_clear=args.end_on_clear,
    )()

    print(f"Action indices used: NOOP={NOOP} {ACTION_TABLE[NOOP]}, "
          f"RUN={RUN_RIGHT} {ACTION_TABLE[RUN_RIGHT]}, "
          f"JUMP={JUMP_RIGHT_LONG} {ACTION_TABLE[JUMP_RIGHT_LONG]}")

    stand_total, stand_n = run_policy(env, lambda i: NOOP, "STAND (always no-op)")
    run_total, run_n = run_policy(env, lambda i: RUN_RIGHT, "RUN (always RIGHT+B)")
    runjump_total, runjump_n = run_policy(
        env, lambda i: JUMP_RIGHT_LONG if i % 6 == 5 else RUN_RIGHT,
        "RUN+JUMP (RIGHT+B, jump every 6th)")
    jumpleft_total, jumpleft_n = run_policy(
        env, lambda i: JUMP_LEFT_LONG,
        "JUMP-LEFT (the observed left-edge jump-spam exploit)")
    env.close()

    print("\n================= SUMMARY =================")
    print(f"  STAND     : {stand_total:+9.2f}  over {stand_n} decisions")
    print(f"  RUN       : {run_total:+9.2f}  over {run_n} decisions")
    print(f"  RUN+JUMP  : {runjump_total:+9.2f}  over {runjump_n} decisions")
    print(f"  JUMP-LEFT : {jumpleft_total:+9.2f}  over {jumpleft_n} decisions")
    # Per-decision, not totals: these policies run for very different numbers of
    # decisions, and one that dies sooner accrues less stuck-penalty bleed -- so
    # on totals alone jump-spam can look better than standing while earning
    # nothing at all.
    st_rate = stand_total / max(1, stand_n)
    jl_rate = jumpleft_total / max(1, jumpleft_n)
    print(f"  per decision: STAND {st_rate:+.2f}   RUN {run_total/max(1, run_n):+.2f}"
          f"   RUN+JUMP {runjump_total/max(1, runjump_n):+.2f}   JUMP-LEFT {jl_rate:+.2f}")
    print("===========================================")

    if jl_rate > st_rate + 0.05:
        print("\nWARNING: JUMP-LEFT earns more PER DECISION than STAND -- some")
        print("per-action bonus is still farmable at the left edge. Check")
        print("--jump-bonus is 0 and that no term pays for motionless actions.")

    if run_total > stand_total + 20:
        print("\nVERDICT: progress pays and standing loses -- the reward pipeline is")
        print("healthy. If a trained agent still converges to standing, the problem")
        print("is optimization stability, not shaping (train.py's reward")
        print("normalization addresses that).")
    else:
        print("\nVERDICT: RUN did not clearly beat STAND -- the progress signal is")
        print("NOT paying inside the training stack. Paste this output to Claude;")
        print("that's a plumbing bug (check --progress-address was passed and that")
        print("the RAM read works in this wrapper stack).")


if __name__ == "__main__":
    main()
